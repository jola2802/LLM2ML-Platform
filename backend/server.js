import express from 'express';
import cors from 'cors';
import multer from 'multer';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { pythonClient } from './services/clients/python_client.js';
import { extractMetricsFromOutput } from './services/utils/metrics_extractor.js';
import { cleanupOldPredictScripts } from './services/utils/predict_cache_cleanup.js';

// Service-Imports
import { initializeLogging } from './services/monitoring/log.js';
import {
  initializeDatabase,
  getProject,
  updateProjectTraining,
  updateProjectCode,
  updateProjectInsights,
  extractHyperparametersFromCode
} from './services/database/db.js';
import { masClient } from './services/clients/mas_client.js';

// Route-Imports
import { setupAPIEndpoints, setTrainingFunctions, setupPredictionEndpoint } from './services/api/api_endpoints.js';
import { setupProjectRoutes } from './services/api/routes/projects.js';
import { setupUploadRoutes } from './services/api/routes/upload.js';
import { setupAnalyzeRoutes } from './services/api/routes/analyze.js';
import { setupLLMRoutes } from './services/api/routes/llm.js';
import { setupCacheRoutes } from './services/api/routes/cache.js';
import { setupMonitoringRoutes } from './services/api/routes/monitoring.js';
import { setupQueueRoutes } from './services/api/routes/queue.js';
import { setupFileRoutes } from './services/api/routes/files.js';
import { setupWorkerStatusRoutes } from './services/api/routes/worker_status.js';
import { setupScalingRoutes } from './services/api/routes/scaling.js';
import { setupPredictCacheRoutes } from './services/api/routes/predict_cache.js';
import { setupAgentRoutes } from './services/api/routes/agents.js';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Logging initialisieren
await initializeLogging();

// Express-App initialisieren
const app = express();
const PORT = process.env.PORT || 3000;

// CORS für Frontend - erlaube mehrere Origins für Development und Docker
const allowedOrigins = [
  process.env.FRONTEND_URL,
  'http://localhost:5173',
  'http://localhost:5174',
  'http://localhost:3000',
  'http://127.0.0.1:5173',
  'http://127.0.0.1:5174',
  'http://127.0.0.1:3000',
  'http://frontend:80',
  'http://frontend',
  'http://localhost:80'
].filter(Boolean);

app.use(cors({
  origin: (origin, callback) => {
    // Erlaube Requests ohne Origin (z.B. Postman, curl, Server-to-Server)
    if (!origin) {
      return callback(null, true);
    }

    // Prüfe ob Origin erlaubt ist
    if (allowedOrigins.some(allowed => origin.startsWith(allowed))) {
      callback(null, true);
    } else if (allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      console.log('⚠️ CORS blocked origin:', origin);
      console.log('✅ Allowed origins:', allowedOrigins);
      // In Development: Erlaube alle localhost Origins
      if (origin.includes('localhost') || origin.includes('127.0.0.1')) {
        callback(null, true);
      } else {
        callback(new Error('Not allowed by CORS'));
      }
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));

app.use(express.json());

// Upload-Ordner erstellen falls nicht vorhanden
// Im Docker-Container: /app/uploads, lokal: backend/uploads
const uploadDir = process.env.UPLOADS_DIR || path.join(__dirname, 'uploads');
const modelDir = process.env.MODELS_DIR || path.join(__dirname, 'models');
const scriptDir = process.env.SCRIPT_DIR || path.join(__dirname, 'scripts');

// Python Worker Pool wird jetzt im Python-Service verwaltet
// Keine lokale Instanz mehr nötig

// Job-Completion-Handler werden jetzt über Python-Service Webhooks verwaltet
// Webhook-Endpoint für Job-Completion
app.post('/api/webhooks/job-completed', async (req, res) => {
  try {
    const { jobId, jobType, projectId, result, status } = req.body;

    console.log(`Webhook empfangen: Job ${jobId} (${jobType}) für Projekt ${projectId}, Status: ${status}`);

    if (status === 'completed') {
      if (jobType === 'training') {
        await handleTrainingJobCompleted({
          id: jobId,
          type: jobType,
          data: { projectId },
          result: result || {}
        });
      } else if (jobType === 'retraining') {
        await handleRetrainingJobCompleted({
          id: jobId,
          type: jobType,
          data: { projectId },
          result: result || {}
        });
      }
    } else if (status === 'failed') {
      await handleTrainingJobFailed({
        id: jobId,
        type: jobType,
        data: { projectId },
        error: result?.error || result || 'Unbekannter Fehler'
      });
    }

    res.json({ success: true });
  } catch (error) {
    console.error('Fehler beim Verarbeiten des Job-Completion-Webhooks:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

try {
  await fs.mkdir(uploadDir, { recursive: true });
  await fs.mkdir(modelDir, { recursive: true });
  await fs.mkdir(scriptDir, { recursive: true });
  // console.log('Upload-, Model- und Script-Ordner erstellt');
} catch (error) {
  console.error('Fehler beim Erstellen der Ordner:', error);
}

// Datenbank initialisieren
try {
  await initializeDatabase();
  // console.log('Datenbank initialisiert');
} catch (error) {
  console.error('Fehler bei der Datenbankinitialisierung:', error);
  process.exit(1);
}

// Multer für Datei-Uploads konfigurieren
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    // Extension am Ende beibehalten, Name sicher normalisieren
    const ext = path.extname(file.originalname);
    const base = path.basename(file.originalname, ext).replace(/[^a-zA-Z0-9._-]+/g, '_');
    const uniqueName = `${base}-${Date.now()}-${Math.round(Math.random() * 1E3)}${ext}`;
    cb(null, uniqueName);
  }
});

const upload = multer({
  storage,
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB limit
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['.csv', '.json', '.xlsx', '.xls', '.txt'];
    const fileExt = path.extname(file.originalname).toLowerCase();
    if (allowedTypes.includes(fileExt)) {
      cb(null, true);
    } else {
      cb(new Error('Nur CSV, JSON, Excel und TXT Dateien sind erlaubt'), false);
    }
  }
});

// API-Routen einrichten
setupAPIEndpoints(app, upload, scriptDir, null); // venvDir nicht mehr benötigt
setupProjectRoutes(app, scriptDir, null, trainModelAsync, retrainModelAsync); // venvDir nicht mehr benötigt
setupUploadRoutes(app, upload);
setupAnalyzeRoutes(app);
setupLLMRoutes(app);
setupAgentRoutes(app);
setupCacheRoutes(app);
setupMonitoringRoutes(app);
setupQueueRoutes(app);
setupFileRoutes(app);
// Worker-Status wird jetzt über Python-Service abgefragt
setupWorkerStatusRoutes(app);
setupScalingRoutes(app);
setupPredictCacheRoutes(app, scriptDir);
setupPredictionEndpoint(app, scriptDir, null); // venvDir nicht mehr benötigt

// Health-Check Endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'backend' });
});

// API-Route nicht gefunden
app.get('/api/*', (req, res) => {
  res.status(404).json({ error: 'API-Route nicht gefunden' });
});

// Job-Handler-Funktionen
async function handleTrainingJobCompleted(job) {
  const { projectId } = job.data;

  // Prüfe ob job.result existiert (kann bei fehlgeschlagenen Jobs null sein)
  if (!job.result) {
    console.error(`Job ${job.id} für Projekt ${projectId} hat kein result - möglicherweise fehlgeschlagen`);
    await handleTrainingJobFailed(job);
    return;
  }

  const { output, stderr, metrics } = job.result;

  console.log(`Job-Result für Projekt ${projectId}:`, {
    hasOutput: !!output,
    outputLength: output?.length || 0,
    hasStderr: !!stderr,
    hasMetrics: !!metrics,
    metricsKeys: metrics ? Object.keys(metrics) : []
  });

  try {
    const project = await getProject(projectId);
    if (!project) return;

    // Performance Metriken aus der Ausgabe extrahieren
    let extractedMetrics = metrics || {};
    try {
      // Falls keine Metriken im Result, versuche aus Output zu extrahieren
      if (!extractedMetrics || Object.keys(extractedMetrics).length === 0) {
        extractedMetrics = extractMetricsFromOutput(output || '', project.modelType || 'classification');
      }
      console.log('Verwendete Metriken:', extractedMetrics);
    } catch (metricsError) {
      console.error('Fehler bei Metrik-Extraktion:', metricsError.message);
      extractedMetrics = extractedMetrics || { error: 'Metrik-Extraktion fehlgeschlagen' };
    }

    // Hyperparameter aus dem gespeicherten Python-Code extrahieren
    let hyperparameters = {};
    try {
      if (project.pythonCode) {
        hyperparameters = extractHyperparametersFromCode(project.pythonCode);
      } else {
        hyperparameters = project.hyperparameters || {};
      }
      //console.log('Extrahierte Hyperparameter:', hyperparameters);
    } catch (hyperError) {
      console.error('Fehler bei Hyperparameter-Extraktion:', hyperError.message);
      hyperparameters = project.hyperparameters || {};
    }

    // Model-Datei-Pfad
    const modelPath = `models/model_${project.id}.pkl`;

    // Projekt in DB aktualisieren
    await updateProjectTraining(projectId, {
      status: 'Completed',
      performanceMetrics: extractedMetrics,
      pythonCode: project.pythonCode, // Bereits in DB gespeichert
      originalPythonCode: project.pythonCode,
      modelPath: modelPath,
      hyperparameters: hyperparameters
    });

    console.log(`Projekt ${projectId} erfolgreich als 'Completed' markiert`);

    // Automatische Performance-Evaluation
    try {
      // console.log(`Starte automatische Performance-Evaluation für Projekt: ${project.name}`);
      const performanceInsights = await masClient.evaluatePerformance(await getProject(projectId));
      await updateProjectInsights(projectId, performanceInsights);
      // console.log(`Performance-Evaluation erfolgreich abgeschlossen für Projekt: ${project.name}`);
    } catch (evaluationError) {
      console.error(`Performance-Evaluation fehlgeschlagen für Projekt ${projectId}:`, evaluationError.message);
    }

    // console.log(`Training erfolgreich abgeschlossen für Projekt: ${project.name}`);

  } catch (error) {
    console.error(`Fehler beim Verarbeiten des abgeschlossenen Training-Jobs für Projekt ${projectId}:`, error.message);
  }
}

async function handleRetrainingJobCompleted(job) {
  const { projectId } = job.data;

  // Prüfe ob job.result existiert (kann bei fehlgeschlagenen Jobs null sein)
  if (!job.result) {
    console.error(`Retraining-Job ${job.id} für Projekt ${projectId} hat kein result - möglicherweise fehlgeschlagen`);
    await handleTrainingJobFailed(job);
    return;
  }

  const { output, stderr } = job.result;

  try {
    const project = await getProject(projectId);
    if (!project) return;

    // Performance Metriken aus der Ausgabe extrahieren
    let metrics = {};
    try {
      metrics = extractMetricsFromOutput(output, project.modelType);
      // console.log('Extrahierte Re-Training Metriken:', metrics);
    } catch (metricsError) {
      console.error('Fehler bei Metrik-Extraktion:', metricsError.message);
      metrics = { error: 'Metrik-Extraktion fehlgeschlagen' };
    }

    // Model-Datei pfad (neues Model)
    const modelFileName = `model_${projectId}.pkl`;
    const modelPath = `models/${modelFileName}`;
    const fullModelPath = path.join(__dirname, modelPath);

    // Model-Datei vom Script-Verzeichnis zum models-Verzeichnis verschieben
    const tempModelPath = path.join(scriptDir, 'model.pkl');
    try {
      await fs.access(tempModelPath);
      // Altes Model backup erstellen
      if (project.modelPath) {
        const oldModelPath = path.join(__dirname, project.modelPath);
        const backupPath = path.join(__dirname, `models/backup_${projectId}.pkl`);
        try {
          await fs.rename(oldModelPath, backupPath);
        } catch (backupErr) {
          console.log('Could not backup old model:', backupErr.message);
        }
      }

      await fs.rename(tempModelPath, fullModelPath);
      // console.log(`Re-trained model erfolgreich verschoben: ${fullModelPath}`);
    } catch (err) {
      console.log('Could not move retrained model file:', err.message);
    }

    // Hyperparameter aus dem Python-Code extrahieren
    const hyperparameters = extractHyperparametersFromCode(project.pythonCode);

    // Projekt in DB aktualisieren
    await updateProjectTraining(projectId, {
      status: 'Completed',
      performanceMetrics: metrics,
      pythonCode: project.pythonCode, // Python-Code bereits in DB
      modelPath: modelPath,
      hyperparameters: hyperparameters
    });

    // Automatische Performance-Evaluation
    try {
      const performanceInsights = await masClient.evaluatePerformance(await getProject(projectId));
      await updateProjectInsights(projectId, performanceInsights);
      // console.log(`Performance-Evaluation nach Re-Training erfolgreich abgeschlossen für Projekt: ${project.name}`);
    } catch (evaluationError) {
      console.error(`Performance-Evaluation nach Re-Training fehlgeschlagen für Projekt ${projectId}:`, evaluationError.message);
    }

    // console.log(`Re-Training erfolgreich abgeschlossen für Projekt: ${project.name}`);

  } catch (error) {
    console.error(`Fehler beim Verarbeiten des abgeschlossenen Re-Training-Jobs für Projekt ${projectId}:`, error.message);
  }
}

async function handleTrainingJobFailed(job) {
  const { projectId } = job.data;

  try {
    const project = await getProject(projectId);
    const status = job.type === 'training' ? 'Failed' : 'Re-training Failed';

    await updateProjectTraining(projectId, {
      status: status,
      performanceMetrics: { error: job.error },
      pythonCode: project?.pythonCode || '',
      originalPythonCode: project?.originalPythonCode || '',
      modelPath: project?.modelPath || '',
      hyperparameters: project?.hyperparameters || {}
    });

    // console.log(`Projekt ${projectId} als fehlgeschlagen markiert: ${job.error}`);

  } catch (error) {
    console.error(`Fehler beim Verarbeiten des fehlgeschlagenen Training-Jobs für Projekt ${projectId}:`, error.message);
  }
}

// Training-Funktionen den API-Endpoints zur Verfügung stellen
setTrainingFunctions(trainModelAsync, retrainModelAsync);

// Asynchrone Trainingsfunktion mit Worker Pool
async function trainModelAsync(projectId) {
  try {
    const project = await getProject(projectId);
    if (!project) return;

    // Status auf "Training" setzen
    await updateProjectTraining(projectId, {
      status: 'Training',
      performanceMetrics: {},
      pythonCode: project?.pythonCode || '',
      originalPythonCode: project?.pythonCode || '',
      modelPath: project?.modelPath || '',
      hyperparameters: project?.hyperparameters || {}
    });

    console.log(`🚀 Starte Training für Projekt ${projectId} - Pipeline wird Code generieren und Training direkt starten`);

    // Pipeline aufrufen - diese generiert Code und startet Training direkt
    try {
      const pipelineResponse = await masClient.runAgentPipeline(project);

      // Pipeline gibt zurück: { result: pythonCode, jobId: ..., projectId: ..., status: 'training_started' }
      const pythonCode = pipelineResponse.result || pipelineResponse;
      const jobId = pipelineResponse.jobId;

      if (pythonCode && typeof pythonCode === 'string') {
        // Speichere den generierten Code in der DB
        await updateProjectCode(projectId, pythonCode);
        console.log(`✅ Python-Code in DB gespeichert (${pythonCode.length} Zeichen)`);
      }

      if (jobId) {
        console.log(`✅ Training-Job ${jobId} wurde direkt im mas-service gestartet`);
        console.log(`⏳ Warte auf Training-Abschluss (Webhook wird benachrichtigen)`);
      } else {
        console.log(`⚠️ Keine Job-ID zurückgegeben, aber Pipeline erfolgreich`);
      }

    } catch (pipelineError) {
      console.error(`❌ Pipeline-Fehler: ${pipelineError.message}`);
      await updateProjectTraining(projectId, {
        status: 'Failed',
        performanceMetrics: { error: `Pipeline fehlgeschlagen: ${pipelineError.message}` },
        pythonCode: project?.pythonCode || '',
        originalPythonCode: project?.pythonCode || '',
        modelPath: project?.modelPath || '',
        hyperparameters: project?.hyperparameters || {}
      });
    }

  } catch (error) {
    console.error(`Fehler beim Starten des Trainings für Projekt ${projectId}:`, error.message);

    try {
      await updateProjectTraining(projectId, {
        status: 'Failed',
        performanceMetrics: { error: error.message },
        pythonCode: project?.pythonCode || '',
        originalPythonCode: project?.pythonCode || '',
        modelPath: project?.modelPath || '',
        hyperparameters: project?.hyperparameters || {}
      });
    } catch (dbError) {
      console.error('Fehler beim Markieren des Projekts als fehlgeschlagen:', dbError.message);
    }
  }
}

// Re-Training mit bearbeitetem Code (mit Worker Pool)
async function retrainModelAsync(projectId, customPythonCode) {
  try {
    const project = await getProject(projectId);
    if (!project) return;

    // Bearbeiteten Python-Code in der DB aktualisieren
    await updateProjectCode(projectId, customPythonCode);

    // Job über Python-Service starten
    const result = await pythonClient.startRetraining(projectId, customPythonCode);
    const jobId = result.jobId;

  } catch (error) {
    console.error(`Fehler beim Re-Training für Projekt ${projectId}:`, error.message);

    try {
      await updateProjectTraining(projectId, {
        status: 'Re-training Failed',
        performanceMetrics: { error: error.message },
        pythonCode: customPythonCode || project?.pythonCode || '',
        originalPythonCode: project?.originalPythonCode || '',
        modelPath: project?.modelPath || '',
        hyperparameters: project?.hyperparameters || {}
      });
    } catch (dbError) {
      console.error('Fehler beim Markieren des Re-Trainings als fehlgeschlagen:', dbError.message);
    }
  }
}

// Automatische Bereinigung alter Predict-Skripte (alle 24 Stunden)
setInterval(async () => {
  try {
    const cleaned = await cleanupOldPredictScripts(scriptDir, 168); // 7 Tage
    if (cleaned > 0) {
      console.log(`Automatische Bereinigung abgeschlossen: ${cleaned} Skripte entfernt`);
    }
  } catch (error) {
    console.error('Fehler bei automatischer Bereinigung:', error.message);
  }
}, 24 * 60 * 60 * 1000); // 24 Stunden

// Initiale Bereinigung beim Server-Start (nach 60 Sekunden)
setTimeout(async () => {
  try {
    const cleaned = await cleanupOldPredictScripts(scriptDir, 168);
    if (cleaned > 0) {
      console.log(`Initiale Bereinigung abgeschlossen: ${cleaned} Skripte entfernt`);
    }
  } catch (error) {
    console.error('Fehler bei initialer Bereinigung:', error.message);
  }
}, 60000); // 60 Sekunden nach Server-Start

// Server starten - auf 0.0.0.0 binden, damit er von außen erreichbar ist
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Backend läuft auf Port ${PORT}`);
  console.log(`API: http://localhost:${PORT}/api`);
  console.log(`Health: http://localhost:${PORT}/health`);
});
