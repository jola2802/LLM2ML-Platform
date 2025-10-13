import express from 'express';
import cors from 'cors';
import multer from 'multer';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { PythonWorkerPool } from './services/execution/python_worker_pool.js';
import { jobQueue } from './services/monitoring/job_queue.js';

// Service-Imports
import { initializeLogging } from './services/monitoring/log.js';
import { 
  validatePythonCodeWithLLM,
  executePythonScript,
  generatePredictionScript,
  extractMetricsFromOutput,
  cleanupOldPredictScripts
} from './services/execution/code_exec.js';
import { 
  initializeDatabase,
  getProject,
  updateProjectTraining,
  updateProjectCode,
  updateProjectInsights,
  extractHyperparametersFromCode
} from './services/database/db.js';
import { evaluatePerformanceWithLLM } from './services/llm/api/llm_api.js';
import { runNetworkAgentPipeline } from './services/llm/agents/10_master_agent.js';

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
const PORT = process.env.PORT || 3001;

// CORS für Frontend (React läuft auf Port 5173)
app.use(cors({
  origin: 'http://localhost:5173',
  credentials: true
}));

app.use(express.json());

// Upload-Ordner erstellen falls nicht vorhanden
const uploadDir = path.join(__dirname, 'uploads');
const modelDir = path.join(__dirname, 'models');
const scriptDir = path.join(__dirname, 'scripts');
const venvDir = path.join(__dirname, 'services/python/venv');

// Python Worker Pool initialisieren
const pythonWorkerPool = new PythonWorkerPool(scriptDir, venvDir, 5);

// Worker Pool Event-Handler
pythonWorkerPool.on('jobProgress', ({ jobId, progress, workerId }) => {
  console.log(`Worker ${workerId} Progress für Job ${jobId}:`, progress);
});

// Job-Completion-Handler
jobQueue.on('jobCompleted', async (job) => {
  try {
    if (job.type === 'training') {
      await handleTrainingJobCompleted(job);
    } else if (job.type === 'retraining') {
      await handleRetrainingJobCompleted(job);
    }
  } catch (error) {
    console.error(`Fehler beim Verarbeiten des abgeschlossenen Jobs ${job.id}:`, error.message);
  }
});

// Job-Failure-Handler
jobQueue.on('jobCompleted', async (job) => {
  if (job.status === 'failed') {
    try {
      if (job.type === 'training' || job.type === 'retraining') {
        await handleTrainingJobFailed(job);
      }
    } catch (error) {
      console.error(`Fehler beim Verarbeiten des fehlgeschlagenen Jobs ${job.id}:`, error.message);
    }
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
setupAPIEndpoints(app, upload, scriptDir, venvDir);
setupProjectRoutes(app, scriptDir, venvDir, trainModelAsync, retrainModelAsync);
setupUploadRoutes(app, upload);
setupAnalyzeRoutes(app, upload, venvDir);
setupLLMRoutes(app);
setupAgentRoutes(app);
setupCacheRoutes(app);
setupMonitoringRoutes(app);
setupQueueRoutes(app);
setupFileRoutes(app);
setupWorkerStatusRoutes(app, pythonWorkerPool);
setupScalingRoutes(app);
setupPredictCacheRoutes(app, scriptDir);
setupPredictionEndpoint(app, scriptDir, venvDir);

// Job-Handler-Funktionen
async function handleTrainingJobCompleted(job) {
  const { projectId } = job.data;
  
  // Prüfe ob job.result existiert (kann bei fehlgeschlagenen Jobs null sein)
  if (!job.result) {
    console.error(`Job ${job.id} für Projekt ${projectId} hat kein result - möglicherweise fehlgeschlagen`);
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
      // console.log('Extrahierte Metriken:', metrics);
    } catch (metricsError) {
      console.error('Fehler bei Metrik-Extraktion:', metricsError.message);
      metrics = { error: 'Metrik-Extraktion fehlgeschlagen' };
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
      performanceMetrics: metrics,
      pythonCode: project.pythonCode, // Bereits in DB gespeichert
      originalPythonCode: project.pythonCode,
      modelPath: modelPath,
      hyperparameters: hyperparameters
    });
    
    // Automatische Performance-Evaluation
    try {
      // console.log(`Starte automatische Performance-Evaluation für Projekt: ${project.name}`);
      const performanceInsights = await evaluatePerformanceWithLLM(await getProject(projectId));
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
      const performanceInsights = await evaluatePerformanceWithLLM(await getProject(projectId));
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
        
    // Python Script aus der Datenbank holen
    let pythonCode = '';
    try {
      if (!project.pythonCode) {
        // throw new Error('Kein Python-Code in der Datenbank gefunden. Bitte zuerst die Agent-Pipeline ausführen.');
        // Führe die Agent-Pipeline aus
        const response = await runNetworkAgentPipeline(project);
        pythonCode = response;
      } else {
        pythonCode = project.pythonCode;
        console.log(`Python-Code aus Datenbank geladen (${pythonCode.length} Zeichen)`);
      }
      
      // Validiere Python-Code
      if (!pythonCode || typeof pythonCode !== 'string') {
        throw new Error('Ungültiger Python-Code in der Datenbank');
      }
      
      
    } catch (error) {
      console.error('Fehler beim Laden des Python-Codes:', error.message);
      await updateProjectTraining(projectId, {
        status: 'Failed',
        performanceMetrics: { error: `Python-Code nicht verfügbar: ${error.message}` },
        pythonCode: '',
        originalPythonCode: '',
        modelPath: '',
        hyperparameters: project?.hyperparameters || {}
      });
      return;
    }

    // Python-Code in der DB speichern vor der Ausführung
    await updateProjectCode(projectId, pythonCode);

    // Job zur Worker Queue hinzufügen
    const jobId = pythonWorkerPool.addTrainingJob(projectId, pythonCode, 1);
    // console.log(`Training-Job ${jobId} für Projekt ${projectId} zur Queue hinzugefügt`);
    
  } catch (error) {
    console.error(`Fehler beim Vorbereiten des Trainings für Projekt ${projectId}:`, error.message);
    
    try {
      await updateProjectTraining(projectId, {
        status: 'Failed',
        performanceMetrics: { error: error.message },
        pythonCode: '',
        originalPythonCode: '',
        modelPath: '',
        hyperparameters: {}
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
    
    // Job zur Worker Queue hinzufügen
    const jobId = pythonWorkerPool.addRetrainingJob(projectId, customPythonCode, 1);
    
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

// Server starten
app.listen(PORT, () => {
  console.log(`Backend läuft auf Port ${PORT}`);
});
