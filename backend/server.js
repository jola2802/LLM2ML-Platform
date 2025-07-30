import express from 'express';
import cors from 'cors';
import multer from 'multer';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

// Service-Imports
import { initializeLogging } from './services/log.js';
import { 
  validatePythonCodeWithLLM,
  validatePythonCode,
  executePythonScript,
  extractMetricsFromOutput
} from './services/code_exec.js';
import { generatePythonScriptWithLLM } from './services/python_generator.js';
import { 
  initializeDatabase,
  getProject,
  updateProjectTraining,
  updateProjectStatus,
  updateProjectInsights
} from './services/db.js';
import { setupAPIEndpoints, setTrainingFunctions, setupPredictionEndpoint } from './services/api_endpoints.js';
import { evaluatePerformanceWithLLM } from './services/llm_api.js';


// Environment-Variablen laden
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Initialisierungen
await initializeLogging();
await initializeDatabase();

const app = express();
const PORT = process.env.PORT;

// Middleware
app.use(cors());
app.use(express.json());

// Vercel serverless function export
if (process.env.NODE_ENV === 'production') {
  // In Vercel: keine statischen Dateien servieren, nur API
  console.log('Running in Vercel production mode');
} else {
  // Serve static files from frontend build (for local development)
  const frontendPath = path.join(__dirname, '../frontend/dist');
  app.use(express.static(frontendPath));
  
  // Serve index.html for all non-API routes (SPA routing)
  app.get('*', (req, res) => {
    if (!req.path.startsWith('/api')) {
      res.sendFile(path.join(frontendPath, 'index.html'));
    }
  });
}

// Upload-Ordner erstellen falls nicht vorhanden
const uploadDir = path.join(__dirname, 'uploads');
const modelDir = path.join(__dirname, 'models');
const scriptDir = path.join(__dirname, 'scripts');
const venvDir = path.join(__dirname, 'services/python/venv');

try {
  await fs.mkdir(uploadDir, { recursive: true });
  await fs.mkdir(modelDir, { recursive: true }); 
  await fs.mkdir(scriptDir, { recursive: true });
} catch (err) {
  console.log('Directories already exist or error creating them:', err.message);
}

// Multer für File Uploads konfigurieren
const upload = multer({ 
  storage: multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
      // Behalte die ursprüngliche Dateiendung bei
      const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
      const fileExtension = path.extname(file.originalname);
      const fileName = uniqueSuffix + fileExtension;
      cb(null, fileName);
    }
  }),
  fileFilter: (req, file, cb) => {
    // Nur CSV-, Excel-, JSON-, Text-, XML-, Word- und PDF-Dateien erlauben
    if (file.mimetype === 'text/csv' || file.originalname.endsWith('.csv') || file.originalname.endsWith('.xlsx') || file.originalname.endsWith('.xls') || file.originalname.endsWith('.json') || file.originalname.endsWith('.txt') || file.originalname.endsWith('.xml') || file.originalname.endsWith('.docx') || file.originalname.endsWith('.doc')) {
      cb(null, true);
    } else {
      cb(new Error('Nur CSV-, Excel-, JSON-, Text-, XML-, Word- und PDF-Dateien sind erlaubt'), false);
    }
  },
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB Limit
  }
});

// API Endpoints Setup
setupAPIEndpoints(app, upload, scriptDir, venvDir);
setupPredictionEndpoint(app, scriptDir, venvDir);

// Training-Funktionen den API-Endpoints zur Verfügung stellen
setTrainingFunctions(trainModelAsync, retrainModelAsync);

// Asynchrone Trainingsfunktion
async function trainModelAsync(projectId) {
  try {
    const project = await getProject(projectId);
    if (!project) return;
        
    // Python Script mit LLM generieren
    const pythonCode = await generatePythonScriptWithLLM(project);

    const scriptPath = path.join(scriptDir, `${projectId}.py`);
    
    // Script in Datei schreiben
    await fs.writeFile(scriptPath, pythonCode);
    
    // Python Script ausführen
    const { stdout, stderr } = await executePythonScript(scriptPath, scriptDir, venvDir);
    
    // console.log('Python output:', stdout);
    if (stderr) {
      console.log('Python stderr:', stderr);
      console.log('Python code:', pythonCode);
    }

    // Baue eine retry-Logik ein, wenn es bei der Ausführung Fehler gibt
    if (stderr) {
      console.log('Fehler bei der Ausführung des Python-Scripts. Versuche es erneut...');
      const retryCode = await validatePythonCodeWithLLM(pythonCode);
      await fs.writeFile(scriptPath, retryCode);
      return trainModelAsync(projectId);
    }
    
    // Performance Metriken aus der Ausgabe extrahieren
    const metrics = extractMetricsFromOutput(stdout, project.modelType);
    
    // Model-Datei pfad
    const modelFileName = `model_${projectId}.pkl`;
    const modelPath = `models/${modelFileName}`;
    const fullModelPath = path.join(__dirname, modelPath);
    
    // Model-Datei verschieben
    const tempModelPath = path.join(scriptDir, 'model.pkl');
    try {
      await fs.access(tempModelPath);
      await fs.rename(tempModelPath, fullModelPath);
      // console.log(`Model erfolgreich verschoben: ${fullModelPath}`);
    } catch (err) {
      // console.log('Could not move model file:', err.message);
    }
    
    // Projekt in DB aktualisieren (inkl. originalPythonCode)
    await updateProjectTraining(projectId, {
      status: 'Completed',
      performanceMetrics: metrics,
      pythonCode: pythonCode, // Bearbeitbar
      originalPythonCode: pythonCode, // Original für Backup
      modelPath: modelPath
    });
    
    // console.log(`Training completed for project: ${project.name}`);
    
    // Automatische Performance-Evaluation nach erfolgreichem Training
    try {
      // console.log(`Starte automatische Performance-Evaluation für Projekt: ${project.name}`);
      const performanceInsights = await evaluatePerformanceWithLLM(await getProject(projectId));
      
      // Performance-Insights in DB speichern
      try {
        await updateProjectInsights(projectId, performanceInsights);
        // console.log(`Performance-Evaluation erfolgreich abgeschlossen für Projekt: ${project.name}`);
      } catch (err) {
        console.error('Fehler beim Speichern der Performance-Insights:', err);
      }
    } catch (evaluationError) {
      console.error(`Performance-Evaluation fehlgeschlagen für Projekt ${projectId}:`, evaluationError);
      // Training ist erfolgreich, auch wenn Evaluation fehlschlägt
    }
    
  } catch (error) {
    console.error(`Training failed for project ${projectId}:`, error);
    
    // Status auf Failed setzen
    await updateProjectTraining(projectId, {
      status: 'Failed',
      performanceMetrics: '',
      pythonCode: pythonCode,
      originalPythonCode: '',
      modelPath: ''
    });
  }
}

// Re-Training mit bearbeitetem Code
async function retrainModelAsync(projectId, customPythonCode) {
  try {
    const project = await getProject(projectId);
    if (!project) return;
    
    // console.log(`Re-Training model for project: ${project.name} with custom code`);
    
    const scriptPath = path.join(scriptDir, `${projectId}.py`);
    
    // Bearbeiteten Python-Code in Datei schreiben
    await fs.writeFile(scriptPath, customPythonCode);
    
    // Python Script ausführen
    const { stdout, stderr } = await executePythonScript(scriptPath, scriptDir, venvDir);
    
    // console.log('Re-training Python output:', stdout);
    if (stderr) console.log('Re-training Python stderr:', stderr);
    
    // Performance Metriken aus der Ausgabe extrahieren
    const metrics = extractMetricsFromOutput(stdout, project.modelType);
    
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
      // console.log('Could not move retrained model file:', err.message);
    }
    
    // Projekt in DB aktualisieren
    await updateProjectTraining(projectId, {
      status: 'Completed',
      performanceMetrics: metrics,
      modelPath: modelPath
    });
    
    // console.log(`Re-training completed for project: ${project.name}`);
    
    // Automatische Performance-Evaluation nach erfolgreichem Re-Training
    try {
      // console.log(`Starte automatische Performance-Evaluation nach Re-Training für Projekt: ${project.name}`);
      const performanceInsights = await evaluatePerformanceWithLLM(await getProject(projectId));
      
      // Performance-Insights in DB speichern
      try {
        await updateProjectInsights(projectId, performanceInsights);
        //console.log(`Performance-Evaluation nach Re-Training erfolgreich abgeschlossen für Projekt: ${project.name}`);
      } catch (err) {
        console.error('Fehler beim Speichern der Performance-Insights nach Re-Training:', err);
      }
    } catch (evaluationError) {
      console.error(`Performance-Evaluation nach Re-Training fehlgeschlagen für Projekt ${projectId}:`, evaluationError);
      // Re-Training ist erfolgreich, auch wenn Evaluation fehlschlägt
    }
    
  } catch (error) {
    console.error(`Re-training failed for project ${projectId}:`, error);
    
    // Status auf Failed setzen
    await updateProjectStatus(projectId, 'Re-training Failed');
  }
}

// Server starten
app.listen(PORT, () => {
  console.log(`Server läuft auf Port ${PORT}`);
});

// Export für Vercel
module.exports = app;