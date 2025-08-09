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
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

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
      const now = new Date();
      const uniqueSuffix = now.toISOString().split('T')[0] + '-' + now.getHours() + '-' + now.getMinutes() + '-' + now.getSeconds();
      const fileName =   uniqueSuffix + file.originalname;
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

// Health Check Endpoint
app.get('/health', (req, res) => {
  res.status(200).json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    service: 'ml-platform-backend'
  });
});

// API Endpoints Setup
setupAPIEndpoints(app, upload, scriptDir, venvDir);
setupPredictionEndpoint(app, scriptDir, venvDir);

// Training-Funktionen den API-Endpoints zur Verfügung stellen
setTrainingFunctions(trainModelAsync, retrainModelAsync);

// Asynchrone Trainingsfunktion
async function trainModelAsync(projectId) {
  let pythonCode = '';
  
  try {
    const project = await getProject(projectId);
    if (!project) return;
        
    // Python Script mit LLM generieren
    try {
      const response = await generatePythonScriptWithLLM(project);
      
      // Validiere Response-Format
      if (typeof response === 'string') {
        pythonCode = response;
      } else if (response && response.result) {
        pythonCode = response.result;
      } else if (response && response.text) {
        pythonCode = response.text;
      } else {
        throw new Error('Ungültiges Response-Format vom Python-Generator');
      }
      
      // Validiere Python-Code
      if (!pythonCode || typeof pythonCode !== 'string') {
        throw new Error('Leerer oder ungültiger Python-Code erhalten');
      }
      
      console.log(`Python-Code erfolgreich generiert (${pythonCode.length} Zeichen)`);
      
    } catch (generationError) {
      console.error('Fehler bei Python-Code-Generierung:', generationError.message);
      throw new Error(`Python-Code-Generierung fehlgeschlagen: ${generationError.message}`);
    }

    const scriptPath = path.join(scriptDir, `${projectId}.py`);
    
    // Script in Datei schreiben
    try {
      await fs.writeFile(scriptPath, pythonCode);
      console.log(`Python-Script gespeichert: ${scriptPath}`);
    } catch (writeError) {
      console.error('Fehler beim Schreiben der Python-Datei:', writeError.message);
      throw new Error(`Datei-Schreibfehler: ${writeError.message}`);
    }
    
    // Python Script ausführen
    let executionResult;
    try {
      executionResult = await executePythonScript(scriptPath, scriptDir, venvDir);
      console.log('Python-Script erfolgreich ausgeführt');
    } catch (executionError) {
      console.error('Fehler bei Python-Script-Ausführung:', executionError.message);
      throw new Error(`Python-Ausführungsfehler: ${executionError.message}`);
    }
    
    const { stdout, stderr } = executionResult;
    
    // Log Ausgabe für Debugging
    if (stdout) {
      console.log('Python stdout (erste 500 Zeichen):', stdout.substring(0, 500));
    }
    if (stderr) {
      console.log('Python stderr (erste 500 Zeichen):', stderr.substring(0, 500));
    }

    // Performance Metriken aus der Ausgabe extrahieren
    let metrics = {};
    try {
      metrics = extractMetricsFromOutput(stdout, project.modelType);
      console.log('Extrahierte Metriken:', metrics);
    } catch (metricsError) {
      console.error('Fehler bei Metrik-Extraktion:', metricsError.message);
      metrics = { error: 'Metrik-Extraktion fehlgeschlagen' };
    }
    
    // Model-Datei pfad
    const modelFileName = `model_${projectId}.pkl`;
    const modelPath = `models/${modelFileName}`;
    const fullModelPath = path.join(__dirname, modelPath);
    
    // Model-Datei verschieben
    const tempModelPath = path.join(scriptDir, 'model.pkl');
    try {
      await fs.access(tempModelPath);
      await fs.rename(tempModelPath, fullModelPath);
      console.log(`Model-Datei verschoben: ${fullModelPath}`);
    } catch (modelError) {
      console.log('Model-Datei nicht gefunden oder Fehler beim Verschieben:', modelError.message);
    }
    
    // Hyperparameter aus dem Python-Code extrahieren
    let hyperparameters = {};
    try {
      hyperparameters = extractHyperparametersFromCode(pythonCode);
      console.log('Extrahierte Hyperparameter:', hyperparameters);
    } catch (hyperError) {
      console.error('Fehler bei Hyperparameter-Extraktion:', hyperError.message);
      hyperparameters = project.hyperparameters || {};
    }
    
    // Projekt in DB aktualisieren (inkl. originalPythonCode und Hyperparameter)
    try {
      await updateProjectTraining(projectId, {
        status: 'Completed',
        performanceMetrics: metrics,
        pythonCode: pythonCode, // Bearbeitbar
        originalPythonCode: pythonCode, // Original für Backup
        modelPath: modelPath,
        hyperparameters: hyperparameters
      });
      console.log(`Projekt ${project.name} erfolgreich aktualisiert`);
    } catch (dbError) {
      console.error('Fehler beim DB-Update:', dbError.message);
      throw new Error(`DB-Update fehlgeschlagen: ${dbError.message}`);
    }
    
    // Automatische Performance-Evaluation nach erfolgreichem Training
    try {
      console.log(`Starte automatische Performance-Evaluation für Projekt: ${project.name}`);
      const performanceInsights = await evaluatePerformanceWithLLM(await getProject(projectId));
      
      // Performance-Insights in DB speichern
      try {
        await updateProjectInsights(projectId, performanceInsights);
        console.log(`Performance-Evaluation erfolgreich abgeschlossen für Projekt: ${project.name}`);
      } catch (insightsError) {
        console.error('Fehler beim Speichern der Performance-Insights:', insightsError.message);
        // Nicht kritisch - Training war erfolgreich
      }
    } catch (evaluationError) {
      console.error(`Performance-Evaluation fehlgeschlagen für Projekt ${projectId}:`, evaluationError.message);
      // Training ist erfolgreich, auch wenn Evaluation fehlschlägt
    }
    
    console.log(`Training erfolgreich abgeschlossen für Projekt: ${project.name}`);
    
  } catch (error) {
    console.error(`Training failed for project ${projectId}:`, error.message);
    
    // Hyperparameter aus dem Python-Code extrahieren (falls vorhanden)
    let hyperparameters = {};
    try {
      hyperparameters = pythonCode ? extractHyperparametersFromCode(pythonCode) : (project?.hyperparameters || {});
    } catch (hyperError) {
      console.error('Fehler bei Hyperparameter-Extraktion im Fehlerfall:', hyperError.message);
      hyperparameters = project?.hyperparameters || {};
    }
    
    // Status auf Failed setzen
    try {
      await updateProjectTraining(projectId, {
        status: 'Failed',
        performanceMetrics: { error: error.message },
        pythonCode: pythonCode,
        originalPythonCode: '',
        modelPath: '',
        hyperparameters: hyperparameters
      });
      console.log(`Projekt ${projectId} als fehlgeschlagen markiert`);
    } catch (dbError) {
      console.error('Fehler beim Markieren des Projekts als fehlgeschlagen:', dbError.message);
    }
  }
}

// Hyperparameter aus Python-Code extrahieren
function extractHyperparametersFromCode(pythonCode) {
  try {
    // Suche nach der hyperparameters-Zeile
    const lines = pythonCode.split('\n');
    for (const line of lines) {
      if (line.includes('hyperparameters = ')) {
        // Verschiedene Formate unterstützen
        let match = line.match(/hyperparameters = "(.+)"/);
        if (match) {
          const jsonStr = match[1].replace(/\\"/g, '"');
          const hyperparameters = JSON.parse(jsonStr);
          return convertHyperparametersToNumbers(hyperparameters);
        }
        
        // Alternative: hyperparameters = {...}
        match = line.match(/hyperparameters = (\{.*\})/);
        if (match) {
          const hyperparameters = JSON.parse(match[1]);
          return convertHyperparametersToNumbers(hyperparameters);
        }
        
        // Alternative: hyperparameters = {...} (mit Zeilenumbrüchen)
        const startIndex = line.indexOf('hyperparameters = {');
        if (startIndex !== -1) {
          let jsonStr = line.substring(startIndex + 'hyperparameters = '.length);
          let braceCount = 0;
          let inString = false;
          let escapeNext = false;
          
          for (let i = 0; i < jsonStr.length; i++) {
            const char = jsonStr[i];
            if (escapeNext) {
              escapeNext = false;
              continue;
            }
            if (char === '\\') {
              escapeNext = true;
              continue;
            }
            if (char === '"' && !escapeNext) {
              inString = !inString;
            }
            if (!inString) {
              if (char === '{') braceCount++;
              if (char === '}') {
                braceCount--;
                if (braceCount === 0) {
                  jsonStr = jsonStr.substring(0, i + 1);
                  break;
                }
              }
            }
          }
          
          try {
            const hyperparameters = JSON.parse(jsonStr);
            return convertHyperparametersToNumbers(hyperparameters);
          } catch (e) {
            console.error('Fehler beim Parsen der Hyperparameter:', e);
          }
        }
      }
    }
    return null;
  } catch (error) {
    console.error('Fehler beim Extrahieren der Hyperparameter:', error);
    return null;
  }
}

// Hyperparameter zu numerischen Werten konvertieren
function convertHyperparametersToNumbers(hyperparameters) {
  if (!hyperparameters || typeof hyperparameters !== 'object') {
    return hyperparameters;
  }
  
  const converted = {};
  for (const [key, value] of Object.entries(hyperparameters)) {
    if (typeof value === 'string' && !isNaN(Number(value)) && value.trim() !== '') {
      converted[key] = Number(value);
    } else {
      converted[key] = value;
    }
  }
  return converted;
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
    
    // Hyperparameter aus dem Python-Code extrahieren
    const hyperparameters = extractHyperparametersFromCode(customPythonCode);
    
    // Projekt in DB aktualisieren
    await updateProjectTraining(projectId, {
      status: 'Completed',
      performanceMetrics: metrics,
      pythonCode: customPythonCode, // Wichtig: Python-Code aktualisieren
      modelPath: modelPath,
      hyperparameters: hyperparameters
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
    
    // Hyperparameter aus dem Python-Code extrahieren (falls vorhanden)
    const hyperparameters = customPythonCode ? extractHyperparametersFromCode(customPythonCode) : null;
    
    // Status auf Failed setzen
    await updateProjectTraining(projectId, {
      status: 'Re-training Failed',
      pythonCode: customPythonCode, // Python-Code auch bei Fehler beibehalten
      hyperparameters: hyperparameters
    });
  }
}

// Server starten
app.listen(PORT, () => {
  console.log(`Server läuft auf Port ${PORT}`);
});