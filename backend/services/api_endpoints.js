import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { predictWithModel } from './code_exec.js';

// Service-Imports
import { 
  analyzeCsvFile, 
  analyzeJsonFile, 
  analyzeExcelFile, 
  analyzeTextFile, 
  analyzeGenericFile,
  getLLMRecommendations,
  evaluatePerformanceWithLLM 
} from './llm_api.js';
import { 
  executePythonScript,
  generatePredictionScript 
} from './code_exec.js';
import { 
  getProject,
  getAllProjects,
  createProject,
  createSmartProject,
  updateProjectCode,
  updateProjectStatus,
  updateProjectInsights,
  deleteProject 
} from './db.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Import der Training-Funktionen (diese bleiben in server.js)
let trainModelAsync, retrainModelAsync;

export function setTrainingFunctions(trainFn, retrainFn) {
  trainModelAsync = trainFn;
  retrainModelAsync = retrainFn;
}

export function setupAPIEndpoints(app, upload, scriptDir, venvDir) {
  
  // Datei hochladen und intelligente Analyse
  app.post('/api/upload', upload.single('file'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'Keine Datei hochgeladen' });
      }

      const filePath = req.file.path;
      const originalName = req.file.originalname;
      const fileExtension = path.extname(originalName).toLowerCase();
      
      console.log(`Datei hochgeladen: ${originalName} (${fileExtension}) -> ${filePath}`);
      
      // Datei basierend auf Typ analysieren
      let analysis;
      if (fileExtension === '.csv') {
        analysis = await analyzeCsvFile(filePath);
      } else if (fileExtension === '.json') {
        analysis = await analyzeJsonFile(filePath);
      } else if (fileExtension === '.xlsx' || fileExtension === '.xls') {
        analysis = await analyzeExcelFile(filePath);
      } else if (fileExtension === '.txt') {
        analysis = await analyzeTextFile(filePath);
      } else if (fileExtension === '.pdf') {
        analysis = await analyzeGenericFile(filePath, fileExtension);
      } else if (fileExtension === '.xml') {
        analysis = await analyzeGenericFile(filePath, fileExtension);
      } else if (fileExtension === '.docx' || fileExtension === '.doc') {
        analysis = await analyzeGenericFile(filePath, fileExtension);
      } else {
        analysis = await analyzeGenericFile(filePath, fileExtension);
      }
      
      // LLM-basierte Empfehlungen für Algorithmus und Features
      const recommendations = await getLLMRecommendations(analysis, filePath);
      
      res.json({
        fileName: originalName,
        filePath: filePath,
        fileType: fileExtension,
        columns: analysis.columns,
        rowCount: analysis.rowCount,
        dataTypes: analysis.dataTypes,
        sampleData: analysis.sampleData,
        llmAnalysis: analysis.llm_analysis,
        recommendations: recommendations // Automatische LLM-Empfehlungen
      });
    } catch (error) {
      console.error('Fehler beim Datei-Upload:', error);
      res.status(500).json({ error: 'Fehler beim Analysieren der Datei: ' + error.message });
    }
  });

  // Intelligentes Projekt erstellen (mit LLM-Empfehlungen)
  app.post('/api/projects/smart-create', async (req, res) => {
    try {
      const { name, csvFilePath, recommendations } = req.body;
      
      const project = await createSmartProject({ name, csvFilePath, recommendations });
      
      res.json(project);
      
      // Asynchron das Training starten
      if (trainModelAsync) {
        trainModelAsync(project.id);
      }
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // Alle Projekte abrufen
  app.get('/api/projects', async (req, res) => {
    try {
      const projects = await getAllProjects();
      res.json(projects);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // Einzelnes Projekt abrufen
  app.get('/api/projects/:id', async (req, res) => {
    try {
      const project = await getProject(req.params.id);
      if (!project) {
        return res.status(404).json({ error: 'Project not found' });
      }
      res.json(project);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // Neues Projekt erstellen
  app.post('/api/projects', async (req, res) => {
    try {
      const { name, modelType, dataSourceName, targetVariable, features, csvFilePath, algorithm, hyperparameters } = req.body;
      
      const project = await createProject({ 
        name, modelType, dataSourceName, targetVariable, features, csvFilePath, algorithm, hyperparameters 
      });
      
      res.json(project);
      
      // Asynchron das Training starten
      if (trainModelAsync) {
        trainModelAsync(project.id);
      }
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // Modell herunterladen
  app.get('/api/projects/:id/download', async (req, res) => {
    try {
      const project = await getProject(req.params.id);
      if (!project) {
        return res.status(404).json({ error: 'Project not found' });
      }
      
      if (!project.modelPath) {
        return res.status(404).json({ error: 'Model file not found' });
      }
      
      const modelPath = path.join(path.dirname(__dirname), project.modelPath);
      
      // Prüfen ob Datei existiert
      try {
        await fs.access(modelPath);
      } catch {
        return res.status(404).json({ error: 'Model file not found on disk' });
      }
      
      // Download headers setzen
      res.setHeader('Content-Disposition', `attachment; filename="${project.name}_model.pkl"`);
      res.setHeader('Content-Type', 'application/octet-stream');
      
      // Datei senden
      res.sendFile(modelPath);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // Projekt löschen  
  app.delete('/api/projects/:id', async (req, res) => {
    const { id } = req.params;
    
    try {
      const project = await getProject(id);
      if (project && project.modelPath) {
        // Model-Datei löschen falls vorhanden
        try {
          await fs.unlink(path.join(path.dirname(__dirname), project.modelPath));
        } catch (err) {
          console.log('Could not delete model file:', err.message);
        }
      }
      
      const result = await deleteProject(id);
      
      if (result.changes === 0) {
        res.status(404).json({ error: 'Project not found' });
        return;
      }
      
      res.json({ message: 'Project deleted successfully' });
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // Python-Code eines Projekts bearbeiten
  app.put('/api/projects/:id/code', async (req, res) => {
    try {
      const { id } = req.params;
      const { pythonCode } = req.body;
      
      const project = await getProject(id);
      if (!project) {
        return res.status(404).json({ error: 'Project not found' });
      }
      
      // Python-Code in der Datenbank aktualisieren
      await updateProjectCode(id, pythonCode);
      
      res.json({ message: 'Python-Code erfolgreich aktualisiert', id });
      
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // Projekt mit bearbeitetem Code re-trainieren
  app.post('/api/projects/:id/retrain', async (req, res) => {
    try {
      const { id } = req.params;
      
      const project = await getProject(id);
      if (!project) {
        return res.status(404).json({ error: 'Project not found' });
      }
      
      if (!project.pythonCode) {
        return res.status(400).json({ error: 'Kein Python-Code zum Re-Training verfügbar' });
      }
      
      // Status auf Re-Training setzen
      await updateProjectStatus(id, 'Re-Training');
      
      res.json({ message: 'Re-Training gestartet', id });
      
      // Asynchron das Re-Training mit dem bearbeiteten Code starten
      if (retrainModelAsync) {
        retrainModelAsync(id, project.pythonCode);
      }
      
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // API-Endpoint für intelligente Performance-Evaluation
  app.post('/api/projects/:id/evaluate-performance', async (req, res) => {
    try {
      const { id } = req.params;
      const project = await getProject(id);
      
      if (!project) {
        return res.status(404).json({ error: 'Projekt nicht gefunden' });
      }
      
      if (!project.performanceMetrics) {
        return res.status(400).json({ error: 'Keine Performance-Metriken verfügbar für Evaluation' });
      }
      
      console.log(`Starte intelligente Performance-Evaluation für Projekt: ${project.name}`);
      
      // LLM-basierte Performance-Evaluation durchführen
      const performanceInsights = await evaluatePerformanceWithLLM(project);
      
      // Performance-Insights in DB speichern
      await updateProjectInsights(id, performanceInsights);
      
      res.json({
        message: 'Performance-Evaluation erfolgreich abgeschlossen',
        insights: performanceInsights
      });
      
    } catch (error) {
      console.error('Fehler bei Performance-Evaluation:', error);
      res.status(500).json({ error: 'Performance-Evaluation fehlgeschlagen: ' + error.message });
    }
  });

}

// Prediction-Endpoint für echte API-Nutzung
export function setupPredictionEndpoint(app, scriptDir, venvDir) {
  app.post('/api/predict/:id', async (req, res) => {
    try {
      const { id } = req.params;
      const inputs = req.body;

      const project = await getProject(id);
      if (!project) {
        return res.status(404).json({ error: 'Project not found' });
      }
      
      const prediction = await predictWithModel(project, inputs, scriptDir, venvDir);
      
      res.json({ prediction });
    } catch (error) {
      console.error('Prediction error:', error);
      res.status(500).json({ error: 'Prediction failed: ' + error.message });
    }
  });
}
