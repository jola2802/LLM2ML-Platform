import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { predictWithModel } from './code_exec.js';

// Service-Imports
import { 
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
  updateProjectHyperparameters,
  updateProjectStatus,
  updateProjectInsights,
  deleteProject 
} from './db.js';
import { 
  analyzeCsvFile, 
  analyzeJsonFile, 
  analyzeExcelFile, 
  analyzeTextFile, 
  analyzeGenericFile,
} from './file_analysis.js';
import {
  getCachedDataAnalysis,
  clearAnalysisCache,
  getAnalysisCacheStatus,
  analyzeDataForLLM
} from './data_exploration.js';
import { logRESTAPIRequest } from './log.js';
import { getLLMQueueStatus, cancelLLMRequest } from './llm.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Import der Training-Funktionen (diese bleiben in server.js)
let trainModelAsync, retrainModelAsync;

export function setTrainingFunctions(trainFn, retrainFn) {
  trainModelAsync = trainFn;
  retrainModelAsync = retrainFn;
}

export function setupAPIEndpoints(app, upload, scriptDir, venvDir) {
  
  // Datei hochladen und Basis-Analyse (ohne LLM)
  app.post('/api/upload', upload.single('file'), async (req, res) => {
    try {
      logRESTAPIRequest('upload', req.file);
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
        analysis = await analyzeCsvFile(filePath, true);
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
      
      // Nur Basis-Analyse zurückgeben (ohne LLM-Empfehlungen)
      res.json({
        fileName: originalName,
        filePath: filePath,
        fileType: fileExtension,
        columns: analysis.columns,
        rowCount: analysis.rowCount,
        dataTypes: analysis.dataTypes,
        sampleData: analysis.sampleData,
        llmAnalysis: analysis.llm_analysis
      });
    } catch (error) {
      console.error('Fehler beim Datei-Upload:', error);
      res.status(500).json({ error: 'Fehler beim Analysieren der Datei: ' + error.message });
    }
  });

  // Intelligente LLM-Empfehlungen für manipulierte Daten
  app.post('/api/analyze-data', async (req, res) => {
    try {
      logRESTAPIRequest('analyze-data', req.body);
      const { filePath, excludedColumns, excludedFeatures, selectedColumns } = req.body;
      
      if (!filePath) {
        return res.status(400).json({ error: 'filePath ist erforderlich' });
      }
      
      const fileExtension = path.extname(filePath).toLowerCase();
      
      // Datei basierend auf Typ analysieren
      let analysis;
      if (fileExtension === '.csv') {
        analysis = await analyzeCsvFile(filePath, true);
        analysis.file_type = 'CSV';
      } else if (fileExtension === '.json') {
        analysis = await analyzeJsonFile(filePath);
        analysis.file_type = 'JSON';
      } else if (fileExtension === '.xlsx' || fileExtension === '.xls') {
        analysis = await analyzeExcelFile(filePath);
        analysis.file_type = 'Excel';
      } else if (fileExtension === '.txt') {
        analysis = await analyzeTextFile(filePath);
        analysis.file_type = 'Text';
      } else if (fileExtension === '.pdf') {
        analysis = await analyzeGenericFile(filePath, fileExtension);
        analysis.file_type = 'PDF';
      } else if (fileExtension === '.xml') {
        analysis = await analyzeGenericFile(filePath, fileExtension);
        analysis.file_type = 'XML';
      } else if (fileExtension === '.docx' || fileExtension === '.doc') {
        analysis = await analyzeGenericFile(filePath, fileExtension);
        analysis.file_type = 'Word';
      } else {
        analysis = await analyzeGenericFile(filePath, fileExtension);
        analysis.file_type = fileExtension.substring(1).toUpperCase();
      }
      
      // Spalten basierend auf Manipulationen anpassen
      let manipulatedAnalysis = { ...analysis };
      
      if (excludedColumns && excludedColumns.length > 0) {
        manipulatedAnalysis.columns = analysis.columns.filter(col => !excludedColumns.includes(col));
        manipulatedAnalysis.sampleData = analysis.sampleData.map(row => 
          row.filter((_, index) => !excludedColumns.includes(analysis.columns[index]))
        );
      }
      
      if (selectedColumns && selectedColumns.length > 0) {
        manipulatedAnalysis.columns = selectedColumns;
        manipulatedAnalysis.sampleData = analysis.sampleData.map(row => 
          selectedColumns.map(col => row[analysis.columns.indexOf(col)])
        );
      }
      
      if (excludedFeatures && excludedFeatures.length > 0) {
        manipulatedAnalysis.columns = analysis.columns.filter(col => !excludedFeatures.includes(col));
      }
      
      // LLM-basierte Empfehlungen für manipulierte Daten
      const recommendations = await getLLMRecommendations(
        manipulatedAnalysis, 
        filePath, 
        venvDir, 
        selectedColumns, 
        excludedFeatures
      );
      
      // Sicherstellen, dass recommendations.features existiert
      if (!recommendations.features || recommendations.features.length === 0) {
        console.warn('LLM gab keine Features zurück, verwende Fallback');
        recommendations.features = manipulatedAnalysis.columns.filter(col => col !== recommendations.targetVariable);
      }
      
      res.json({
        analysis: manipulatedAnalysis,
        recommendations: recommendations,
        availableFeatures: recommendations.features // LLM-empfohlene Features, nicht alle Spalten
      });
      
    } catch (error) {
      console.error('Fehler bei der Datenanalyse:', error);
      res.status(500).json({ error: 'Fehler bei der Datenanalyse: ' + error.message });
    }
  });

  // Intelligentes Projekt erstellen (mit LLM-Empfehlungen)
  app.post('/api/projects/smart-create', async (req, res) => {
    try {
      logRESTAPIRequest('smart-create-project', req.body);
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
      logRESTAPIRequest('get-all-projects', req.body);
      const projects = await getAllProjects();
      res.json(projects);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // Einzelnes Projekt abrufen
  app.get('/api/projects/:id', async (req, res) => {
    try {
      logRESTAPIRequest('get-project', req);
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
      logRESTAPIRequest('create-project', req.body);
      console.log('Neues Projekt erstellen:', req.body);
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
      logRESTAPIRequest('download-project', req.params.id);
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
    logRESTAPIRequest('delete-project', req.params.id);
    const { id } = req.params;
    
    try {
      const project = await getProject(id);
      if (project && project.modelPath) {
        // Upload-Datei, Skirpt und Predict Datei sowie Model-Datei löschen falls vorhanden
        try {
          await fs.unlink( project.csvFilePath);
          await fs.unlink(path.join(path.dirname(__dirname), project.modelPath));
          //  Suche im Ordner scripts nach allen Dateien, die die Projekt-ID im Namen haben und lösche sie
          const scriptDir = path.join(path.dirname(__dirname), 'scripts');
          const scriptFiles = await fs.readdir(scriptDir);
          for (const file of scriptFiles) {
            if (file.includes(id)) {
              await fs.unlink(path.join(scriptDir, file));
            }
          }
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
      logRESTAPIRequest('update-project-code', req.params.id);
      const { id } = req.params;
      const { pythonCode, hyperparameters } = req.body;
      
      const project = await getProject(id);
      if (!project) {
        return res.status(404).json({ error: 'Project not found' });
      }
      
      // Python-Code in der Datenbank aktualisieren
      await updateProjectCode(id, pythonCode);
      
      // Hyperparameter aktualisieren, falls vorhanden
      if (hyperparameters) {
        await updateProjectHyperparameters(id, hyperparameters);
      }
      
      res.json({ message: 'Python-Code und Hyperparameter erfolgreich aktualisiert', id });
      
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // Projekt mit bearbeitetem Code re-trainieren
  app.post('/api/projects/:id/retrain', async (req, res) => {
    try {
      logRESTAPIRequest('retrain-project', req.params.id);
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
      logRESTAPIRequest('evaluate-performance', req.params.id);
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

  // Erweiterte Datenstatistiken für Data Insights
  app.get('/api/projects/:id/data-statistics', async (req, res) => {
    try {
      logRESTAPIRequest('data-statistics', req.params.id);
      const { id } = req.params;
      const project = await getProject(id);
      
      if (!project) {
        return res.status(404).json({ error: 'Projekt nicht gefunden' });
      }

      // Prüfe ob CSV-Pfad vorhanden ist
      if (!project.csvFilePath) {
        return res.status(400).json({ error: 'Keine CSV-Datei für dieses Projekt verfügbar' });
      }
      
      // CSV-Datei analysieren für detaillierte Statistiken
      const fileExtension = path.extname(project.csvFilePath).toLowerCase();
      let analysis;
      
      if (fileExtension === '.csv') {
        analysis = await analyzeCsvFile(project.csvFilePath, false);
      } else {
        return res.status(400).json({ error: 'Nur CSV-Dateien werden für Datenstatistiken unterstützt' });
      }

      // Erweiterte Statistiken basierend auf der Analyse erstellen
      const statistics = {
        basicInfo: {
          fileName: path.basename(project.csvFilePath),
          fileSize: 0, // Wird bei Bedarf implementiert
          rowCount: analysis.rowCount,
          columnCount: analysis.columns.length,
          dataTypes: analysis.dataTypes
        },
        columnAnalysis: analysis.columns.map(column => ({
          name: column,
          dataType: analysis.dataTypes[column],
          sampleValues: analysis.sampleData.slice(0, 50).map(row => {
            const colIndex = analysis.columns.indexOf(column);
            return row[colIndex];
          }).filter(val => val != null && val !== ''),
          isFeature: project.features.includes(column),
          isTarget: project.targetVariable === column
        })),
        sampleData: {
          headers: analysis.columns.slice(0, 50), // Erste 50 Spalten als Preview
          rows: analysis.sampleData.slice(0, 50) // Erste 50 Zeilen als Preview
        },
        mlConfig: {
          algorithm: project.algorithm,
          modelType: project.modelType,
          targetVariable: project.targetVariable,
          features: project.features,
          excludedColumns: project.recommendations?.excludedColumns || [],
          excludedFeatures: project.recommendations?.excludedFeatures || []
        }
      };

      res.json(statistics);
      
    } catch (error) {
      console.error('Fehler bei Datenstatistiken:', error);
      res.status(500).json({ error: 'Datenstatistiken-Abruf fehlgeschlagen: ' + error.message });
    }
  });

  // Datenstatistiken für ein Projekt abrufen
  app.get('/api/projects/:id/stats', async (req, res) => {
    try {
      logRESTAPIRequest('stats', req.params.id);
      const projectId = req.params.id;
      const project = await getProject(projectId);
      
      if (!project) {
        return res.status(404).json({ error: 'Projekt nicht gefunden' });
      }
      
      if (!project.csvFilePath) {
        return res.status(400).json({ error: 'Keine CSV-Datei für dieses Projekt verfügbar' });
      }
      
      const fileExtension = path.extname(project.csvFilePath).toLowerCase();
      
      if (fileExtension === '.csv') {
        analysis = await analyzeCsvFile(project.csvFilePath, false, venvDir);
      } else {
        return res.status(400).json({ error: 'Nur CSV-Dateien werden für Datenstatistiken unterstützt' });
      }
      
      res.json({
        projectId: projectId,
        fileName: project.name,
        stats: analysis
      });
      
    } catch (error) {
      console.error('Fehler beim Abrufen der Datenstatistiken:', error);
      res.status(500).json({ error: 'Fehler beim Abrufen der Datenstatistiken: ' + error.message });
    }
  });

  // Cache löschen
  app.post('/api/cache/clear', async (req, res) => {
    try {
      logRESTAPIRequest('clear-cache', req.body);
      res.json({ message: 'File-Cache wurde entfernt - keine Aktion erforderlich' });
    } catch (error) {
      console.error('Fehler beim Löschen des Caches:', error);
      res.status(500).json({ error: 'Fehler beim Löschen des Caches: ' + error.message });
    }
  });

  // Cache-Status abrufen
  app.get('/api/cache/status', async (req, res) => {
    try {
      logRESTAPIRequest('get-cache-status', req.body);
      res.json({ 
        message: 'File-Cache wurde entfernt',
        cachedFiles: [],
        cacheSize: 0
      });
    } catch (error) {
      console.error('Fehler beim Abrufen des Cache-Status:', error);
      res.status(500).json({ error: 'Fehler beim Abrufen des Cache-Status: ' + error.message });
    }
  });

  // Datenexploration-Cache leeren
  app.post('/api/analysis-cache/clear', async (req, res) => {
    try {
      logRESTAPIRequest('clear-analysis-cache', req.body);
      await clearAnalysisCache();
      res.json({ message: 'Datenanalyse-Cache erfolgreich geleert' });
    } catch (error) {
      console.error('Fehler beim Löschen des Datenanalyse-Caches:', error);
      res.status(500).json({ error: 'Fehler beim Löschen des Datenanalyse-Caches: ' + error.message });
    }
  });

  // Datenexploration-Cache-Status abrufen
  app.get('/api/analysis-cache/status', async (req, res) => {
    try {
      logRESTAPIRequest('get-analysis-cache-status', req.body);
      const status = await getAnalysisCacheStatus();
      res.json(status);
    } catch (error) {
      console.error('Fehler beim Abrufen des Datenanalyse-Cache-Status:', error);
      res.status(500).json({ error: 'Fehler beim Abrufen des Datenanalyse-Cache-Status: ' + error.message });
    }
  });

  // Automatische Datenexploration für eine Datei
  app.post('/api/explore-data', async (req, res) => {
    try {
      logRESTAPIRequest('explore-data', req.body);
      const { filePath } = req.body;
      
      if (!filePath) {
        return res.status(400).json({ error: 'filePath ist erforderlich' });
      }
      
      if (!fs.access(filePath).then(() => true).catch(() => false)) {
        return res.status(404).json({ error: 'Datei nicht gefunden' });
      }
      
      const analysis = await getCachedDataAnalysis(filePath, false);
      res.json(analysis);
      
    } catch (error) {
      console.error('Fehler bei der Datenexploration:', error);
      res.status(500).json({ error: 'Fehler bei der Datenexploration: ' + error.message });
    }
  });

  // ===== NEUE EINHEITLICHE LLM API ENDPOINTS =====

  // Test-Endpoint für LLM-API
  app.get('/api/llm/test', async (req, res) => {
    try {
      res.json({
        success: true,
        message: 'LLM API funktioniert',
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      res.status(500).json({ error: 'Test failed: ' + error.message });
    }
  });

  // Aktuelle LLM-Konfiguration abrufen
  app.get('/api/llm/config', async (req, res) => {
    try {
      logRESTAPIRequest('get-llm-config', req.body);
      const { getLLMConfig } = await import('./llm.js');
      const config = getLLMConfig();
      
      res.json({
        success: true,
        config: config
      });
    } catch (error) {
      console.error('Error getting LLM config:', error);
      res.status(500).json({ error: 'Failed to get config: ' + error.message });
    }
  });

  // Aktiven Provider setzen
  app.post('/api/llm/provider', async (req, res) => {
    try {
      logRESTAPIRequest('set-llm-provider', req.body);
      const { provider } = req.body;
      
      if (!provider) {
        return res.status(400).json({ error: 'Provider erforderlich' });
      }

      const { setActiveProvider, LLM_PROVIDERS } = await import('./llm.js');
      
      if (!Object.values(LLM_PROVIDERS).includes(provider)) {
        return res.status(400).json({ error: 'Ungültiger Provider' });
      }

      setActiveProvider(provider);
      
      res.json({
        success: true,
        message: `Provider auf ${provider} gesetzt`,
        provider: provider
      });
    } catch (error) {
      console.error('Error setting LLM provider:', error);
      res.status(500).json({ error: 'Failed to set provider: ' + error.message });
    }
  });

  // Ollama-spezifische Endpoints
  app.get('/api/llm/ollama/models', async (req, res) => {
    try {
      logRESTAPIRequest('get-ollama-models', req.body);
      const { getAvailableOllamaModels } = await import('./llm.js');
      const result = await getAvailableOllamaModels();
      
      res.json(result);
    } catch (error) {
      console.error('Error getting Ollama models:', error);
      res.status(500).json({ error: 'Failed to get models: ' + error.message });
    }
  });

  app.post('/api/llm/ollama/test', async (req, res) => {
    try {
      logRESTAPIRequest('test-ollama-connection', req.body);
      const { testOllamaConnection } = await import('./llm.js');
      const result = await testOllamaConnection();
      
      res.json(result);
    } catch (error) {
      console.error('Error testing Ollama connection:', error);
      res.status(500).json({ error: 'Failed to test connection: ' + error.message });
    }
  });

  app.post('/api/llm/ollama/config', async (req, res) => {
    try {
      logRESTAPIRequest('update-ollama-config', req.body);
      const { host, defaultModel } = req.body;
      
      const { updateOllamaConfig } = await import('./llm.js');
      const config = {};
      
      if (host) config.host = host;
      if (defaultModel) config.defaultModel = defaultModel;
      
      updateOllamaConfig(config);
      
      res.json({
        success: true,
        message: 'Ollama-Konfiguration aktualisiert',
        config: config
      });
    } catch (error) {
      console.error('Error updating Ollama config:', error);
      res.status(500).json({ error: 'Failed to update config: ' + error.message });
    }
  });

  // Gemini-spezifische Endpoints
  app.post('/api/llm/gemini/test', async (req, res) => {
    try {
      logRESTAPIRequest('test-gemini-connection', req.body);
      const { testGeminiConnection } = await import('./llm.js');
      const result = await testGeminiConnection();
      
      res.json(result);
    } catch (error) {
      console.error('Error testing Gemini connection:', error);
      res.status(500).json({ error: 'Failed to test connection: ' + error.message });
    }
  });

  app.post('/api/llm/gemini/config', async (req, res) => {
    try {
      logRESTAPIRequest('update-gemini-config', req.body);
      const { apiKey, defaultModel } = req.body;
      
      const { updateGeminiConfig } = await import('./llm.js');
      const config = {};
      
      if (apiKey) config.apiKey = apiKey;
      if (defaultModel) config.defaultModel = defaultModel;
      
      updateGeminiConfig(config);
      
      res.json({
        success: true,
        message: 'Gemini-Konfiguration aktualisiert',
        config: {
          ...config,
          apiKey: config.apiKey ? `${config.apiKey.substring(0, 8)}...` : null
        }
      });
    } catch (error) {
      console.error('Error updating Gemini config:', error);
      res.status(500).json({ error: 'Failed to update config: ' + error.message });
    }
  });

  // Einheitlicher LLM-Status
  app.get('/api/llm/status', async (req, res) => {
    try {
      logRESTAPIRequest('get-llm-status', req.body);
      
      const { getLLMConfig, testOllamaConnection, testGeminiConnection } = await import('./llm.js');
      const config = getLLMConfig();
      
      // Teste beide Verbindungen
      const [ollamaResult, geminiResult] = await Promise.allSettled([
        testOllamaConnection(),
        testGeminiConnection()
      ]);
      
      const ollamaStatus = ollamaResult.status === 'fulfilled' ? ollamaResult.value : {
        success: false,
        connected: false,
        error: ollamaResult.reason?.message || 'Test fehlgeschlagen'
      };
      
      const geminiStatus = geminiResult.status === 'fulfilled' ? geminiResult.value : {
        success: false,
        connected: false,
        error: geminiResult.reason?.message || 'Test fehlgeschlagen'
      };
      
      res.json({
        success: true,
        activeProvider: config.activeProvider,
        ollama: {
          connected: ollamaStatus.connected || false,
          available: ollamaStatus.success || false,
          error: ollamaStatus.error || null,
          model: config.ollama.defaultModel
        },
        gemini: {
          connected: geminiStatus.connected || false,
          available: geminiStatus.success || false,
          hasApiKey: !!config.gemini.apiKey,
          error: geminiStatus.error || null,
          model: config.gemini.defaultModel
        },
        lastTested: new Date().toISOString()
      });
    } catch (error) {
      console.error('Error getting LLM status:', error);
      res.status(500).json({ error: 'Failed to get status: ' + error.message });
    }
  });

  // ===== VERALTETE ENDPOINTS (für Kompatibilität) =====

  // Veralteter Gemini-Status-Endpoint (Kompatibilität)
  app.get('/api/gemini/status', async (req, res) => {
    try {
      logRESTAPIRequest('gemini-status-legacy', req.body);
      const { getLLMConfig, testGeminiConnection } = await import('./llm.js');
      
      const config = getLLMConfig();
      const result = await testGeminiConnection();
      
      res.json({ 
        connected: result.connected, 
        error: result.error,
        hasApiKey: !!config.gemini.apiKey 
      });
    } catch (error) {
      console.error('Gemini status check error:', error);
      res.status(500).json({ error: 'Status check failed: ' + error.message });
    }
  });

  // API-Key setzen (temporär für die Session)
  app.post('/api/gemini/api-key', async (req, res) => {
    try {
      logRESTAPIRequest('set-gemini-api-key', req.body);
      const { apiKey } = req.body;
      
      if (!apiKey || typeof apiKey !== 'string') {
        return res.status(400).json({ error: 'Gültiger API-Key erforderlich' });
      }

      // Setze den API-Key als Umgebungsvariable
      process.env.GEMINI_API_KEY = apiKey;
      
      // Teste die Verbindung mit dem neuen API-Key
      try {
        const { callLLMAPI } = await import('./llm.js');
        const testResponse = await callLLMAPI('Antworte nur mit "OK" wenn du diese Nachricht erhältst.');
        
        const isConnected = testResponse && testResponse.toLowerCase().includes('ok');
        
        if (isConnected) {
          res.json({ 
            success: true, 
            connected: true,
            message: 'API-Key erfolgreich gesetzt und getestet'
          });
        } else {
          res.json({ 
            success: false, 
            connected: false,
            error: 'API-Key gesetzt, aber Verbindung fehlgeschlagen'
          });
        }
      } catch (error) {
        res.json({ 
          success: false, 
          connected: false,
          error: 'API-Key gesetzt, aber Test fehlgeschlagen: ' + error.message
        });
      }
    } catch (error) {
      console.error('API-Key setup error:', error);
      res.status(500).json({ error: 'API-Key setup failed: ' + error.message });
    }
  });

  // Aktuellen API-Key-Status abrufen (ohne den Key preiszugeben)
  app.get('/api/gemini/api-key-status', (req, res) => {
    logRESTAPIRequest('get-gemini-api-key-status', req.body);
    const API_KEY = process.env.GEMINI_API_KEY;
    const hasApiKey = Boolean(API_KEY && API_KEY.length > 0);
    const keyPreview = hasApiKey ? `${API_KEY.substring(0, 8)}...${API_KEY.substring(API_KEY.length - 4)}` : null;
    
    res.json({ 
      hasApiKey,
      keyPreview
    });
  });

  // Verfügbare Gemini-Modelle abrufen
  app.get('/api/gemini/models', async (req, res) => {
    try {
      logRESTAPIRequest('get-gemini-models', req.body);
      const { getAvailableGeminiModels, getCurrentGeminiModel } = await import('./llm.js');
      const availableModels = getAvailableGeminiModels();
      const currentModel = getCurrentGeminiModel();
      
      res.json({
        availableModels,
        currentModel,
        customModelSupported: true
      });
    } catch (error) {
      console.error('Error getting Gemini models:', error);
      res.status(500).json({ error: 'Failed to get models: ' + error.message });
    }
  });

  // Gemini-Modell setzen
  app.post('/api/gemini/model', async (req, res) => {
    try {
      logRESTAPIRequest('set-gemini-model', req.body);
      const { model } = req.body;
      
      if (!model || typeof model !== 'string') {
        return res.status(400).json({ error: 'Gültiges Modell erforderlich' });
      }

      const { setCurrentGeminiModel, getAvailableGeminiModels } = await import('./llm.js');
      const availableModels = getAvailableGeminiModels();
      
      // Prüfe ob es ein vordefiniertes Modell ist oder ein custom Model
      const isCustomModel = !availableModels.includes(model);
      
      // Setze das Modell
      setCurrentGeminiModel(model);
      
      // Teste das Modell mit einem einfachen Prompt
      try {
        const { callLLMAPI } = await import('./llm.js');
        const testResponse = await callLLMAPI('Antworte nur mit "OK" wenn du diese Nachricht erhältst.', null, model);
        
        const isWorking = testResponse && testResponse.toLowerCase().includes('ok');
        
        res.json({
          success: true,
          model,
          isCustomModel,
          tested: true,
          working: isWorking,
          message: isWorking ? 'Modell erfolgreich gesetzt und getestet' : 'Modell gesetzt, aber Test fehlgeschlagen'
        });
      } catch (error) {
        res.json({
          success: true,
          model,
          isCustomModel,
          tested: false,
          working: false,
          error: 'Modell gesetzt, aber Test fehlgeschlagen: ' + error.message
        });
      }
    } catch (error) {
      console.error('Error setting Gemini model:', error);
      res.status(500).json({ error: 'Failed to set model: ' + error.message });
    }
  });

  // Aktuelles Gemini-Modell abrufen
  app.get('/api/gemini/current-model', async (req, res) => {
    try {
      logRESTAPIRequest('get-gemini-current-model', req.body);
      const { getCurrentGeminiModel, getAvailableGeminiModels } = await import('./llm.js');
      const currentModel = getCurrentGeminiModel();
      const availableModels = getAvailableGeminiModels();
      const isCustomModel = !availableModels.includes(currentModel);
      
      res.json({
        currentModel,
        isCustomModel
      });
    } catch (error) {
      console.error('Error getting current Gemini model:', error);
      res.status(500).json({ error: 'Failed to get current model: ' + error.message });
    }
  });

}

// Prediction-Endpoint für echte API-Nutzung
export function setupPredictionEndpoint(app, scriptDir, venvDir) {
  app.post('/api/predict/:id', async (req, res) => {
    try {
      logRESTAPIRequest('predict', req.body);
      const { id } = req.params;
      const inputs = req.body;

      console.log('Inputs:', inputs);

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

  // ===== LLM QUEUE MANAGEMENT ENDPOINTS =====

  // LLM Queue Status abrufen
  app.get('/api/llm/queue/status', async (req, res) => {
    await logRESTAPIRequest('GET', '/api/llm/queue/status');
    
    try {
      const status = getLLMQueueStatus();
      res.json({ 
        success: true, 
        status,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('Queue Status Error:', error);
      res.status(500).json({ 
        success: false, 
        error: 'Failed to get queue status: ' + error.message 
      });
    }
  });

  // LLM Request abbrechen
  app.post('/api/llm/queue/cancel/:requestId', async (req, res) => {
    await logRESTAPIRequest('POST', '/api/llm/queue/cancel/:requestId');
    
    try {
      const { requestId } = req.params;
      const { reason = 'User cancelled' } = req.body;
      
      const cancelled = cancelLLMRequest(parseInt(requestId), reason);
      
      if (cancelled) {
        res.json({ 
          success: true, 
          message: `Request ${requestId} cancelled`,
          requestId: parseInt(requestId)
        });
      } else {
        res.status(404).json({ 
          success: false, 
          error: `Request ${requestId} not found or already completed` 
        });
      }
    } catch (error) {
      console.error('Cancel Request Error:', error);
      res.status(500).json({ 
        success: false, 
        error: 'Failed to cancel request: ' + error.message 
      });
    }
  });
}
