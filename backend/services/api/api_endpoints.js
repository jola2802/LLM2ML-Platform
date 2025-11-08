import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { pythonClient } from '../clients/python_client.js';

// Service-Imports
import { masClient } from '../clients/mas_client.js';
import { getProject, updateProjectCode, updateProjectHyperparameters, updateProjectStatus, updateProjectInsights, updateProjectAlgorithmAndHyperparameters } from '../database/db.js';
import { analyzeCsvFile, analyzeJsonFile, analyzeExcelFile, analyzeTextFile, analyzeGenericFile } from '../data/file_analysis.js';
import { logRESTAPIRequest } from '../monitoring/log.js';
import { logPredictionEvent } from '../monitoring/monitoring.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let trainModelAsync, retrainModelAsync;

export function setTrainingFunctions(trainFn, retrainFn) {
  trainModelAsync = trainFn;
  retrainModelAsync = retrainFn;
}

export function setupAPIEndpoints(app, upload, scriptDir, venvDir) {
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

  // Auto-Tune: LLM schlägt bessere Konfiguration vor
  app.post('/api/projects/:id/auto-tune', async (req, res) => {
    try {
      const { id } = req.params;
      const { iterations = 2, apply = false } = req.body || {};
      const project = await getProject(id);
      if (!project) return res.status(404).json({ error: 'Projekt nicht gefunden' });
      // Auto-Tuning über MAS-Service
      const { masClient } = await import('../clients/mas_client.js');
      const proposal = await masClient.autoTuneModel(project, Math.max(1, Math.min(iterations, 5)));

      if (apply) {
        // Update DB mit vorgeschlagenem Algorithmus/Hyperparametern
        await updateProjectAlgorithmAndHyperparameters(id, proposal.algorithm, proposal.hyperparameters);
        // Optional: sofort re-train starten
        if (trainModelAsync) {
          // Status setzen
          await updateProjectStatus(id, 'Re-Training');
          trainModelAsync(id);
        }
      }

      res.json({ success: true, proposal, applied: apply });
    } catch (error) {
      res.status(500).json({ success: false, error: error.message });
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
      const performanceInsights = await masClient.evaluatePerformance(project);

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

      // Prüfe ob Dateipfad vorhanden ist
      if (!project.csvFilePath) {
        return res.status(400).json({ error: 'Keine Datei für dieses Projekt verfügbar' });
      }

      // Datei analysieren für detaillierte Statistiken (ohne zusätzliche LLM-Kosten)
      const fileExtension = path.extname(project.csvFilePath).toLowerCase();
      let analysis;

      if (fileExtension === '.csv') {
        analysis = await analyzeCsvFile(project.csvFilePath, false);
      } else if (fileExtension === '.json') {
        analysis = await analyzeJsonFile(project.csvFilePath, false);
      } else if (fileExtension === '.xlsx' || fileExtension === '.xls') {
        analysis = await analyzeExcelFile(project.csvFilePath, false);
      } else if (fileExtension === '.txt') {
        analysis = await analyzeTextFile(project.csvFilePath, false);
      } else {
        analysis = await analyzeGenericFile(project.csvFilePath, fileExtension, false);
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
        return res.status(400).json({ error: 'Keine Datei für dieses Projekt verfügbar' });
      }

      const fileExtension = path.extname(project.csvFilePath).toLowerCase();


      if (fileExtension === '.csv') {
        analysis = await analyzeCsvFile(project.csvFilePath, false);
      } else if (fileExtension === '.json') {
        analysis = await analyzeJsonFile(project.csvFilePath, false);
      } else if (fileExtension === '.xlsx' || fileExtension === '.xls') {
        analysis = await analyzeExcelFile(project.csvFilePath, false);
      } else if (fileExtension === '.txt') {
        analysis = await analyzeTextFile(project.csvFilePath, false);
      } else {
        analysis = await analyzeGenericFile(project.csvFilePath, fileExtension, false);
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

  // Datenexploration-Cache leeren (über Python-Service)
  app.post('/api/analysis-cache/clear', async (req, res) => {
    try {
      logRESTAPIRequest('clear-analysis-cache', req.body);
      // Cache wird über Python-Service verwaltet
      const response = await fetch(`${process.env.PYTHON_SERVICE_URL || 'http://localhost:3003'}/api/data/cache/clear`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const result = await response.json();
      res.json({ message: 'Datenanalyse-Cache erfolgreich geleert', ...result });
    } catch (error) {
      console.error('Fehler beim Löschen des Datenanalyse-Caches:', error);
      res.status(500).json({ error: 'Fehler beim Löschen des Datenanalyse-Caches: ' + error.message });
    }
  });

  // Datenexploration-Cache-Status abrufen (über Python-Service)
  app.get('/api/analysis-cache/status', async (req, res) => {
    try {
      logRESTAPIRequest('get-analysis-cache-status', req.body);
      // Cache wird über Python-Service verwaltet
      const response = await fetch(`${process.env.PYTHON_SERVICE_URL || 'http://localhost:3003'}/api/data/cache/status`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      const status = await response.json();
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

      const analysis = await pythonClient.analyzeData(filePath, false);
      res.json(analysis);

    } catch (error) {
      console.error('Fehler bei der Datenexploration:', error);
      res.status(500).json({ error: 'Fehler bei der Datenexploration: ' + error.message });
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

      // console.log('Inputs:', inputs);

      const project = await getProject(id);
      if (!project) {
        return res.status(404).json({ error: 'Project not found' });
      }

      const result = await pythonClient.executePrediction(project, inputs);
      const prediction = result.prediction;

      // Monitoring-Event loggen (ohne truth)
      try { await logPredictionEvent(id, { features: inputs, prediction }); } catch { }

      res.json({ prediction });
    } catch (error) {
      console.error('Prediction error:', error);
      res.status(500).json({ error: 'Prediction failed: ' + error.message });
    }
  });

  // LLM Queue-Management ist ausgelagert in routes/queue.js
}
