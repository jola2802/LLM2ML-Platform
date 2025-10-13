import path from 'path';
import fs from 'fs/promises';
import { logRESTAPIRequest } from '../../monitoring/log.js';
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
} from '../../database/db.js';
import { evaluatePerformanceWithLLM } from '../../llm/api/llm_api.js';

export function setupProjectRoutes(app, scriptDir, venvDir, trainModelAsync, retrainModelAsync) {
  // Intelligentes Projekt erstellen (mit LLM-Empfehlungen)
  app.post('/api/projects/smart-create', async (req, res) => {
    try {
      logRESTAPIRequest('smart-create-project', req.body);
      const { name, csvFilePath, recommendations } = req.body;
      const project = await createSmartProject({ name, csvFilePath, recommendations });
      res.json(project);
      if (trainModelAsync) trainModelAsync(project.id);
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
      if (!project) return res.status(404).json({ error: 'Project not found' });
      res.json(project);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // Neues Projekt erstellen
  app.post('/api/projects', async (req, res) => {
    try {
      logRESTAPIRequest('create-project', req.body);
      const { name, modelType, dataSourceName, targetVariable, features, csvFilePath, algorithm, hyperparameters } = req.body;
      const project = await createProject({ name, modelType, dataSourceName, targetVariable, features, csvFilePath, algorithm, hyperparameters });
      res.json(project);
      if (trainModelAsync) trainModelAsync(project.id);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // Modell herunterladen
  app.get('/api/projects/:id/download', async (req, res) => {
    try {
      logRESTAPIRequest('download-project', req.params.id);
      const project = await getProject(req.params.id);
      if (!project) return res.status(404).json({ error: 'Project not found' });
      if (!project.modelPath) return res.status(404).json({ error: 'Model file not found' });
      const modelPath = path.join(path.dirname(__dirname), project.modelPath);
      try { await fs.access(modelPath); } catch { return res.status(404).json({ error: 'Model file not found on disk' }); }
      res.setHeader('Content-Disposition', `attachment; filename="${project.name}_model.pkl"`);
      res.setHeader('Content-Type', 'application/octet-stream');
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
        try {
          await fs.unlink(project.csvFilePath);
          await fs.unlink(path.join(path.dirname(__dirname), project.modelPath));
          const scriptsDir = path.join(path.dirname(__dirname), 'scripts');
          const scriptFiles = await fs.readdir(scriptsDir);
          for (const file of scriptFiles) {
            if (file.includes(id)) await fs.unlink(path.join(scriptsDir, file));
          }
        } catch (err) {
          console.log('Could not delete model/script files:', err.message);
        }
      }
      const result = await deleteProject(id);
      if (result.changes === 0) return res.status(404).json({ error: 'Project not found' });
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
      if (!project) return res.status(404).json({ error: 'Project not found' });
      await updateProjectCode(id, pythonCode);
      if (hyperparameters) await updateProjectHyperparameters(id, hyperparameters);
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
      if (!project) return res.status(404).json({ error: 'Project not found' });
      if (!project.pythonCode) return res.status(400).json({ error: 'Kein Python-Code zum Re-Training verfügbar' });
      await updateProjectStatus(id, 'Re-Training');
      res.json({ message: 'Re-Training gestartet', id });
      if (retrainModelAsync) retrainModelAsync(id, project.pythonCode);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // Performance-Evaluation (manuell – existiert weiter, wird aber sonst automatisch angestoßen)
  app.post('/api/projects/:id/evaluate-performance', async (req, res) => {
    try {
      logRESTAPIRequest('evaluate-performance', req.params.id);
      const { id } = req.params;
      const project = await getProject(id);
      if (!project) return res.status(404).json({ error: 'Projekt nicht gefunden' });
      if (!project.performanceMetrics) return res.status(400).json({ error: 'Keine Performance-Metriken verfügbar für Evaluation' });
      const performanceInsights = await evaluatePerformanceWithLLM(project);
      await updateProjectInsights(id, performanceInsights);
      res.json({ message: 'Performance-Evaluation erfolgreich abgeschlossen', insights: performanceInsights });
    } catch (error) {
      console.error('Fehler bei Performance-Evaluation:', error);
      res.status(500).json({ error: 'Performance-Evaluation fehlgeschlagen: ' + error.message });
    }
  });
}


