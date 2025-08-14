import { logRESTAPIRequest } from '../../monitoring/log.js';
import { initializeMonitoringBaseline, logPredictionEvent, getMonitoringStatus, clearMonitoring } from '../../monitoring/monitoring.js';

export function setupMonitoringRoutes(app) {
  // Monitoring: Baseline initialisieren
  app.post('/api/projects/:id/monitoring/init', async (req, res) => {
    try {
      const { id } = req.params;
      logRESTAPIRequest('monitoring-init', { id });
      const baseline = await initializeMonitoringBaseline(id);
      res.json({ success: true, baseline });
    } catch (error) {
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // Monitoring: Prediction-Event loggen (optional mit truth)
  app.post('/api/projects/:id/monitoring/event', async (req, res) => {
    try {
      const { id } = req.params;
      logRESTAPIRequest('monitoring-event', { id });
      const result = await logPredictionEvent(id, req.body || {});
      res.json({ success: true, ...result });
    } catch (error) {
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // Monitoring: Status abrufen
  app.get('/api/projects/:id/monitoring/status', async (req, res) => {
    try {
      const { id } = req.params;
      logRESTAPIRequest('monitoring-status', { id });
      const status = await getMonitoringStatus(id);
      res.json(status);
    } catch (error) {
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // Monitoring: Zurücksetzen
  app.post('/api/projects/:id/monitoring/reset', async (req, res) => {
    try {
      const { id } = req.params;
      logRESTAPIRequest('monitoring-reset', { id });
      const result = await clearMonitoring(id);
      res.json(result);
    } catch (error) {
      res.status(500).json({ success: false, error: error.message });
    }
  });
}


