import { logRESTAPIRequest } from '../../monitoring/log.js';
import { clearAnalysisCache, getAnalysisCacheStatus } from '../../data/data_exploration.js';

export function setupCacheRoutes(app) {
  // File-Cache (legacy – noop)
  app.post('/api/cache/clear', async (req, res) => {
    try {
      logRESTAPIRequest('clear-cache', req.body);
      res.json({ message: 'File-Cache wurde entfernt - keine Aktion erforderlich' });
    } catch (error) {
      res.status(500).json({ error: 'Fehler beim Löschen des Caches: ' + error.message });
    }
  });

  app.get('/api/cache/status', async (req, res) => {
    try {
      logRESTAPIRequest('get-cache-status', req.body);
      res.json({ message: 'File-Cache wurde entfernt', cachedFiles: [], cacheSize: 0 });
    } catch (error) {
      res.status(500).json({ error: 'Fehler beim Abrufen des Cache-Status: ' + error.message });
    }
  });

  // Analyse-Cache
  app.post('/api/analysis-cache/clear', async (req, res) => {
    try {
      logRESTAPIRequest('clear-analysis-cache', req.body);
      await clearAnalysisCache();
      res.json({ message: 'Datenanalyse-Cache erfolgreich geleert' });
    } catch (error) {
      res.status(500).json({ error: 'Fehler beim Löschen des Datenanalyse-Caches: ' + error.message });
    }
  });

  app.get('/api/analysis-cache/status', async (req, res) => {
    try {
      logRESTAPIRequest('get-analysis-cache-status', req.body);
      const status = await getAnalysisCacheStatus();
      res.json(status);
    } catch (error) {
      res.status(500).json({ error: 'Fehler beim Abrufen des Datenanalyse-Cache-Status: ' + error.message });
    }
  });
}


