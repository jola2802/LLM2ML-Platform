import { logRESTAPIRequest } from '../../monitoring/log.js';
import { masClient } from '../../clients/mas_client.js';

export function setupLLMRoutes(app) {
  // Test-Endpoint für LLM-API
  app.get('/api/llm/test', async (req, res) => {
    try {
      res.json({ success: true, message: 'LLM API funktioniert', timestamp: new Date().toISOString() });
    } catch (error) {
      res.status(500).json({ error: 'Test failed: ' + error.message });
    }
  });

  // Aktuelle LLM-Konfiguration abrufen
  app.get('/api/llm/config', async (req, res) => {
    try {
      logRESTAPIRequest('get-llm-config', req.body);
      const config = await masClient.getLLMConfig();
      res.json({ success: true, config });
    } catch (error) {
      res.status(500).json({ error: 'Failed to get config: ' + error.message });
    }
  });

  // Ollama-spezifische Endpoints
  app.get('/api/llm/ollama/models', async (req, res) => {
    try {
      logRESTAPIRequest('get-ollama-models', req.body);
      const result = await masClient.getAvailableOllamaModels();
      res.json(result);
    } catch (error) {
      res.status(500).json({ error: 'Failed to get models: ' + error.message });
    }
  });

  app.post('/api/llm/ollama/test', async (req, res) => {
    try {
      logRESTAPIRequest('test-ollama-connection', req.body);
      const result = await masClient.testOllamaConnection();
      res.json(result);
    } catch (error) {
      res.status(500).json({ error: 'Failed to test connection: ' + error.message });
    }
  });

  app.post('/api/llm/ollama/config', async (req, res) => {
    try {
      logRESTAPIRequest('update-ollama-config', req.body);
      // TODO: Update-Ollama-Config über MAS-Service implementieren
      res.json({ success: true, message: 'Ollama-Konfiguration aktualisiert (über MAS-Service)' });
    } catch (error) {
      res.status(500).json({ error: 'Failed to update config: ' + error.message });
    }
  });

  // Einheitlicher LLM-Status
  app.get('/api/llm/status', async (req, res) => {
    try {
      logRESTAPIRequest('get-llm-status', req.body);
      const status = await masClient.getLLMStatus();
      res.json(status);
    } catch (error) {
      res.status(500).json({ error: 'Failed to get status: ' + error.message });
    }
  });
}


