import { logRESTAPIRequest } from '../../monitoring/log.js';

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
      const { getLLMConfig } = await import('../../llm/llm.js');
      const config = getLLMConfig();
      res.json({ success: true, config });
    } catch (error) {
      res.status(500).json({ error: 'Failed to get config: ' + error.message });
    }
  });

  // Ollama-spezifische Endpoints
  app.get('/api/llm/ollama/models', async (req, res) => {
    try {
      logRESTAPIRequest('get-ollama-models', req.body);
      const { getAvailableOllamaModels } = await import('../../llm/llm.js');
      const result = await getAvailableOllamaModels();
      res.json(result);
    } catch (error) {
      res.status(500).json({ error: 'Failed to get models: ' + error.message });
    }
  });

  app.post('/api/llm/ollama/test', async (req, res) => {
    try {
      logRESTAPIRequest('test-ollama-connection', req.body);
      const { testOllamaConnection } = await import('../../llm/llm.js');
      const result = await testOllamaConnection();
      res.json(result);
    } catch (error) {
      res.status(500).json({ error: 'Failed to test connection: ' + error.message });
    }
  });

  app.post('/api/llm/ollama/config', async (req, res) => {
    try {
      logRESTAPIRequest('update-ollama-config', req.body);
      const { host, defaultModel } = req.body;
      const { updateOllamaConfig } = await import('../../llm/llm.js');
      const config = {};
      if (host) config.host = host;
      if (defaultModel) config.defaultModel = defaultModel;
      updateOllamaConfig(config);
      res.json({ success: true, message: 'Ollama-Konfiguration aktualisiert', config });
    } catch (error) {
      res.status(500).json({ error: 'Failed to update config: ' + error.message });
    }
  });


  // Einheitlicher LLM-Status
  app.get('/api/llm/status', async (req, res) => {
    try {
      logRESTAPIRequest('get-llm-status', req.body);
      const { getLLMConfig, testOllamaConnection } = await import('../../llm/llm.js');
      const config = getLLMConfig();
      const ollamaResult = await testOllamaConnection();
      const ollamaStatus = { connected: ollamaResult.connected || false, available: ollamaResult.success || false, error: ollamaResult.error || null, model: config.ollama.defaultModel };
      res.json({ success: true, activeProvider: config.activeProvider, ollama: ollamaStatus, lastTested: new Date().toISOString() });
    } catch (error) {
      res.status(500).json({ error: 'Failed to get status: ' + error.message });
    }
  });

}


