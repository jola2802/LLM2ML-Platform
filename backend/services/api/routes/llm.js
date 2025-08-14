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

  // Aktiven Provider setzen
  app.post('/api/llm/provider', async (req, res) => {
    try {
      logRESTAPIRequest('set-llm-provider', req.body);
      const { provider } = req.body;
      if (!provider) return res.status(400).json({ error: 'Provider erforderlich' });
      const { setActiveProvider, LLM_PROVIDERS } = await import('../../llm/llm.js');
      if (!Object.values(LLM_PROVIDERS).includes(provider)) return res.status(400).json({ error: 'Ungültiger Provider' });
      setActiveProvider(provider);
      res.json({ success: true, message: `Provider auf ${provider} gesetzt`, provider });
    } catch (error) {
      res.status(500).json({ error: 'Failed to set provider: ' + error.message });
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

  // Gemini-spezifische Endpoints
  app.post('/api/llm/gemini/test', async (req, res) => {
    try {
      logRESTAPIRequest('test-gemini-connection', req.body);
      const { testGeminiConnection } = await import('../../llm/llm.js');
      const result = await testGeminiConnection();
      res.json(result);
    } catch (error) {
      res.status(500).json({ error: 'Failed to test connection: ' + error.message });
    }
  });

  app.post('/api/llm/gemini/config', async (req, res) => {
    try {
      logRESTAPIRequest('update-gemini-config', req.body);
      const { apiKey, defaultModel } = req.body;
      const { updateGeminiConfig } = await import('../../llm/llm.js');
      const config = {};
      if (apiKey) config.apiKey = apiKey;
      if (defaultModel) config.defaultModel = defaultModel;
      updateGeminiConfig(config);
      res.json({ success: true, message: 'Gemini-Konfiguration aktualisiert', config: { ...config, apiKey: config.apiKey ? `${config.apiKey.substring(0, 8)}...` : null } });
    } catch (error) {
      res.status(500).json({ error: 'Failed to update config: ' + error.message });
    }
  });

  // Einheitlicher LLM-Status
  app.get('/api/llm/status', async (req, res) => {
    try {
      logRESTAPIRequest('get-llm-status', req.body);
      const { getLLMConfig, testOllamaConnection, testGeminiConnection } = await import('../../llm/llm.js');
      const config = getLLMConfig();
      const [ollamaResult, geminiResult] = await Promise.allSettled([ testOllamaConnection(), testGeminiConnection() ]);
      const ollamaStatus = ollamaResult.status === 'fulfilled' ? ollamaResult.value : { success: false, connected: false, error: ollamaResult.reason?.message || 'Test fehlgeschlagen' };
      const geminiStatus = geminiResult.status === 'fulfilled' ? geminiResult.value : { success: false, connected: false, error: geminiResult.reason?.message || 'Test fehlgeschlagen' };
      res.json({ success: true, activeProvider: config.activeProvider, ollama: { connected: ollamaStatus.connected || false, available: ollamaStatus.success || false, error: ollamaStatus.error || null, model: config.ollama.defaultModel }, gemini: { connected: geminiStatus.connected || false, available: geminiStatus.success || false, hasApiKey: !!config.gemini.apiKey, error: geminiStatus.error || null, model: config.gemini.defaultModel }, lastTested: new Date().toISOString() });
    } catch (error) {
      res.status(500).json({ error: 'Failed to get status: ' + error.message });
    }
  });

  // Legacy Gemini Endpoints (Kompatibilität)
  app.get('/api/gemini/status', async (req, res) => {
    try {
      logRESTAPIRequest('gemini-status-legacy', req.body);
      const { getLLMConfig, testGeminiConnection } = await import('../../llm/llm.js');
      const config = getLLMConfig();
      const result = await testGeminiConnection();
      res.json({ connected: result.connected, error: result.error, hasApiKey: !!config.gemini.apiKey });
    } catch (error) {
      res.status(500).json({ error: 'Status check failed: ' + error.message });
    }
  });

  app.post('/api/gemini/api-key', async (req, res) => {
    try {
      logRESTAPIRequest('set-gemini-api-key', req.body);
      const { apiKey } = req.body;
      if (!apiKey || typeof apiKey !== 'string') return res.status(400).json({ error: 'Gültiger API-Key erforderlich' });
      process.env.GEMINI_API_KEY = apiKey;
      try {
        const { callLLMAPI } = await import('../../llm/llm.js');
        const testResponse = await callLLMAPI('Antworte nur mit "OK" wenn du diese Nachricht erhältst.');
        const isConnected = testResponse && testResponse.toLowerCase().includes('ok');
        res.json({ success: isConnected, connected: isConnected, message: isConnected ? 'API-Key erfolgreich gesetzt und getestet' : undefined, error: isConnected ? undefined : 'API-Key gesetzt, aber Verbindung fehlgeschlagen' });
      } catch (error) {
        res.json({ success: false, connected: false, error: 'API-Key gesetzt, aber Test fehlgeschlagen: ' + error.message });
      }
    } catch (error) {
      res.status(500).json({ error: 'API-Key setup failed: ' + error.message });
    }
  });

  app.get('/api/gemini/api-key-status', (req, res) => {
    logRESTAPIRequest('get-gemini-api-key-status', req.body);
    const API_KEY = process.env.GEMINI_API_KEY;
    const hasApiKey = Boolean(API_KEY && API_KEY.length > 0);
    const keyPreview = hasApiKey ? `${API_KEY.substring(0, 8)}...${API_KEY.substring(API_KEY.length - 4)}` : null;
    res.json({ hasApiKey, keyPreview });
  });

  app.get('/api/gemini/models', async (req, res) => {
    try {
      logRESTAPIRequest('get-gemini-models', req.body);
      const { getAvailableGeminiModels, getCurrentGeminiModel } = await import('../../llm/llm.js');
      const availableModels = getAvailableGeminiModels();
      const currentModel = getCurrentGeminiModel();
      res.json({ availableModels, currentModel, customModelSupported: true });
    } catch (error) {
      res.status(500).json({ error: 'Failed to get models: ' + error.message });
    }
  });

  app.post('/api/gemini/model', async (req, res) => {
    try {
      logRESTAPIRequest('set-gemini-model', req.body);
      const { model } = req.body;
      if (!model || typeof model !== 'string') return res.status(400).json({ error: 'Gültiges Modell erforderlich' });
      const { setCurrentGeminiModel, getAvailableGeminiModels } = await import('../../llm/llm.js');
      const availableModels = getAvailableGeminiModels();
      const isCustomModel = !availableModels.includes(model);
      setCurrentGeminiModel(model);
      try {
        const { callLLMAPI } = await import('../../llm/llm.js');
        const testResponse = await callLLMAPI('Antworte nur mit "OK" wenn du diese Nachricht erhältst.', null, model);
        const isWorking = testResponse && testResponse.toLowerCase().includes('ok');
        res.json({ success: true, model, isCustomModel, tested: true, working: isWorking, message: isWorking ? 'Modell erfolgreich gesetzt und getestet' : 'Modell gesetzt, aber Test fehlgeschlagen' });
      } catch (error) {
        res.json({ success: true, model, isCustomModel, tested: false, working: false, error: 'Modell gesetzt, aber Test fehlgeschlagen: ' + error.message });
      }
    } catch (error) {
      res.status(500).json({ error: 'Failed to set model: ' + error.message });
    }
  });

  app.get('/api/gemini/current-model', async (req, res) => {
    try {
      logRESTAPIRequest('get-gemini-current-model', req.body);
      const { getCurrentGeminiModel, getAvailableGeminiModels } = await import('../../llm/llm.js');
      const currentModel = getCurrentGeminiModel();
      const availableModels = getAvailableGeminiModels();
      const isCustomModel = !availableModels.includes(currentModel);
      res.json({ currentModel, isCustomModel });
    } catch (error) {
      res.status(500).json({ error: 'Failed to get current model: ' + error.message });
    }
  });
}


