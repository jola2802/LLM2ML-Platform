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
            const { getLLMConfig } = await import('../llm/api/llm.js');
            const config = getLLMConfig();
            res.json({ success: true, config });
        } catch (error) {
            res.status(500).json({ error: 'Failed to get config: ' + error.message });
        }
    });

    // Ollama-spezifische Endpoints
    app.get('/api/llm/ollama/models', async (req, res) => {
        try {
            const { getAvailableOllamaModels } = await import('../llm/api/llm.js');
            const result = await getAvailableOllamaModels();
            res.json(result);
        } catch (error) {
            res.status(500).json({ error: 'Failed to get models: ' + error.message });
        }
    });

    app.post('/api/llm/ollama/test', async (req, res) => {
        try {
            const { testOllamaConnection } = await import('../llm/api/llm.js');
            const result = await testOllamaConnection();
            res.json(result);
        } catch (error) {
            res.status(500).json({ error: 'Failed to test connection: ' + error.message });
        }
    });

    app.post('/api/llm/ollama/config', async (req, res) => {
        try {
            const { host, defaultModel } = req.body;
            const { updateOllamaConfig } = await import('../llm/api/llm.js');
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
            const { getLLMConfig, testOllamaConnection } = await import('../llm/api/llm.js');
            const config = getLLMConfig();
            const ollamaResult = await testOllamaConnection();
            const ollamaStatus = {
                connected: ollamaResult.connected || false,
                available: ollamaResult.success || false,
                error: ollamaResult.error || null,
                model: config.ollama.defaultModel
            };
            res.json({
                success: true,
                activeProvider: config.activeProvider,
                ollama: ollamaStatus,
                lastTested: new Date().toISOString()
            });
        } catch (error) {
            res.status(500).json({ error: 'Failed to get status: ' + error.message });
        }
    });

    // LLM-API-Call direkt
    app.post('/api/llm/call', async (req, res) => {
        try {
            const { prompt, filePath, customModel, maxRetries } = req.body;

            if (!prompt) {
                return res.status(400).json({ error: 'prompt ist erforderlich' });
            }

            const { callLLMAPI } = await import('../llm/api/llm.js');
            const result = await callLLMAPI(prompt, filePath || null, customModel || null, maxRetries || 3);

            res.json({ success: true, result });
        } catch (error) {
            console.error('Fehler bei LLM-API-Call:', error);
            res.status(500).json({ error: 'LLM-API-Call fehlgeschlagen: ' + error.message });
        }
    });

    // LLM-Empfehlungen für ML-Pipeline
    app.post('/api/llm/recommendations', async (req, res) => {
        try {
            const { analysis, filePath, selectedFeatures, excludedFeatures, userPreferences } = req.body;

            if (!analysis) {
                return res.status(400).json({ error: 'analysis ist erforderlich' });
            }

            const { getLLMRecommendations } = await import('../llm/api/llm_api.js');
            const recommendations = await getLLMRecommendations(
                analysis,
                filePath || null,
                selectedFeatures || null,
                excludedFeatures || null,
                userPreferences || null
            );

            res.json({ success: true, recommendations });
        } catch (error) {
            console.error('Fehler bei LLM-Empfehlungen:', error);
            res.status(500).json({ error: 'LLM-Empfehlungen fehlgeschlagen: ' + error.message });
        }
    });

    // Performance-Evaluation mit LLM
    app.post('/api/llm/evaluate-performance', async (req, res) => {
        try {
            const { project } = req.body;

            if (!project) {
                return res.status(400).json({ error: 'project ist erforderlich' });
            }

            const { evaluatePerformanceWithLLM } = await import('../llm/api/llm_api.js');
            const evaluation = await evaluatePerformanceWithLLM(project);

            res.json({ success: true, evaluation });
        } catch (error) {
            console.error('Fehler bei Performance-Evaluation:', error);
            res.status(500).json({ error: 'Performance-Evaluation fehlgeschlagen: ' + error.message });
        }
    });
}

