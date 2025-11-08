export function setupAgentRoutes(app) {
    // Alle verfügbaren Agents abrufen
    app.get('/api/agents', async (req, res) => {
        try {
            const {
                ALL_AGENTS,
                getAgentStats
            } = await import('../llm/agents/config_agent_network.js');

            const agents = Object.entries(ALL_AGENTS).map(([key, config]) => ({
                key,
                ...config
            }));
            const stats = getAgentStats();

            res.json({
                success: true,
                agents,
                stats,
                totalAgents: agents.length
            });
        } catch (error) {
            console.error('Fehler beim Abrufen der Agents:', error);
            res.status(500).json({
                success: false,
                error: 'Fehler beim Abrufen der Agents: ' + error.message
            });
        }
    });

    // Spezifische Agent-Konfiguration abrufen
    app.get('/api/agents/:agentKey', async (req, res) => {
        try {
            const { agentKey } = req.params;
            const {
                getAgentConfig,
                isValidAgent
            } = await import('../llm/agents/config_agent_network.js');

            if (!isValidAgent(agentKey)) {
                return res.status(404).json({
                    success: false,
                    error: `Agent '${agentKey}' nicht gefunden`
                });
            }

            const config = getAgentConfig(agentKey);
            res.json({ success: true, agent: { key: agentKey, ...config } });
        } catch (error) {
            console.error('Fehler beim Abrufen der Agent-Konfiguration:', error);
            res.status(500).json({
                success: false,
                error: 'Fehler beim Abrufen der Agent-Konfiguration: ' + error.message
            });
        }
    });

    // Master-Agent Pipeline starten
    app.post('/api/agents/pipeline/run', async (req, res) => {
        try {
            const { project } = req.body;

            if (!project) {
                return res.status(400).json({ error: 'project ist erforderlich' });
            }

            const { runNetworkAgentPipeline } = await import('../llm/agents/10_master_agent.js');
            const result = await runNetworkAgentPipeline(project);

            res.json({ success: true, result });
        } catch (error) {
            console.error('Fehler beim Starten der Pipeline:', error);
            res.status(500).json({
                success: false,
                error: 'Pipeline-Start fehlgeschlagen: ' + error.message
            });
        }
    });

    // Pipeline-Status abrufen
    app.get('/api/agents/pipeline/status', async (req, res) => {
        try {
            const { masterAgent } = await import('../llm/agents/10_master_agent.js');
            const pipelineStatus = masterAgent.getPipelineStatus();
            const availableWorkers = masterAgent.getAvailableWorkers();

            res.json({
                success: true,
                pipelineStatus,
                availableWorkers
            });
        } catch (error) {
            console.error('Fehler beim Abrufen des Pipeline-Status:', error);
            res.status(500).json({
                success: false,
                error: 'Fehler beim Abrufen des Pipeline-Status: ' + error.message
            });
        }
    });

    // Auto-Tuning für Modell-Optimierung
    app.post('/api/agents/auto-tune', async (req, res) => {
        try {
            const { project, maxIterations = 2 } = req.body;

            if (!project) {
                return res.status(400).json({ error: 'project ist erforderlich' });
            }

            const { autoTuneModelWithLLM } = await import('../llm/agents/tuning.js');
            const proposal = await autoTuneModelWithLLM(project, Math.max(1, Math.min(maxIterations, 5)));

            res.json({ success: true, proposal });
        } catch (error) {
            console.error('Fehler beim Auto-Tuning:', error);
            res.status(500).json({ error: 'Auto-Tuning fehlgeschlagen: ' + error.message });
        }
    });

    // Worker-Agent testen
    app.post('/api/agents/worker/test/:agentKey', async (req, res) => {
        try {
            const { agentKey } = req.params;

            const { masterAgent } = await import('../llm/agents/10_master_agent.js');
            const testResult = await masterAgent.testWorkerAgent(agentKey);

            res.json({
                success: testResult.success,
                agentKey,
                testResult,
                timestamp: new Date().toISOString()
            });
        } catch (error) {
            console.error('Fehler beim Testen des Worker-Agents:', error);
            res.status(500).json({
                success: false,
                error: 'Fehler beim Testen des Worker-Agents: ' + error.message
            });
        }
    });
}

