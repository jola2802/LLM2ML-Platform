import { logRESTAPIRequest } from '../../monitoring/log.js';
import { masClient } from '../../clients/mas_client.js';
// Alle Agent-Funktionen werden jetzt über MAS-Service verwendet (masClient)

/**
 * API-Routen für Agent-Konfiguration
 */
export function setupAgentRoutes(app) {

  // Alle verfügbaren Agents abrufen
  app.get('/api/agents', async (req, res) => {
    try {
      logRESTAPIRequest('get-agents', {});
      const result = await masClient.getAllAgents();
      res.json(result);
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
      logRESTAPIRequest('get-agent-config', { agentKey });
      const result = await masClient.getAgentConfig(agentKey);
      res.json(result);
    } catch (error) {
      console.error('Fehler beim Abrufen der Agent-Konfiguration:', error);
      res.status(500).json({
        success: false,
        error: 'Fehler beim Abrufen der Agent-Konfiguration: ' + error.message
      });
    }
  });

  // Agent-Modell aktualisieren (nur für Worker-Agents)
  app.put('/api/agents/:agentKey/model', async (req, res) => {
    try {
      const { agentKey } = req.params;
      const { model } = req.body;

      logRESTAPIRequest('update-agent-model', { agentKey, model });

      if (!model) {
        return res.status(400).json({
          success: false,
          error: 'Modell-Parameter ist erforderlich'
        });
      }

      // Modell-Änderungen werden über MAS-Service gehandhabt
      const agentInfo = await masClient.getAgentConfig(agentKey);
      res.json({
        success: false,
        message: 'Modell-Änderungen zur Laufzeit sind nicht möglich. Ändern Sie die Konfiguration im MAS-Service',
        agentKey,
        currentModel: agentInfo.agent?.model,
        config: agentInfo.agent
      });
    } catch (error) {
      console.error('Fehler beim Aktualisieren des Agent-Modells:', error);
      res.status(500).json({
        success: false,
        error: 'Fehler beim Aktualisieren des Agent-Modells: ' + error.message
      });
    }
  });

  // Agent-Statistiken abrufen
  app.get('/api/agents/stats', async (req, res) => {
    try {
      logRESTAPIRequest('get-agent-stats', {});
      const result = await masClient.getAllAgents();
      res.json({
        success: true,
        stats: result.stats
      });
    } catch (error) {
      console.error('Fehler beim Abrufen der Agent-Statistiken:', error);
      res.status(500).json({
        success: false,
        error: 'Fehler beim Abrufen der Agent-Statistiken: ' + error.message
      });
    }
  });

  // Verfügbare Agent-Keys abrufen
  app.get('/api/agents/keys', async (req, res) => {
    try {
      logRESTAPIRequest('get-agent-keys', {});
      const result = await masClient.getAllAgents();
      const agentKeys = result.agents?.map(agent => agent.key) || [];
      const workerKeys = result.agents?.filter(agent => agent.role === 'worker').map(agent => agent.key) || [];

      res.json({
        success: true,
        agentKeys: agentKeys,
        workerKeys: workerKeys,
        masterKey: 'MASTER_AGENT',
        keysList: agentKeys
      });
    } catch (error) {
      console.error('Fehler beim Abrufen der Agent-Keys:', error);
      res.status(500).json({
        success: false,
        error: 'Fehler beim Abrufen der Agent-Keys: ' + error.message
      });
    }
  });

  // Frontend-optimierte Agent-Konfiguration abrufen
  app.get('/api/agents/frontend-config', async (req, res) => {
    try {
      logRESTAPIRequest('get-frontend-agent-config', {});
      const result = await masClient.getAllAgents();
      const agents = result.agents || [];
      const stats = result.stats || {};

      // Optimierte Konfiguration für das Frontend
      const frontendConfig = {
        agents: agents.map(agent => ({
          key: agent.key,
          name: agent.name,
          description: agent.description,
          model: agent.model,
          category: agent.category,
          icon: agent.icon,
          temperature: agent.temperature,
          maxTokens: agent.maxTokens,
          // Sensitive Daten wie promptFunction werden nicht übertragen
        })),
        categories: [...new Set(agents.map(a => a.category))].map(cat => ({
          name: cat,
          agents: agents.filter(a => a.category === cat).map(a => a.key)
        })),
        stats,
        totalAgents: agents.length
      };

      res.json({
        success: true,
        config: frontendConfig,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('Fehler beim Abrufen der Frontend-Agent-Konfiguration:', error);
      res.status(500).json({
        success: false,
        error: 'Fehler beim Abrufen der Frontend-Agent-Konfiguration: ' + error.message
      });
    }
  });

  // Agent-Konfiguration testen (Modell-Verbindungstest)
  app.post('/api/agents/:agentKey/test', async (req, res) => {
    try {
      const { agentKey } = req.params;

      logRESTAPIRequest('test-agent-config', { agentKey });

      const agentInfo = await masClient.getAgentConfig(agentKey);
      if (!agentInfo.success || !agentInfo.agent) {
        return res.status(404).json({
          success: false,
          error: `Agent '${agentKey}' nicht gefunden`
        });
      }

      const config = agentInfo.agent;
      const model = config.model;

      // Teste das zugewiesene Modell mit einem einfachen Prompt
      const { masClient } = await import('../../clients/mas_client.js');

      const testPrompt = `Du bist der ${config.name}. Antworte nur mit "OK" wenn du diese Nachricht erhältst.`;

      try {
        const response = await masClient.callLLM(testPrompt, null, model, 1);
        const result = typeof response === 'string' ? response : response?.result || '';

        const isWorking = result.toLowerCase().includes('ok');

        res.json({
          success: true,
          agentKey,
          model,
          tested: true,
          working: isWorking,
          response: result,
          config
        });
      } catch (llmError) {
        res.json({
          success: false,
          agentKey,
          model,
          tested: true,
          working: false,
          error: llmError.message,
          config
        });
      }
    } catch (error) {
      console.error('Fehler beim Testen der Agent-Konfiguration:', error);
      res.status(500).json({
        success: false,
        error: 'Fehler beim Testen der Agent-Konfiguration: ' + error.message
      });
    }
  });

  // ===== NETZWERK-AGENT-PIPELINE-TRACKING =====

  // Pipeline-Status abrufen
  app.get('/api/agents/pipeline/status', async (req, res) => {
    try {
      logRESTAPIRequest('get-pipeline-status', {});

      const { masClient } = await import('../../clients/mas_client.js');
      const pipelineInfo = await masClient.getPipelineStatus();
      const pipelineStatus = pipelineInfo.pipelineStatus;
      const availableWorkers = pipelineInfo.availableWorkers;

      res.json({
        success: true,
        pipelineStatus,
        availableWorkers,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('Fehler beim Abrufen des Pipeline-Status:', error);
      res.status(500).json({
        success: false,
        error: 'Fehler beim Abrufen des Pipeline-Status: ' + error.message
      });
    }
  });

  // Worker-Agent testen
  app.post('/api/agents/workers/:agentKey/test', async (req, res) => {
    try {
      const { agentKey } = req.params;
      logRESTAPIRequest('test-worker-agent', { agentKey });

      const testResult = await masClient.testWorkerAgent(agentKey);

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

  console.log('✅ Agent-Management und Activity-Tracking API-Routen eingerichtet');
}
