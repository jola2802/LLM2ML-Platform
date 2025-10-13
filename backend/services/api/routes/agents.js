import { logRESTAPIRequest } from '../../monitoring/log.js';
import { 
  ALL_AGENTS,
  getAgentConfig, 
  getAgentModel, 
  getAgentStats,
  isValidAgent,
  getWorkerAgents,
  getAgentsByCategory
} from '../../llm/agents/config_agent_network.js';
import { masterAgent } from '../../llm/agents/10_master_agent.js';

/**
 * API-Routen für Agent-Konfiguration
 */
export function setupAgentRoutes(app) {
  
  // Alle verfügbaren Agents abrufen
  app.get('/api/agents', async (req, res) => {
    try {
      logRESTAPIRequest('get-agents', {});
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
      logRESTAPIRequest('get-agent-config', { agentKey });
      
      if (!isValidAgent(agentKey)) {
        return res.status(404).json({
          success: false,
          error: `Agent '${agentKey}' nicht gefunden`
        });
      }
      
      const config = getAgentConfig(agentKey);
      
      res.json({
        success: true,
        agentKey,
        config
      });
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
      
      if (!isValidAgent(agentKey)) {
        return res.status(404).json({
          success: false,
          error: `Agent '${agentKey}' nicht gefunden`
        });
      }
      
      // In der neuen Struktur sind Modell-Änderungen zur Laufzeit nicht möglich
      // Die Konfiguration ist statisch in network_agent_config.js
      res.json({
        success: false,
        message: 'Modell-Änderungen zur Laufzeit sind nicht möglich. Ändern Sie die Konfiguration in network_agent_config.js',
        agentKey,
        currentModel: getAgentModel(agentKey),
        config: getAgentConfig(agentKey)
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
      const stats = getAgentStats();
      
      res.json({
        success: true,
        stats
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
      
      const agentKeys = Object.keys(ALL_AGENTS);
      const workerKeys = getWorkerAgents().map(agent => agent.key);
      
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
      
      const agents = Object.entries(ALL_AGENTS).map(([key, config]) => ({
        key,
        ...config
      }));
      const stats = getAgentStats();
      
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
      
      if (!isValidAgent(agentKey)) {
        return res.status(404).json({
          success: false,
          error: `Agent '${agentKey}' nicht gefunden`
        });
      }
      
      const config = getAgentConfig(agentKey);
      const model = getAgentModel(agentKey);
      
      // Teste das zugewiesene Modell mit einem einfachen Prompt
      const { callLLMAPI } = await import('../../llm/api/llm.js');
      
      const testPrompt = `Du bist der ${config.name}. Antworte nur mit "OK" wenn du diese Nachricht erhältst.`;
      
      try {
        const response = await callLLMAPI(testPrompt, null, model, 1);
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
      
      const pipelineStatus = masterAgent.getPipelineStatus();
      const availableWorkers = masterAgent.getAvailableWorkers();
      
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
      
      if (!isValidAgent(agentKey)) {
        return res.status(404).json({
          success: false,
          error: `Worker-Agent '${agentKey}' nicht gefunden`
        });
      }
      
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

  console.log('✅ Agent-Management und Activity-Tracking API-Routen eingerichtet');
}
