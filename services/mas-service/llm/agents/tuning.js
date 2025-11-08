import { callLLMAPI } from '../api/llm.js';
import { getAgentModel, logAgentCall, getAgentConfig, WORKER_AGENTS } from './config_agent_network.js';
import { AUTO_TUNING_PROMPT, formatPrompt } from './prompts.js';

// LLM-getriebenes Auto-Tuning (Algorithmus + Hyperparameter)
export async function autoTuneModelWithLLM(project, maxIterations = 2) {
  const agentModel = getAgentModel(WORKER_AGENTS.HYPERPARAMETER_OPTIMIZER.key);
  const agentConfig = getAgentConfig(WORKER_AGENTS.HYPERPARAMETER_OPTIMIZER.key);
  
  logAgentCall(WORKER_AGENTS.HYPERPARAMETER_OPTIMIZER.key, agentModel, 'Auto-Tuning für Modell-Optimierung');
  console.log(`🤖 ${agentConfig.name} startet Auto-Tuning mit Modell: ${agentModel}`);

  let best = {
    algorithm: project.algorithm,
    hyperparameters: project.hyperparameters || {},
    expectedGain: 0,
    rationale: 'Ausgangskonfiguration'
  };

  for (let i = 0; i < maxIterations; i++) {
    console.log(`📊 Auto-Tuning Iteration ${i + 1}/${maxIterations}`);
    
    const prompt = formatPrompt(AUTO_TUNING_PROMPT, {
      modelType: project.modelType,
      algorithm: project.algorithm,
      hyperparameters: JSON.stringify(project.hyperparameters),
      features: Array.isArray(project.features) ? project.features.join(', ') : '',
      targetVariable: project.targetVariable,
      performanceMetrics: JSON.stringify(project.performanceMetrics || {}),
      dataSourceName: project.dataSourceName
    });

    try {
      const response = await callLLMAPI(prompt, null, agentModel, agentConfig.retries || 2);
      let jsonText = response?.result || String(response || '');
      jsonText = jsonText.replace(/```json/g, '').replace(/```/g, '').trim();
      const match = jsonText.match(/\{[\s\S]*\}/);
      
      if (!match) {
        console.warn(`⚠️ Keine gültige JSON-Antwort in Iteration ${i + 1}`);
        continue;
      }
      
      const proposal = JSON.parse(match[0]);
      if (typeof proposal.expectedGain === 'number' && proposal.expectedGain > (best.expectedGain || 0)) {
        console.log(`✅ Bessere Konfiguration gefunden: Expected Gain ${proposal.expectedGain}`);
        best = {
          algorithm: proposal.algorithm || best.algorithm,
          hyperparameters: proposal.hyperparameters || best.hyperparameters,
          expectedGain: proposal.expectedGain,
          rationale: proposal.rationale || best.rationale
        };
      }
    } catch (error) {
      console.warn(`⚠️ Fehler in Auto-Tuning Iteration ${i + 1}:`, error.message);
    }
  }
  
  console.log(`✅ ${agentConfig.name} Auto-Tuning abgeschlossen. Beste Konfiguration: Expected Gain ${best.expectedGain}`);
  return best;
}


