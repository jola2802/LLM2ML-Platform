import { callLLMAPI } from './llm.js';
import { getAgentModel, logAgentCall, AGENTS, getAgentConfig } from './agent_config.js';

// LLM-getriebenes Auto-Tuning (Algorithmus + Hyperparameter)
export async function autoTuneModelWithLLM(project, maxIterations = 2) {
  const agentModel = getAgentModel(AGENTS.AUTO_TUNER);
  const agentConfig = getAgentConfig(AGENTS.AUTO_TUNER);
  
  logAgentCall(AGENTS.AUTO_TUNER, agentModel, 'Auto-Tuning für Modell-Optimierung');
  console.log(`🤖 ${agentConfig.name} startet Auto-Tuning mit Modell: ${agentModel}`);

  let best = {
    algorithm: project.algorithm,
    hyperparameters: project.hyperparameters || {},
    expectedGain: 0,
    rationale: 'Ausgangskonfiguration'
  };

  for (let i = 0; i < maxIterations; i++) {
    console.log(`📊 Auto-Tuning Iteration ${i + 1}/${maxIterations}`);
    
    const prompt = `Du bist ein erfahrener Machine-Learning-Experte. Basierend auf diesem Kontext, schlage eine verbesserte Konfiguration (Algorithmus + Hyperparameter) vor, die die Performance voraussichtlich erhöht. Gib NUR JSON zurück.

KONTEXT:
- Modelltyp: ${project.modelType}
- Aktueller Algorithmus: ${project.algorithm}
- Aktuelle Hyperparameter: ${JSON.stringify(project.hyperparameters)}
- Features: ${Array.isArray(project.features) ? project.features.join(', ') : ''}
- Zielvariable: ${project.targetVariable}
- Letzte Performance-Metriken: ${JSON.stringify(project.performanceMetrics || {})}
- Datensatz: ${project.dataSourceName}

ANTWORTFORMAT (JSON):
{
  "algorithm": "Name des Algorithmus",
  "hyperparameters": { "param": Wert },
  "expectedGain": Zahl zwischen 0 und 1,
  "rationale": "kurze Begründung"
}`;

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


