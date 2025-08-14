import { callLLMAPI } from './llm.js';

// LLM-getriebenes Auto-Tuning (Algorithmus + Hyperparameter)
export async function autoTuneModelWithLLM(project, maxIterations = 2) {
  let best = {
    algorithm: project.algorithm,
    hyperparameters: project.hyperparameters || {},
    expectedGain: 0,
    rationale: 'Ausgangskonfiguration'
  };

  for (let i = 0; i < maxIterations; i++) {
    const prompt = `Du bist ein erfahrener ML-AutoML-Experte. Basierend auf diesem Kontext, schlage eine verbesserte Konfiguration (Algorithmus + Hyperparameter) vor, die die Performance voraussichtlich erhöht. Gib NUR JSON zurück.

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

    const response = await callLLMAPI(prompt, null, 'gemini-2.5-flash-lite', 2);
    let jsonText = response?.result || String(response || '');
    jsonText = jsonText.replace(/```json/g, '').replace(/```/g, '').trim();
    const match = jsonText.match(/\{[\s\S]*\}/);
    if (!match) continue;
    try {
      const proposal = JSON.parse(match[0]);
      if (typeof proposal.expectedGain === 'number' && proposal.expectedGain > (best.expectedGain || 0)) {
        best = {
          algorithm: proposal.algorithm || best.algorithm,
          hyperparameters: proposal.hyperparameters || best.hyperparameters,
          expectedGain: proposal.expectedGain,
          rationale: proposal.rationale || best.rationale
        };
      }
    } catch {}
  }
  return best;
}


