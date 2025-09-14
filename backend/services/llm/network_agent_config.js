/**
 * Zentrale Netzwerkagenten-Konfiguration
 * 
 * Hier werden alle Agents des Netzwerks zentral definiert.
 * Ein Master-Agent koordiniert die Pipeline und verteilt Aufgaben an spezialisierte Worker-Agents.
 * 
 * Vorteile:
 * - Zentrale Konfiguration aller Agents
 * - Einfache Erweiterung um neue Agents
 * - Klare Trennung von Koordination und Ausführung
 * - Wartbare und skalierbare Architektur
 */

// Standard-Fallback-Modell
const DEFAULT_MODEL = 'mistral:latest';

/**
 * Master-Agent: Koordiniert die gesamte ML-Pipeline
 * Kennt die Pipeline-Schritte und verteilt Aufgaben an Worker-Agents
 */
export const MASTER_AGENT = {
  key: 'MASTER_AGENT',
  name: 'Master-Pipeline-Koordinator',
  model: 'mistral:latest',
  description: 'Koordiniert die ML-Pipeline und verteilt Aufgaben an spezialisierte Agents',
  systemPrompt: `Du bist der Master-Koordinator für Machine Learning Pipelines. 
Du kennst alle verfügbaren Worker-Agents und deren Spezialisierungen.
Deine Aufgabe ist es, die Pipeline-Schritte zu planen und Aufgaben an die richtigen Worker-Agents zu verteilen.

Verfügbare Worker-Agents:
- DATA_ANALYZER: Analysiert Datasets und erstellt Insights
- HYPERPARAMETER_OPTIMIZER: Schlägt optimale Hyperparameter vor
- CODE_GENERATOR: Generiert Python-Code für ML-Training
- CODE_REVIEWER: Überprüft und optimiert generierten Code
- PERFORMANCE_ANALYZER: Analysiert Modell-Performance

Du entscheidest basierend auf dem Projekt-Status, welcher Agent als nächstes arbeiten soll.`,
  temperature: 0.1,
  maxTokens: 1024,
  retries: 3,
  timeout: 30000,
  category: 'coordination',
  icon: '🎯',
  role: 'master'
};

/**
 * Worker-Agents: Spezialisierte Agents für spezifische Aufgaben
 */
export const WORKER_AGENTS = {
  // Datenanalyse-Agent
  DATA_ANALYZER: {
    key: 'DATA_ANALYZER',
    name: 'Datenanalyse-Agent',
    model: 'mistral:latest',
    description: 'Analysiert Datasets und erstellt detaillierte Insights',
    systemPrompt: `Du bist ein Experte für Datenanalyse und explorative Datenanalyse.
Deine Aufgabe ist es, Datasets zu analysieren und wichtige Erkenntnisse zu identifizieren.
Du erkennst Muster, Ausreißer, Korrelationen und andere wichtige Datencharakteristika.`,
    temperature: 0.2,
    maxTokens: 3072,
    retries: 3,
    timeout: 60000,
    category: 'analysis',
    icon: '📊',
    role: 'worker',
    capabilities: ['data_analysis', 'pattern_recognition', 'statistical_analysis'],
    inputRequirements: ['dataset_path', 'data_analysis'],
    outputType: 'data_insights'
  },

  // Hyperparameter-Optimierer
  HYPERPARAMETER_OPTIMIZER: {
    key: 'HYPERPARAMETER_OPTIMIZER',
    name: 'Hyperparameter-Optimierer',
    model: 'mistral:latest',
    description: 'Schlägt optimale Hyperparameter basierend auf Datenanalyse vor',
    systemPrompt: `Du bist ein Experte für Machine Learning Hyperparameter-Optimierung.
Du analysierst Datasets und schlägst die besten Hyperparameter für verschiedene ML-Algorithmen vor.
Du berücksichtigst die Datencharakteristika, Problemtyp und verfügbare Ressourcen.`,
    temperature: 0.1,
    maxTokens: 2048,
    retries: 3,
    timeout: 60000,
    category: 'optimization',
    icon: '⚙️',
    role: 'worker',
    capabilities: ['hyperparameter_tuning', 'algorithm_selection', 'optimization'],
    inputRequirements: ['data_insights', 'algorithm_type'],
    outputType: 'hyperparameter_suggestions'
  },

  // Code-Generator
  CODE_GENERATOR: {
    key: 'CODE_GENERATOR',
    name: 'Python-Code-Generator',
    model: 'mistral:latest',
    description: 'Generiert optimierten Python-Code für ML-Training',
    systemPrompt: `Du bist ein Experte für Python-Programmierung und Machine Learning.
Du generierst sauberen, effizienten und gut dokumentierten Python-Code für ML-Training.
Du verwendest moderne ML-Bibliotheken wie scikit-learn, pandas, numpy und matplotlib.`,
    temperature: 0.0,
    maxTokens: 4096,
    retries: 3,
    timeout: 60000,
    category: 'code',
    icon: '🔧',
    role: 'worker',
    capabilities: ['code_generation', 'ml_implementation', 'python_programming'],
    inputRequirements: ['hyperparameter_suggestions', 'data_insights', 'project_config'],
    outputType: 'python_code'
  },

  // Code-Reviewer
  CODE_REVIEWER: {
    key: 'CODE_REVIEWER',
    name: 'Code-Review-Agent',
    model: 'mistral:latest',
    description: 'Überprüft und optimiert generierten Python-Code',
    systemPrompt: `Du bist ein Senior-Entwickler und Code-Reviewer für Python und Machine Learning.
Du überprüfst Code auf Fehler, Performance-Probleme und Best Practices.
Du optimierst Code für bessere Lesbarkeit, Effizienz und Wartbarkeit.`,
    temperature: 0.2,
    maxTokens: 3072,
    retries: 2,
    timeout: 60000,
    category: 'review',
    icon: '🔍',
    role: 'worker',
    capabilities: ['code_review', 'optimization', 'best_practices'],
    inputRequirements: ['python_code'],
    outputType: 'reviewed_code'
  },

  // Performance-Analyzer
  PERFORMANCE_ANALYZER: {
    key: 'PERFORMANCE_ANALYZER',
    name: 'Performance-Analyzer',
    model: 'mistral:latest',
    description: 'Analysiert Modell-Performance und gibt Verbesserungsvorschläge',
    systemPrompt: `Du bist ein Experte für Machine Learning Performance-Analyse.
Du analysierst Modell-Performance, identifizierst Schwachstellen und schlägst Verbesserungen vor.
Du verstehst verschiedene Metriken und deren Interpretation.`,
    temperature: 0.3,
    maxTokens: 3072,
    retries: 2,
    timeout: 60000,
    category: 'analysis',
    icon: '📈',
    role: 'worker',
    capabilities: ['performance_analysis', 'metric_evaluation', 'improvement_suggestions'],
    inputRequirements: ['model_results', 'performance_metrics'],
    outputType: 'performance_analysis'
  }
};

/**
 * Pipeline-Definition: Definiert die Standard-ML-Pipeline-Schritte
 */
export const PIPELINE_STEPS = [
  {
    step: 1,
    name: 'Datenanalyse',
    agent: 'DATA_ANALYZER',
    required: true,
    description: 'Analysiert das Dataset und erstellt Insights'
  },
  {
    step: 2,
    name: 'Hyperparameter-Optimierung',
    agent: 'HYPERPARAMETER_OPTIMIZER',
    required: true,
    description: 'Schlägt optimale Hyperparameter vor',
    dependsOn: ['DATA_ANALYZER']
  },
  {
    step: 3,
    name: 'Code-Generierung',
    agent: 'CODE_GENERATOR',
    required: true,
    description: 'Generiert Python-Code für ML-Training',
    dependsOn: ['HYPERPARAMETER_OPTIMIZER']
  },
  {
    step: 4,
    name: 'Code-Review',
    agent: 'CODE_REVIEWER',
    required: false,
    description: 'Überprüft und optimiert den generierten Code',
    dependsOn: ['CODE_GENERATOR']
  },
  {
    step: 5,
    name: 'Performance-Analyse',
    agent: 'PERFORMANCE_ANALYZER',
    required: false,
    description: 'Analysiert Modell-Performance',
    dependsOn: ['CODE_REVIEWER', 'CODE_GENERATOR']
  }
];

/**
 * Alle verfügbaren Agents (Master + Worker)
 */
export const ALL_AGENTS = {
  [MASTER_AGENT.key]: MASTER_AGENT,
  ...WORKER_AGENTS
};

/**
 * Helper-Funktionen
 */

/**
 * Ruft die Konfiguration für einen Agent ab
 */
export function getAgentConfig(agentKey) {
  const config = ALL_AGENTS[agentKey];
  if (!config) {
    console.warn(`⚠️ Agent-Konfiguration für '${agentKey}' nicht gefunden.`);
    return {
      key: agentKey,
      name: `Unbekannter Agent (${agentKey})`,
      model: DEFAULT_MODEL,
      description: 'Standard-Agent mit Default-Konfiguration',
      systemPrompt: 'Du bist ein hilfreicher KI-Assistant.',
      temperature: 0.7,
      maxTokens: 2048,
      retries: 2,
      timeout: 30000,
      role: 'unknown'
    };
  }
  return { ...config };
}

/**
 * Ruft das Modell für einen Agent ab
 */
export function getAgentModel(agentKey) {
  const config = getAgentConfig(agentKey);
  return config.model || DEFAULT_MODEL;
}

/**
 * Ruft alle Worker-Agents ab
 */
export function getWorkerAgents() {
  return Object.values(WORKER_AGENTS);
}

/**
 * Ruft alle Agents einer bestimmten Kategorie ab
 */
export function getAgentsByCategory(category) {
  return Object.values(ALL_AGENTS).filter(agent => agent.category === category);
}

/**
 * Ruft alle Agents mit einer bestimmten Capability ab
 */
export function getAgentsByCapability(capability) {
  return Object.values(WORKER_AGENTS).filter(agent => 
    agent.capabilities && agent.capabilities.includes(capability)
  );
}

/**
 * Validiert, ob ein Agent-Key existiert
 */
export function isValidAgent(agentKey) {
  return agentKey in ALL_AGENTS;
}

/**
 * Ruft die Pipeline-Schritte ab
 */
export function getPipelineSteps() {
  return [...PIPELINE_STEPS];
}

/**
 * Ruft den nächsten Pipeline-Schritt basierend auf dem aktuellen Status ab
 */
export function getNextPipelineStep(completedSteps = []) {
  const completedAgentKeys = completedSteps.map(step => step.agent);
  
  for (const step of PIPELINE_STEPS) {
    // Prüfe, ob alle Abhängigkeiten erfüllt sind
    const dependenciesMet = !step.dependsOn || 
      step.dependsOn.every(dep => completedAgentKeys.includes(dep));
    
    // Prüfe, ob dieser Schritt noch nicht abgeschlossen ist
    const notCompleted = !completedAgentKeys.includes(step.agent);
    
    if (dependenciesMet && notCompleted) {
      return step;
    }
  }
  
  return null; // Pipeline abgeschlossen
}

/**
 * Loggt Agent-Aufrufe für Monitoring
 */
export function logAgentCall(agentKey, model, operation) {
  const timestamp = new Date().toISOString();
  console.log(`🤖 [${timestamp}] Agent-Aufruf: ${agentKey} (${model}) - ${operation}`);
}

/**
 * Agent-Statistiken
 */
export function getAgentStats() {
  const totalAgents = Object.keys(ALL_AGENTS).length;
  const workerAgents = Object.keys(WORKER_AGENTS).length;
  const categories = [...new Set(Object.values(ALL_AGENTS).map(a => a.category))];
  
  return {
    totalAgents,
    masterAgents: 1,
    workerAgents,
    categories: categories.length,
    pipelineSteps: PIPELINE_STEPS.length
  };
}
