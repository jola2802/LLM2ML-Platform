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

import { getSystemInstruction } from './system_instructions.js';

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
  systemPrompt: getSystemInstruction('MASTER_AGENT'),
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
    // model: 'qwen3:4b',
    model: 'llama3.2:latest',
    description: 'Analysiert Datasets und erstellt detaillierte Insights',
    systemPrompt: getSystemInstruction('DATA_ANALYZER'),
    temperature: 0.2,
    maxTokens: 4096,
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
    name: 'Hyperparameter Optimizer',
    // model: 'qwen3:4b',
    model: 'llama3.2:latest',
    description: 'Suggests optimal hyperparameters and features based on data analysis',
    systemPrompt: getSystemInstruction('HYPERPARAMETER_OPTIMIZER'),
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
    name: 'Python Code Generator',
    // model: 'qwen3:4b',
    model: 'llama3.2:latest',
    description: 'Generates optimized Python code for ML training',
    systemPrompt: getSystemInstruction('CODE_GENERATOR'),
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
    name: 'Code Review Agent',
    // model: 'qwen3:4b',
    model: 'llama3.2:latest',
    description: 'Reviews and optimizes generated Python code',
    systemPrompt: getSystemInstruction('CODE_REVIEWER'),
    temperature: 0.2,
    maxTokens: 4096,
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
    name: 'Performance Analyzer',
    // model: 'qwen3:4b',
    model: 'llama3.2:latest',
    description: 'Analyzes model performance and gives improvement suggestions',
    systemPrompt: getSystemInstruction('PERFORMANCE_ANALYZER'),
    temperature: 0.3,
    maxTokens: 4096,
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
    name: 'Data Analysis',
    agent: 'DATA_ANALYZER',
    required: true,
    description: 'Analysiert das Dataset und erstellt Insights'
  },
  {
    step: 2,
    name: 'Hyperparameter Optimization',
    agent: 'HYPERPARAMETER_OPTIMIZER',
    required: true,
    description: 'Suggests optimal hyperparameters',
    dependsOn: ['DATA_ANALYZER']
  },
  {
    step: 3,
    name: 'Code Generation',
    agent: 'CODE_GENERATOR',
    required: true,
    description: 'Generates Python code for ML training',
    dependsOn: ['HYPERPARAMETER_OPTIMIZER']
  },
  // {
  //   step: 4,
  //   name: 'Code-Review',
  //   agent: 'CODE_REVIEWER',
  //   required: false,
  //   description: 'Überprüft und optimiert den generierten Code',
  //   dependsOn: ['CODE_GENERATOR']
  // },
  // {
  //   step: 5,
  //   name: 'Performance-Analyse',
  //   agent: 'PERFORMANCE_ANALYZER',
  //   required: false,
  //   description: 'Analysiert Modell-Performance',
  //   // dependsOn: ['CODE_REVIEWER', 'CODE_GENERATOR']
  //   dependsOn: ['CODE_GENERATOR']
  // }
];

/**
 * Alle verfügbaren Agents (Master + Worker)
 */
export const ALL_AGENTS = {
  [MASTER_AGENT.key]: MASTER_AGENT,
  ...WORKER_AGENTS
};

export function getAgentConfig(agentKey) {
  const config = ALL_AGENTS[agentKey];
  if (!config) {
    console.warn(`⚠️ Agent configuration for '${agentKey}' not found.`);
    return {
      key: agentKey,
      name: `Unknown Agent (${agentKey})`,
      model: DEFAULT_MODEL,
      description: 'Standard agent with default configuration',
      systemPrompt: 'You are a helpful AI assistant.',
      temperature: 0.7,
      maxTokens: 2048,
      retries: 2,
      timeout: 30000,
      role: 'unknown'
    };
  }
  return { ...config };
}

export function getAgentModel(agentKey) {
  const config = getAgentConfig(agentKey);
  return config.model || DEFAULT_MODEL;
}

export function getWorkerAgents() {
  return Object.values(WORKER_AGENTS);
}

export function getAgentsByCategory(category) {
  return Object.values(ALL_AGENTS).filter(agent => agent.category === category);
}

export function getAgentsByCapability(capability) {
  return Object.values(WORKER_AGENTS).filter(agent =>
    agent.capabilities && agent.capabilities.includes(capability)
  );
}

export function isValidAgent(agentKey) {
  return agentKey in ALL_AGENTS;
}

export function getPipelineSteps() {
  return [...PIPELINE_STEPS];
}

export function getNextPipelineStep(completedSteps = []) {
  const completedAgentKeys = completedSteps.map(step => step.agent);

  for (const step of PIPELINE_STEPS) {
    const dependenciesMet = !step.dependsOn ||
      step.dependsOn.every(dep => completedAgentKeys.includes(dep));

    // Prüfe, ob dieser Schritt noch nicht abgeschlossen ist
    const notCompleted = !completedAgentKeys.includes(step.agent);

    if (dependenciesMet && notCompleted) {
      return step;
    }
  }

  return null;
}

export function logAgentCall(agentKey, model, operation) {
  const timestamp = new Date().toISOString();
  console.log(`🤖 [${timestamp}] Agent call: ${agentKey} (${model}) - ${operation}`);
}

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
