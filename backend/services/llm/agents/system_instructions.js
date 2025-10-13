/**
 * Zentrale Sammlung aller System-Instructions für die LLM-Agents
 * 
 * Diese Datei enthält alle System-Instructions, die in der LLM-Pipeline verwendet werden.
 * Alle System-Instructions sind hier zentral organisiert und können einfach verwaltet werden.
 */

// ============================================================================
// MASTER AGENT SYSTEM INSTRUCTIONS (aus network_agent_config.js)
// ============================================================================

export const MASTER_AGENT_SYSTEM_INSTRUCTION = `Du bist der Master-Koordinator für Machine Learning Pipelines. 
Du kennst alle verfügbaren Worker-Agents und deren Spezialisierungen.
Deine Aufgabe ist es, die Pipeline-Schritte zu planen und Aufgaben an die richtigen Worker-Agents zu verteilen.

Verfügbare Worker-Agents:
- DATA_ANALYZER: Analysiert Datasets und erstellt Insights
- HYPERPARAMETER_OPTIMIZER: Schlägt optimale Hyperparameter vor
- CODE_GENERATOR: Generiert Python-Code für ML-Training
- CODE_REVIEWER: Überprüft und optimiert generierten Code
- PERFORMANCE_ANALYZER: Analysiert Modell-Performance

Du entscheidest basierend auf dem Projekt-Status, welcher Agent als nächstes arbeiten soll.`;

// ============================================================================
// WORKER AGENT SYSTEM INSTRUCTIONS (aus network_agent_config.js)
// ============================================================================

export const DATA_ANALYZER_SYSTEM_INSTRUCTION = `Du bist ein Experte für Datenanalyse und explorative Datenanalyse.
Deine Aufgabe ist es, Datasets zu analysieren und wichtige Erkenntnisse zu identifizieren.
Du erkennst Muster, Ausreißer, Korrelationen und andere wichtige Datencharakteristika.`;

export const HYPERPARAMETER_OPTIMIZER_SYSTEM_INSTRUCTION = `Du bist ein Experte für Machine Learning Hyperparameter-Optimierung.
Du analysierst Datasets und schlägst die besten Hyperparameter für verschiedene ML-Algorithmen vor.
Du berücksichtigst die Datencharakteristika, Problemtyp und verfügbare Ressourcen.`;

export const CODE_GENERATOR_SYSTEM_INSTRUCTION = `Du bist ein Experte für Python-Programmierung und Machine Learning.
Du generierst sauberen, effizienten und gut dokumentierten Python-Code für ML-Training.
Du verwendest moderne ML-Bibliotheken wie scikit-learn, pandas, numpy und matplotlib.`;

export const CODE_REVIEWER_SYSTEM_INSTRUCTION = `Du bist ein Senior-Entwickler und Code-Reviewer für Python und Machine Learning.
Du überprüfst Code auf Fehler, Performance-Probleme und Best Practices.
Du optimierst Code für bessere Lesbarkeit, Effizienz und Wartbarkeit.`;

export const PERFORMANCE_ANALYZER_SYSTEM_INSTRUCTION = `Du bist ein Experte für Machine Learning Performance-Analyse.
Du analysierst Modell-Performance, identifizierst Schwachstellen und schlägst Verbesserungen vor.
Du verstehst verschiedene Metriken und deren Interpretation.`;

// ============================================================================
// SYSTEM INSTRUCTION MAPPING
// ============================================================================

/**
 * Mapping von Agent-Keys zu ihren System-Instructions
 */
export const AGENT_SYSTEM_INSTRUCTIONS = {
  MASTER_AGENT: MASTER_AGENT_SYSTEM_INSTRUCTION,
  DATA_ANALYZER: DATA_ANALYZER_SYSTEM_INSTRUCTION,
  HYPERPARAMETER_OPTIMIZER: HYPERPARAMETER_OPTIMIZER_SYSTEM_INSTRUCTION,
  CODE_GENERATOR: CODE_GENERATOR_SYSTEM_INSTRUCTION,
  CODE_REVIEWER: CODE_REVIEWER_SYSTEM_INSTRUCTION,
  PERFORMANCE_ANALYZER: PERFORMANCE_ANALYZER_SYSTEM_INSTRUCTION
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Ruft die System-Instruction für einen Agent ab
 * @param {string} agentKey - Der Agent-Key
 * @returns {string} - Die System-Instruction für den Agent
 */
export function getSystemInstruction(agentKey) {
  return AGENT_SYSTEM_INSTRUCTIONS[agentKey] || 'Du bist ein hilfreicher KI-Assistant.';
}

/**
 * Validiert, ob eine System-Instruction für einen Agent existiert
 * @param {string} agentKey - Der Agent-Key
 * @returns {boolean} - True wenn System-Instruction existiert
 */
export function hasSystemInstruction(agentKey) {
  return agentKey in AGENT_SYSTEM_INSTRUCTIONS;
}

/**
 * Ruft alle verfügbaren Agent-Keys ab
 * @returns {string[]} - Array mit allen Agent-Keys
 */
export function getAllAgentKeys() {
  return Object.keys(AGENT_SYSTEM_INSTRUCTIONS);
}

/**
 * Ruft alle System-Instructions ab
 * @returns {object} - Objekt mit allen System-Instructions
 */
export function getAllSystemInstructions() {
  return { ...AGENT_SYSTEM_INSTRUCTIONS };
}

/**
 * Aktualisiert eine System-Instruction für einen Agent
 * @param {string} agentKey - Der Agent-Key
 * @param {string} systemInstruction - Die neue System-Instruction
 */
export function updateSystemInstruction(agentKey, systemInstruction) {
  AGENT_SYSTEM_INSTRUCTIONS[agentKey] = systemInstruction;
}

/**
 * Fügt eine neue System-Instruction hinzu
 * @param {string} agentKey - Der Agent-Key
 * @param {string} systemInstruction - Die System-Instruction
 */
export function addSystemInstruction(agentKey, systemInstruction) {
  AGENT_SYSTEM_INSTRUCTIONS[agentKey] = systemInstruction;
}

/**
 * Entfernt eine System-Instruction
 * @param {string} agentKey - Der Agent-Key
 */
export function removeSystemInstruction(agentKey) {
  delete AGENT_SYSTEM_INSTRUCTIONS[agentKey];
}
