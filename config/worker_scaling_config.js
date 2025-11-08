/**
 * Worker Scaling Konfiguration
 * 
 * Diese Datei enthält alle Konfigurationsparameter für die dynamische 
 * Skalierung der Worker-Pools (Python Execution und LLM).
 */

export const PYTHON_WORKER_CONFIG = {
    // Minimum und Maximum Worker
    minWorkers: 1,           // Minimum 1 Worker immer aktiv (verhindert ständiges Down-Scaling)
    maxWorkers: 5,           // Maximum 5 Worker bei hoher Last

    // Skalierungs-Schwellenwerte
    scaleUpThreshold: 0.8,   // Skaliere hoch wenn 80% der Worker beschäftigt sind
    scaleDownThreshold: 0.3, // Skaliere runter wenn weniger als 30% beschäftigt sind

    // Timing-Parameter
    scaleCheckInterval: 10000,  // Prüfe alle 10 Sekunden (Python ist langsamer)
    idleTimeout: 20000,         // Worker werden nach 20s Inaktivität entfernt

    // Performance-Kriterien
    queueLengthThreshold: 2,    // Skaliere hoch wenn mehr als 2 Jobs in der Warteschlange
    avgExecutionTimeThreshold: 5000, // Berücksichtige durchschnittliche Ausführungszeit
};

export const LLM_WORKER_CONFIG = {
    // Minimum und Maximum Worker
    minWorkers: 1,           // Minimum 1 Worker immer aktiv
    maxWorkers: 3,           // Maximum 3 Worker (LLM ist ressourcenintensiv)

    // Skalierungs-Schwellenwerte
    scaleUpThreshold: 0.8,   // Skaliere hoch wenn 80% der Worker beschäftigt sind
    scaleDownThreshold: 0.3, // Skaliere runter wenn weniger als 30% beschäftigt sind

    // Timing-Parameter
    scaleCheckInterval: 8000,   // Prüfe alle 8 Sekunden (LLM ist schneller als Python)
    idleTimeout: 20000,         // Worker werden nach 20s Inaktivität entfernt

    // LLM-spezifische Parameter
    maxConcurrency: 3,          // Maximal 3 parallele LLM-Anfragen
    queueLengthThreshold: 1,    // Skaliere hoch wenn mehr als 1 Job in der Warteschlange
};

// Globale Skalierungs-Einstellungen
export const GLOBAL_SCALING_CONFIG = {
    // System-Ressourcen berücksichtigen
    enableResourceBasedScaling: true,  // Aktiviere ressourcenbasierte Skalierung
    maxCpuUsage: 85,                   // Skaliere nicht wenn CPU > 85%
    maxMemoryUsage: 80,                // Skaliere nicht wenn RAM > 80%

    // Logging und Monitoring
    enableScalingLogs: true,           // Aktiviere Skalierungs-Logs
    logLevel: 'info',                  // Log-Level: debug, info, warn, error

    // Sicherheits-Parameter
    maxScaleEventsPerMinute: 10,       // Maximal 10 Skalierungs-Events pro Minute
    cooldownPeriod: 5000,              // 5s Cooldown zwischen Skalierungs-Events
};

/**
 * Hilfsfunktion zum Überschreiben der Konfiguration via Umgebungsvariablen
 */
export function getWorkerConfig() {
    const pythonConfig = {
        ...PYTHON_WORKER_CONFIG,
        minWorkers: parseInt(process.env.PYTHON_MIN_WORKERS) || PYTHON_WORKER_CONFIG.minWorkers,
        maxWorkers: parseInt(process.env.PYTHON_MAX_WORKERS) || PYTHON_WORKER_CONFIG.maxWorkers,
        scaleCheckInterval: parseInt(process.env.PYTHON_SCALE_INTERVAL) || PYTHON_WORKER_CONFIG.scaleCheckInterval,
    };

    const llmConfig = {
        ...LLM_WORKER_CONFIG,
        minWorkers: parseInt(process.env.LLM_MIN_WORKERS) || LLM_WORKER_CONFIG.minWorkers,
        maxWorkers: parseInt(process.env.LLM_MAX_WORKERS) || LLM_WORKER_CONFIG.maxWorkers,
        scaleCheckInterval: parseInt(process.env.LLM_SCALE_INTERVAL) || LLM_WORKER_CONFIG.scaleCheckInterval,
    };

    const globalConfig = {
        ...GLOBAL_SCALING_CONFIG,
        enableResourceBasedScaling: process.env.ENABLE_RESOURCE_SCALING !== 'false',
        enableScalingLogs: process.env.ENABLE_SCALING_LOGS !== 'false',
    };

    return {
        python: pythonConfig,
        llm: llmConfig,
        global: globalConfig
    };
}

export default {
    PYTHON_WORKER_CONFIG,
    LLM_WORKER_CONFIG,
    GLOBAL_SCALING_CONFIG,
    getWorkerConfig
};

