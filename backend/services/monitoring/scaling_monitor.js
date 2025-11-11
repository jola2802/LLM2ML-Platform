/**
 * Scaling Monitor - Überwacht die dynamische Worker-Skalierung
 */

import { EventEmitter } from 'events';
// Import-Pfad - im Container ist config unter /app/config gemountet
// Lokal: config liegt im Root-Verzeichnis, von backend/services/monitoring/ aus: ../../../config/
// Im Container: config liegt unter /app/config, von /app/services/monitoring/ aus: ../../config/
import { getWorkerConfig } from '../../../config/worker_scaling_config.js';

class ScalingMonitor extends EventEmitter {
  constructor() {
    super();
    this.config = getWorkerConfig();
    this.metrics = {
      python: {
        activeWorkers: 0,
        busyWorkers: 0,
        queueLength: 0,
        scalingEvents: [],
        lastScaleUp: null,
        lastScaleDown: null,
        totalScaleUps: 0,
        totalScaleDowns: 0
      },
      llm: {
        activeWorkers: 0,
        busyWorkers: 0,
        queueLength: 0,
        scalingEvents: [],
        lastScaleUp: null,
        lastScaleDown: null,
        totalScaleUps: 0,
        totalScaleDowns: 0
      }
    };

    // Event-Listener für Skalierungs-Events
    this.setupEventListeners();

    // Regelmäßige Metriken-Aufräumung
    this.startCleanupTimer();
  }

  // Event-Listener für Worker-Pools einrichten
  setupEventListeners() {
    // Diese werden von den Worker-Pools aufgerufen
    this.on('python:scaleUp', (data) => this.recordScalingEvent('python', 'up', data));
    this.on('python:scaleDown', (data) => this.recordScalingEvent('python', 'down', data));
    this.on('llm:scaleUp', (data) => this.recordScalingEvent('llm', 'up', data));
    this.on('llm:scaleDown', (data) => this.recordScalingEvent('llm', 'down', data));

    this.on('python:metrics', (data) => this.updateMetrics('python', data));
    this.on('llm:metrics', (data) => this.updateMetrics('llm', data));
  }

  // Skalierungs-Event aufzeichnen
  recordScalingEvent(type, direction, data) {
    const timestamp = new Date().toISOString();
    const event = {
      timestamp,
      direction,
      ...data
    };

    // Event zur Historie hinzufügen
    this.metrics[type].scalingEvents.push(event);

    // Statistiken aktualisieren
    if (direction === 'up') {
      this.metrics[type].totalScaleUps++;
      this.metrics[type].lastScaleUp = timestamp;
    } else {
      this.metrics[type].totalScaleDowns++;
      this.metrics[type].lastScaleDown = timestamp;
    }

    // Logging (wenn aktiviert)
    if (this.config.global.enableScalingLogs) {
      console.log(`[SCALING] ${type.toUpperCase()} ${direction}: ${data.reason || 'Automatisch'} (Worker: ${data.workerCount})`);
    }

    // Event-Historie begrenzen (letzte 50 Events)
    if (this.metrics[type].scalingEvents.length > 50) {
      this.metrics[type].scalingEvents = this.metrics[type].scalingEvents.slice(-50);
    }
  }

  // Metriken aktualisieren
  updateMetrics(type, data) {
    Object.assign(this.metrics[type], data);
  }

  // Aktuelle Skalierungs-Metriken abrufen
  getScalingMetrics() {
    return {
      timestamp: new Date().toISOString(),
      python: {
        ...this.metrics.python,
        config: this.config.python
      },
      llm: {
        ...this.metrics.llm,
        config: this.config.llm
      },
      global: this.config.global
    };
  }

  // Skalierungs-Historie für bestimmten Zeitraum abrufen
  getScalingHistory(type, hours = 24) {
    const cutoffTime = new Date(Date.now() - hours * 60 * 60 * 1000);

    return this.metrics[type].scalingEvents.filter(event =>
      new Date(event.timestamp) > cutoffTime
    );
  }

  // Skalierungs-Effizienz berechnen
  getScalingEfficiency(type) {
    const events = this.metrics[type].scalingEvents;
    if (events.length < 2) {
      return { efficiency: 1, analysis: 'Nicht genügend Daten' };
    }

    const recentEvents = events.slice(-10); // Letzte 10 Events
    let oscillations = 0;

    // Oszillationen zählen (Up -> Down -> Up innerhalb kurzer Zeit)
    for (let i = 0; i < recentEvents.length - 2; i++) {
      const current = recentEvents[i];
      const next = recentEvents[i + 1];
      const afterNext = recentEvents[i + 2];

      if (current.direction !== next.direction &&
        current.direction === afterNext.direction) {
        const timeDiff = new Date(afterNext.timestamp) - new Date(current.timestamp);
        if (timeDiff < 5 * 60 * 1000) { // Innerhalb von 5 Minuten
          oscillations++;
        }
      }
    }

    const efficiency = Math.max(0, 1 - (oscillations / recentEvents.length));
    let analysis = 'Stabile Skalierung';

    if (efficiency < 0.7) {
      analysis = 'Häufige Oszillationen - Schwellenwerte überprüfen';
    } else if (efficiency < 0.9) {
      analysis = 'Gelegentliche Oszillationen - Performance beobachten';
    }

    return { efficiency, analysis, oscillations };
  }

  // Worker-Auslastung analysieren
  analyzeUtilization(type) {
    const metrics = this.metrics[type];
    if (metrics.activeWorkers === 0) {
      return { utilization: 0, recommendation: 'Keine Worker aktiv' };
    }

    const utilization = metrics.busyWorkers / metrics.activeWorkers;
    let recommendation = 'Optimale Auslastung';

    if (utilization > 0.8 && metrics.queueLength > 0) {
      recommendation = 'Skalierung nach oben empfohlen';
    } else if (utilization < 0.3 && metrics.queueLength === 0) {
      recommendation = 'Skalierung nach unten möglich';
    } else if (utilization > 0.9) {
      recommendation = 'Hohe Auslastung - mehr Worker benötigt';
    }

    return { utilization, recommendation };
  }

  // Aufräum-Timer starten
  startCleanupTimer() {
    setInterval(() => {
      const cutoffTime = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000); // 7 Tage

      // Alte Events entfernen
      ['python', 'llm'].forEach(type => {
        this.metrics[type].scalingEvents = this.metrics[type].scalingEvents.filter(
          event => new Date(event.timestamp) > cutoffTime
        );
      });
    }, 60 * 60 * 1000); // Jede Stunde aufräumen
  }

  // Detaillierter Status-Report
  getDetailedReport() {
    const report = {
      timestamp: new Date().toISOString(),
      python: {
        metrics: this.metrics.python,
        utilization: this.analyzeUtilization('python'),
        efficiency: this.getScalingEfficiency('python'),
        recentHistory: this.getScalingHistory('python', 1) // Letzte Stunde
      },
      llm: {
        metrics: this.metrics.llm,
        utilization: this.analyzeUtilization('llm'),
        efficiency: this.getScalingEfficiency('llm'),
        recentHistory: this.getScalingHistory('llm', 1) // Letzte Stunde
      },
      summary: {
        totalScalingEvents: this.metrics.python.scalingEvents.length + this.metrics.llm.scalingEvents.length,
        systemStability: this.getSystemStability()
      }
    };

    return report;
  }

  // System-Stabilität bewerten
  getSystemStability() {
    const pythonEfficiency = this.getScalingEfficiency('python').efficiency;
    const llmEfficiency = this.getScalingEfficiency('llm').efficiency;
    const avgEfficiency = (pythonEfficiency + llmEfficiency) / 2;

    if (avgEfficiency > 0.9) return 'Excellent';
    if (avgEfficiency > 0.8) return 'Good';
    if (avgEfficiency > 0.7) return 'Fair';
    return 'Needs Attention';
  }

  // Konfiguration zur Laufzeit aktualisieren
  updateConfig(newConfig) {
    this.config = { ...this.config, ...newConfig };
    this.emit('configUpdated', this.config);
  }
}

// Singleton-Instanz
export const scalingMonitor = new ScalingMonitor();
export default scalingMonitor;
