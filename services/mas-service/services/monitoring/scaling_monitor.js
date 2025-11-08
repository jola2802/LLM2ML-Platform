/**
 * Scaling Monitor - Überwacht die dynamische Worker-Skalierung
 * Vereinfachte Version für MAS-Service
 */

import { EventEmitter } from 'events';

class ScalingMonitor extends EventEmitter {
    constructor() {
        super();
        this.metrics = {
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
    }

    // Metriken aktualisieren
    updateMetrics(type, data) {
        if (this.metrics[type]) {
            Object.assign(this.metrics[type], data);
        }
    }

    // Skalierungs-Event aufzeichnen
    recordScalingEvent(type, direction, data) {
        if (!this.metrics[type]) return;

        const timestamp = new Date().toISOString();
        const event = {
            timestamp,
            direction,
            ...data
        };

        this.metrics[type].scalingEvents.push(event);

        if (direction === 'up') {
            this.metrics[type].totalScaleUps++;
            this.metrics[type].lastScaleUp = timestamp;
        } else {
            this.metrics[type].totalScaleDowns++;
            this.metrics[type].lastScaleDown = timestamp;
        }

        // Event-Historie begrenzen
        if (this.metrics[type].scalingEvents.length > 50) {
            this.metrics[type].scalingEvents = this.metrics[type].scalingEvents.slice(-50);
        }
    }

    // Event-Listener für Skalierungs-Events
    setupEventListeners() {
        this.on('llm:scaleUp', (data) => this.recordScalingEvent('llm', 'up', data));
        this.on('llm:scaleDown', (data) => this.recordScalingEvent('llm', 'down', data));
        this.on('llm:metrics', (data) => this.updateMetrics('llm', data));
    }

    // Aktuelle Skalierungs-Metriken abrufen
    getScalingMetrics() {
        return {
            timestamp: new Date().toISOString(),
            llm: {
                ...this.metrics.llm
            }
        };
    }
}

// Singleton-Instanz
export const scalingMonitor = new ScalingMonitor();
scalingMonitor.setupEventListeners();

export default scalingMonitor;

