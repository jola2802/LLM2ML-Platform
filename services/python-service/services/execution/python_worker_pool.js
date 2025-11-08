import { Worker } from 'worker_threads';
import path from 'path';
import { fileURLToPath } from 'url';
import { EventEmitter } from 'events';
import { jobQueue, JOB_TYPES, JOB_STATUS } from '../monitoring/job_queue.js';
import { getWorkerConfig } from '../../../../config/worker_scaling_config.js';
import { scalingMonitor } from '../monitoring/scaling_monitor.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class PythonWorkerPool extends EventEmitter {
    constructor(scriptDir, venvDir, maxWorkers = 5) {
        super();
        this.scriptDir = scriptDir;
        this.venvDir = venvDir;

        // Lade Konfiguration
        const config = getWorkerConfig().python;
        this.maxWorkers = maxWorkers || config.maxWorkers;
        this.minWorkers = config.minWorkers;
        this.workers = [];
        this.availableWorkers = [];
        this.busyWorkers = new Set();
        this.nextWorkerId = 0;

        // Skalierungs-Konfiguration aus Config-Datei
        this.scaleUpThreshold = config.scaleUpThreshold;
        this.scaleDownThreshold = config.scaleDownThreshold;
        this.scaleCheckInterval = config.scaleCheckInterval;
        this.idleTimeout = config.idleTimeout;
        
        // Cooldown für Skalierungs-Events (verhindert zu häufige Skalierungen)
        this.lastScaleDownTime = 0;
        this.lastScaleUpTime = 0;
        this.scaleCooldown = 30000; // 30 Sekunden Cooldown zwischen Skalierungen

        this.stats = {
            totalJobsProcessed: 0,
            totalErrors: 0,
            averageExecutionTime: 0,
            queueLength: 0,
            activeWorkers: 0
        };

        // Nur einen Worker initial starten
        this.initializeMinimalWorkers();

        // Job-Queue Events abonnieren
        this.setupQueueListeners();

        // Auto-Scaling starten
        this.startAutoScaling();
    }

    // Minimal Worker initialisieren (nur 1 Worker)
    initializeMinimalWorkers() {
        this.createWorker(this.nextWorkerId++);
    }

    // Auto-Scaling starten
    startAutoScaling() {
        this.scaleTimer = setInterval(() => {
            this.checkAndScale();
        }, this.scaleCheckInterval);
    }

    // Prüfe ob skaliert werden soll
    checkAndScale() {
        const activeWorkers = this.workers.length;
        const busyWorkers = this.busyWorkers.size;
        const queueLength = this.getQueueLength();

        this.stats.activeWorkers = activeWorkers;
        this.stats.queueLength = queueLength;

        // Metriken an Monitor senden
        scalingMonitor.emit('python:metrics', {
            activeWorkers,
            busyWorkers,
            queueLength,
            utilizationRate: busyWorkers / activeWorkers || 0
        });

        // Skaliere hoch wenn:
        // - Warteschlange voll ist ODER
        // - Mehr als 80% der Worker beschäftigt sind UND es Jobs in der Warteschlange gibt
        const shouldScaleUp = (queueLength > 0 && activeWorkers < this.maxWorkers) &&
            (busyWorkers / activeWorkers >= this.scaleUpThreshold || queueLength > activeWorkers);

        // Skaliere runter wenn:
        // - Weniger als 30% der Worker beschäftigt sind UND
        // - Keine Jobs in der Warteschlange UND
        // - Mehr als Minimum-Worker vorhanden
        const shouldScaleDown = activeWorkers > this.minWorkers &&
            queueLength === 0 &&
            busyWorkers / activeWorkers <= this.scaleDownThreshold;

        if (shouldScaleUp) {
            this.scaleUp();
        } else if (shouldScaleDown) {
            this.scaleDown();
        }
    }

    // Worker hinzufügen
    scaleUp() {
        // Prüfe Cooldown
        const now = Date.now();
        if (now - this.lastScaleUpTime < this.scaleCooldown) {
            return; // Cooldown aktiv, keine Skalierung
        }
        
        if (this.workers.length >= this.maxWorkers) {
            return;
        }

        const newWorkerId = this.nextWorkerId++;
        this.createWorker(newWorkerId);

        this.lastScaleUpTime = now;
        
        // Scaling-Event an Monitor senden
        scalingMonitor.emit('python:scaleUp', {
            workerCount: this.workers.length,
            queueLength: this.getQueueLength(),
            reason: 'Hohe Auslastung oder Warteschlange'
        });

        // Nur loggen wenn Logging aktiviert ist
        const globalConfig = getWorkerConfig().global;
        if (globalConfig.enableScalingLogs) {
            console.log(`Python Worker Pool hochskaliert: ${this.workers.length} Worker aktiv`);
        }
    }

    // Worker entfernen (idle Worker)
    scaleDown() {
        // Prüfe Cooldown
        const now = Date.now();
        if (now - this.lastScaleDownTime < this.scaleCooldown) {
            return; // Cooldown aktiv, keine Skalierung
        }
        
        if (this.workers.length <= this.minWorkers) {
            return;
        }

        // Finde einen verfügbaren (nicht beschäftigten) Worker
        const idleWorker = this.availableWorkers.find(worker => !this.busyWorkers.has(worker.workerId));

        if (idleWorker) {
            this.removeWorker(idleWorker.workerId);

            const now = Date.now();
            this.lastScaleDownTime = now;

            // Scaling-Event an Monitor senden
            scalingMonitor.emit('python:scaleDown', {
                workerCount: this.workers.length,
                queueLength: this.getQueueLength(),
                reason: 'Niedrige Auslastung'
            });

            // Nur loggen wenn Logging aktiviert ist
            const globalConfig = getWorkerConfig().global;
            if (globalConfig.enableScalingLogs) {
                console.log(`Python Worker Pool runterskaliert: ${this.workers.length} Worker aktiv`);
            }
        }
    }

    // Worker entfernen
    removeWorker(workerId) {
        const workerIndex = this.workers.findIndex(w => w.workerId === workerId);
        if (workerIndex === -1) return;

        const worker = this.workers[workerIndex];

        // Worker aus allen Listen entfernen
        this.workers.splice(workerIndex, 1);
        this.availableWorkers = this.availableWorkers.filter(w => w.workerId !== workerId);
        this.busyWorkers.delete(workerId);

        // Worker beenden
        worker.terminate();
    }

    // Aktuelle Warteschlangenlänge ermitteln
    getQueueLength() {
        try {
            return jobQueue.getQueueLength();
        } catch (error) {
            return 0;
        }
    }

    // Einzelnen Worker erstellen
    createWorker(workerId) {
        const workerPath = path.join(__dirname, 'python_worker.js');
        const worker = new Worker(workerPath, {
            workerData: {
                workerId,
                scriptDir: this.scriptDir,
                venvDir: this.venvDir
            }
        });

        worker.workerId = workerId;
        worker.currentJobId = null;
        worker.isAvailable = true;

        // Worker-Events
        worker.on('message', (message) => {
            this.handleWorkerMessage(worker, message);
        });

        worker.on('error', (error) => {
            console.error(`Worker ${workerId} Fehler:`, error);
            this.handleWorkerError(worker, error);
        });

        worker.on('exit', (code) => {
            if (code !== 0) {
                console.error(`Worker ${workerId} beendet mit Code ${code}`);
                this.handleWorkerExit(worker, code);
            }
        });

        this.workers.push(worker);
        this.availableWorkers.push(worker);
    }

    // Job-Queue Event-Listener einrichten
    setupQueueListeners() {
        jobQueue.on('executeJob', (job) => {
            this.executeJob(job);
        });
    }

    // Job ausführen
    executeJob(job) {
        const worker = this.getAvailableWorker();
        if (!worker) {
            console.warn(`Kein verfügbarer Worker für Job ${job.id}. Warteschlange...`);
            // Job wird automatisch in der Queue bleiben und später verarbeitet
            return;
        }

        worker.isAvailable = false;
        worker.currentJobId = job.id;
        this.availableWorkers = this.availableWorkers.filter(w => w.workerId !== worker.workerId);
        this.busyWorkers.add(worker);

        console.log(`Worker ${worker.workerId} führt Job ${job.id} (${job.type}) aus`);

        // Job an Worker senden
        worker.postMessage({
            type: 'executeJob',
            job: job
        });
    }

    // Verfügbaren Worker abrufen
    getAvailableWorker() {
        return this.availableWorkers.length > 0 ? this.availableWorkers[0] : null;
    }

    // Worker-Nachrichten verarbeiten
    handleWorkerMessage(worker, message) {
        switch (message.type) {
            case 'jobCompleted':
                this.handleJobCompleted(worker, message);
                break;
            case 'jobFailed':
                this.handleJobFailed(worker, message);
                break;
            case 'jobProgress':
                this.handleJobProgress(worker, message);
                break;
            default:
                console.log(`Unbekannte Worker-Nachricht von Worker ${worker.workerId}:`, message);
        }
    }

    // Job erfolgreich abgeschlossen
    handleJobCompleted(worker, message) {
        const { jobId, result, executionTime } = message;

        // Statistiken aktualisieren
        this.stats.totalJobsProcessed++;
        this.updateAverageExecutionTime(executionTime);

        console.log(`Worker ${worker.workerId} hat Job ${jobId} erfolgreich abgeschlossen (${executionTime}ms)`);

        // Job in Queue als abgeschlossen markieren
        jobQueue.completeJob(jobId, { result });

        // Worker freigeben
        this.releaseWorker(worker);
    }

    // Job fehlgeschlagen
    handleJobFailed(worker, message) {
        const { jobId, error } = message;

        this.stats.totalErrors++;
        console.error(`Worker ${worker.workerId} Job ${jobId} fehlgeschlagen:`, error);

        // Job in Queue als fehlgeschlagen markieren
        jobQueue.completeJob(jobId, { error });

        // Worker freigeben
        this.releaseWorker(worker);
    }

    // Job-Fortschritt
    handleJobProgress(worker, message) {
        const { jobId, progress } = message;
        this.emit('jobProgress', { jobId, progress, workerId: worker.workerId });
    }

    // Worker freigeben
    releaseWorker(worker) {
        worker.isAvailable = true;
        worker.currentJobId = null;
        this.busyWorkers.delete(worker);
        this.availableWorkers.push(worker);

        console.log(`Worker ${worker.workerId} ist wieder verfügbar. Verfügbare Worker: ${this.availableWorkers.length}`);
    }

    // Worker-Fehler behandeln
    handleWorkerError(worker, error) {
        console.error(`Worker ${worker.workerId} Fehler:`, error);

        // Aktuellen Job als fehlgeschlagen markieren
        if (worker.currentJobId) {
            jobQueue.completeJob(worker.currentJobId, {
                error: `Worker-Fehler: ${error.message}`
            });
        }

        // Worker neu starten
        this.restartWorker(worker);
    }

    // Worker-Exit behandeln
    handleWorkerExit(worker, code) {
        console.warn(`Worker ${worker.workerId} beendet mit Code ${code}`);

        // Aktuellen Job als fehlgeschlagen markieren
        if (worker.currentJobId) {
            jobQueue.completeJob(worker.currentJobId, {
                error: `Worker beendet mit Code ${code}`
            });
        }

        // Worker neu starten
        this.restartWorker(worker);
    }

    // Worker neu starten
    restartWorker(worker) {
        const workerId = worker.workerId;

        // Worker aus Arrays entfernen
        this.workers = this.workers.filter(w => w.workerId !== workerId);
        this.availableWorkers = this.availableWorkers.filter(w => w.workerId !== workerId);
        this.busyWorkers.delete(worker);

        // Worker terminieren
        worker.terminate();

        // Neuen Worker erstellen
        setTimeout(() => {
            this.createWorker(workerId);
            console.log(`Worker ${workerId} neu gestartet`);
        }, 1000);
    }

    // Durchschnittliche Ausführungszeit aktualisieren
    updateAverageExecutionTime(executionTime) {
        if (this.stats.totalJobsProcessed === 1) {
            this.stats.averageExecutionTime = executionTime;
        } else {
            // Gleitender Durchschnitt
            const alpha = 0.1; // Gewichtung für neue Werte
            this.stats.averageExecutionTime =
                (1 - alpha) * this.stats.averageExecutionTime + alpha * executionTime;
        }
    }

    // Pool-Status abrufen
    getPoolStatus() {
        return {
            maxWorkers: this.maxWorkers,
            availableWorkers: this.availableWorkers.length,
            busyWorkers: this.busyWorkers.size,
            totalWorkers: this.workers.length,
            stats: { ...this.stats },
            workers: this.workers.map(w => ({
                id: w.workerId,
                isAvailable: w.isAvailable,
                currentJobId: w.currentJobId
            }))
        };
    }

    // Training-Job hinzufügen
    addTrainingJob(projectId, pythonCode, priority = 1) {
        return jobQueue.addJob({
            type: JOB_TYPES.TRAINING,
            priority,
            data: {
                projectId,
                pythonCode,
                jobType: 'training'
            }
        });
    }

    // Re-Training-Job hinzufügen
    addRetrainingJob(projectId, pythonCode, priority = 1) {
        return jobQueue.addJob({
            type: JOB_TYPES.RETRAINING,
            priority,
            data: {
                projectId,
                pythonCode,
                jobType: 'retraining'
            }
        });
    }

    // Prediction-Job hinzufügen
    addPredictionJob(projectId, predictionScript, priority = 0) {
        return jobQueue.addJob({
            type: JOB_TYPES.PREDICTION,
            priority,
            data: {
                projectId,
                predictionScript,
                jobType: 'prediction'
            }
        });
    }

    // Pool herunterfahren
    async shutdown() {
        console.log('Python Worker Pool wird heruntergefahren...');

        // Auto-Scaling Timer stoppen
        if (this.scaleTimer) {
            clearInterval(this.scaleTimer);
            this.scaleTimer = null;
        }

        // Alle Worker terminieren
        const terminationPromises = this.workers.map(worker => {
            return new Promise((resolve) => {
                worker.once('exit', resolve);
                worker.terminate();
            });
        });

        await Promise.all(terminationPromises);
        console.log('Python Worker Pool heruntergefahren');
    }
}

export { PythonWorkerPool };

