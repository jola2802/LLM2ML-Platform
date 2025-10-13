import { Worker } from 'worker_threads';
import path from 'path';
import { fileURLToPath } from 'url';
import { EventEmitter } from 'events';
import { getWorkerConfig } from './worker_scaling_config.js';
import { scalingMonitor } from '../../monitoring/scaling_monitor.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Queue für LLM-Anfragen mit Worker Threads
class LLMQueue extends EventEmitter {
  constructor(maxConcurrency = 3, maxWorkers = 3) {
    super();
    
    // Lade Konfiguration
    const config = getWorkerConfig().llm;
    this.maxConcurrency = maxConcurrency || config.maxConcurrency;
    this.maxWorkers = maxWorkers || config.maxWorkers;
    this.minWorkers = config.minWorkers;
    this.workers = [];
    this.queue = [];
    this.processing = new Map(); // RequestId -> Worker
    this.requestCounter = 0;
    this.results = new Map(); // RequestId -> Result
    this.nextWorkerId = 0;
    
    // Skalierungs-Konfiguration aus Config-Datei
    this.scaleUpThreshold = config.scaleUpThreshold;
    this.scaleDownThreshold = config.scaleDownThreshold;
    this.scaleCheckInterval = config.scaleCheckInterval;
    this.idleTimeout = config.idleTimeout;
    
    // Initialisiere nur minimale Worker
    this.initializeMinimalWorkers();
    
    // Auto-Scaling starten
    this.startAutoScaling();
    
    // console.log(`LLM Queue initialisiert: ${this.minWorkers} Worker, max ${maxWorkers} Worker, max ${maxConcurrency} parallele Anfragen`);
  }
  
  // Minimal Worker initialisieren (nur 1 Worker)
  initializeMinimalWorkers() {
    this.createWorker();
  }

  // Auto-Scaling starten
  startAutoScaling() {
    this.scaleTimer = setInterval(() => {
      this.checkAndScale();
    }, this.scaleCheckInterval);
    
    // console.log('Auto-Scaling für LLM Queue aktiviert');
  }

  // Prüfe ob skaliert werden soll
  checkAndScale() {
    const activeWorkers = this.workers.length;
    const busyWorkers = this.processing.size;
    const queueLength = this.queue.length;
    
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
    
    // Metriken an Monitor senden
    scalingMonitor.emit('llm:metrics', {
      activeWorkers,
      busyWorkers,
      queueLength,
      utilizationRate: busyWorkers / activeWorkers || 0
    });
    
    if (shouldScaleUp) {
      this.scaleUp();
    } else if (shouldScaleDown) {
      this.scaleDown();
    }
  }

  // Worker hinzufügen
  scaleUp() {
    if (this.workers.length >= this.maxWorkers) {
      return;
    }
    
    this.createWorker();
    
    // Scaling-Event an Monitor senden
    scalingMonitor.emit('llm:scaleUp', {
      workerCount: this.workers.length,
      queueLength: this.queue.length,
      reason: 'Hohe Auslastung oder Warteschlange'
    });
    
    console.log(`LLM Queue hochskaliert: ${this.workers.length} Worker aktiv`);
  }

  // Worker entfernen (idle Worker)
  scaleDown() {
    if (this.workers.length <= this.minWorkers) {
      return;
    }
    
    // Finde einen verfügbaren (nicht beschäftigten) Worker
    const idleWorker = this.workers.find(worker => {
      // Prüfe ob Worker beschäftigt ist
      for (const [requestId, busyWorker] of this.processing.entries()) {
        if (busyWorker === worker) {
          return false; // Worker ist beschäftigt
        }
      }
      return true; // Worker ist frei
    });
    
    if (idleWorker) {
      this.removeWorker(idleWorker);
      
      // Scaling-Event an Monitor senden
      scalingMonitor.emit('llm:scaleDown', {
        workerCount: this.workers.length,
        queueLength: this.queue.length,
        reason: 'Niedrige Auslastung'
      });
      
      console.log(`LLM Queue runterskaliert: ${this.workers.length} Worker aktiv`);
    }
  }
  
  createWorker() {
    const workerPath = path.join(__dirname, 'llm_worker.js');
    const worker = new Worker(workerPath);
    
    worker.on('message', (message) => {
      this.handleWorkerMessage(worker, message);
    });
    
    worker.on('error', (error) => {
      console.error('Worker Fehler:', error);
      this.handleWorkerError(worker, error);
    });
    
    worker.on('exit', (code) => {
      console.log(`Worker beendet mit Code ${code}`);
      this.removeWorker(worker);
    });
    
    worker.isIdle = true;
    worker.currentRequestId = null;
    this.workers.push(worker);
    
    return worker;
  }
  
  handleWorkerMessage(worker, message) {
    const { type, requestId, data, error } = message;
    
    switch (type) {
      case 'result':
        this.handleRequestComplete(worker, requestId, data);
        break;
      case 'error':
        this.handleRequestError(worker, requestId, error);
        break;
      case 'progress':
        this.emit('progress', { requestId, ...data });
        break;
      default:
        console.warn('Unbekannter Worker Message Type:', type);
    }
  }
  
  handleRequestComplete(worker, requestId, result) {
    // Worker als idle markieren
    worker.isIdle = true;
    worker.currentRequestId = null;
    
    // Finde den ursprünglichen Request um resolve zu callen
    const requestInfo = this.processing.get(requestId);
    if (requestInfo && requestInfo.resolve) {
      requestInfo.resolve(result);
    }
    
    // Ergebnis speichern für eventuelle spätere Abfragen
    this.results.set(requestId, { success: true, data: result });
    this.processing.delete(requestId);
    
    // Event emittieren
    this.emit('requestComplete', { requestId, result });
    
    // Nächste Anfrage in der Queue verarbeiten
    this.processQueue();
    
  }
  
  handleRequestError(worker, requestId, error) {
    // Worker als idle markieren
    worker.isIdle = true;
    worker.currentRequestId = null;
    
    // Finde den ursprünglichen Request um reject zu callen
    const requestInfo = this.processing.get(requestId);
    if (requestInfo && requestInfo.reject) {
      requestInfo.reject(new Error(error));
    }
    
    // Fehler speichern
    this.results.set(requestId, { success: false, error: error });
    this.processing.delete(requestId);
    
    // Event emittieren
    this.emit('requestError', { requestId, error });
    
    // Nächste Anfrage in der Queue verarbeiten
    this.processQueue();
    
    console.error(`Request ${requestId} fehlgeschlagen:`, error);
  }
  
  handleWorkerError(worker, error) {
    // Alle aktiven Requests dieses Workers als fehlgeschlagen markieren
    if (worker.currentRequestId) {
      const requestId = worker.currentRequestId;
      const requestInfo = this.processing.get(requestId);
      
      // Reject den ursprünglichen Promise
      if (requestInfo && requestInfo.reject) {
        requestInfo.reject(new Error(`Worker Fehler: ${error.message}`));
      }
      
      this.results.set(requestId, { 
        success: false, 
        error: `Worker Fehler: ${error.message}` 
      });
      this.processing.delete(requestId);
      this.emit('requestError', { requestId, error: error.message });
    }
    
    // Worker entfernen und neuen erstellen
    this.removeWorker(worker);
    this.createWorker();
  }
  
  removeWorker(worker) {
    const index = this.workers.indexOf(worker);
    if (index !== -1) {
      this.workers.splice(index, 1);
    }
    
    if (!worker.destroyed) {
      worker.terminate();
    }
  }
  
  // Hauptfunktion: LLM-Anfrage zur Queue hinzufügen
  async addRequest(prompt, filePath = null, customModel = null, maxRetries = 3, timeout = 120000) {
    const requestId = ++this.requestCounter;
    
    return new Promise((resolve, reject) => {
      const request = {
        id: requestId,
        prompt,
        filePath,
        customModel,
        maxRetries,
        timeout,
        resolve,
        reject,
        createdAt: Date.now()
      };
      
      // Request zur Queue hinzufügen
      this.queue.push(request);
      
      // Versuche sofort zu verarbeiten
      this.processQueue();
      
      // Timeout für die gesamte Anfrage
      setTimeout(() => {
        if (this.processing.has(requestId) || this.queue.find(r => r.id === requestId)) {
          this.cancelRequest(requestId, 'Timeout');
          reject(new Error('LLM Request Timeout'));
        }
      }, timeout);
    });
  }
  
  processQueue() {
    // Prüfe ob freie Worker verfügbar sind
    const idleWorkers = this.workers.filter(worker => worker.isIdle);
    
    if (idleWorkers.length === 0 || this.queue.length === 0) {
      return; // Keine Worker oder keine Anfragen
    }
    
    // Verarbeite so viele Anfragen wie möglich
    const maxProcessable = Math.min(
      idleWorkers.length, 
      this.queue.length, 
      this.maxConcurrency - this.processing.size
    );
    
    for (let i = 0; i < maxProcessable; i++) {
      const worker = idleWorkers[i];
      const request = this.queue.shift();
      
      if (!worker || !request) break;
      
      // Worker als beschäftigt markieren
      worker.isIdle = false;
      worker.currentRequestId = request.id;
      
      // Request als in Bearbeitung markieren (mit resolve/reject für Callback)
      this.processing.set(request.id, {
        worker: worker,
        resolve: request.resolve,
        reject: request.reject,
        request: request
      });
      
      // Sende Request an Worker
      worker.postMessage({
        type: 'processRequest',
        requestId: request.id,
        prompt: request.prompt,
        filePath: request.filePath,
        customModel: request.customModel,
        maxRetries: request.maxRetries
      });
    }
  }
  
  // Request abbrechen
  cancelRequest(requestId, reason = 'Cancelled') {
    // Aus Queue entfernen
    const queueIndex = this.queue.findIndex(r => r.id === requestId);
    if (queueIndex !== -1) {
      const request = this.queue.splice(queueIndex, 1)[0];
      request.reject(new Error(`Request cancelled: ${reason}`));
      return true;
    }
    
    // Aus Processing entfernen
    const requestInfo = this.processing.get(requestId);
    if (requestInfo) {
      const worker = requestInfo.worker;
      worker.postMessage({
        type: 'cancelRequest',
        requestId: requestId
      });
      
      worker.isIdle = true;
      worker.currentRequestId = null;
      
      // Reject den ursprünglichen Promise
      if (requestInfo.reject) {
        requestInfo.reject(new Error(`Request cancelled: ${reason}`));
      }
      
      this.processing.delete(requestId);
      
      // Nächste Anfrage verarbeiten
      this.processQueue();
      return true;
    }
    
    return false;
  }
  
  // Status der Queue abrufen
  getStatus() {
    return {
      queueSize: this.queue.length,
      processing: this.processing.size,
      workers: {
        total: this.workers.length,
        idle: this.workers.filter(w => w.isIdle).length,
        busy: this.workers.filter(w => !w.isIdle).length
      },
      maxConcurrency: this.maxConcurrency,
      uptime: process.uptime()
    };
  }
  
  // Graceful Shutdown
  async shutdown() {
    console.log('Beende LLM Queue...');
    
    // Auto-Scaling Timer stoppen
    if (this.scaleTimer) {
      clearInterval(this.scaleTimer);
      this.scaleTimer = null;
    }
    
    // Neue Anfragen ablehnen
    this.queue.forEach(request => {
      request.reject(new Error('Queue wird heruntergefahren'));
    });
    this.queue = [];
    
    // Alle Worker beenden
    const shutdownPromises = this.workers.map(worker => {
      return new Promise((resolve) => {
        worker.on('exit', resolve);
        worker.postMessage({ type: 'shutdown' });
        
        // Force terminate nach 5 Sekunden
        setTimeout(() => {
          if (!worker.exitCode) {
            worker.terminate();
          }
          resolve();
        }, 5000);
      });
    });
    
    await Promise.all(shutdownPromises);
    console.log('LLM Queue beendet');
  }
}

// Singleton Instance
export const llmQueue = new LLMQueue(3, 3); // 3 parallele Anfragen, 3 Worker

// Graceful Shutdown bei Prozess-Ende
process.on('SIGINT', async () => {
  await llmQueue.shutdown();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  await llmQueue.shutdown();
  process.exit(0);
});

export default llmQueue;