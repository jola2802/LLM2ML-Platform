import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';

// Job-Status Konstanten
export const JOB_STATUS = {
  PENDING: 'pending',
  RUNNING: 'running',
  COMPLETED: 'completed',
  FAILED: 'failed',
  CANCELLED: 'cancelled'
};

// Job-Typen
export const JOB_TYPES = {
  TRAINING: 'training',
  RETRAINING: 'retraining',
  PREDICTION: 'prediction'
};

class JobQueue extends EventEmitter {
  constructor() {
    super();
    this.jobs = new Map(); // jobId -> job
    this.queue = []; // Array von jobIds in Warteschlange
    this.runningJobs = new Set(); // Set von aktuell laufenden jobIds
    this.maxConcurrentJobs = 5; // Maximal 5 parallele Jobs
    
    // Statistiken
    this.stats = {
      totalJobs: 0,
      completedJobs: 0,
      failedJobs: 0,
      currentlyRunning: 0
    };
  }

  // Job zur Queue hinzufügen
  addJob(jobData) {
    const jobId = uuidv4();
    const job = {
      id: jobId,
      type: jobData.type,
      status: JOB_STATUS.PENDING,
      data: jobData.data,
      priority: jobData.priority || 0, // Höhere Zahl = höhere Priorität
      createdAt: new Date().toISOString(),
      startedAt: null,
      completedAt: null,
      error: null,
      result: null
    };
    
    this.jobs.set(jobId, job);
    this.queue.push(jobId);
    this.stats.totalJobs++;
    
    // Queue nach Priorität sortieren (höchste zuerst)
    this.queue.sort((a, b) => {
      const jobA = this.jobs.get(a);
      const jobB = this.jobs.get(b);
      return jobB.priority - jobA.priority;
    });
    
    console.log(`Job ${jobId} (${job.type}) zur Queue hinzugefügt. Queue-Länge: ${this.queue.length}`);
    
    // Event emittieren
    this.emit('jobAdded', job);
    
    // Versuche Job sofort zu starten
    this.processQueue();
    
    return jobId;
  }

  // Queue verarbeiten
  processQueue() {
    // Prüfe ob wir freie Worker haben
    if (this.runningJobs.size >= this.maxConcurrentJobs) {
      return; // Alle Worker sind beschäftigt
    }
    
    // Nächster Job aus der Queue
    if (this.queue.length === 0) {
      return; // Keine Jobs in der Queue
    }
    
    const jobId = this.queue.shift();
    const job = this.jobs.get(jobId);
    
    if (!job || job.status !== JOB_STATUS.PENDING) {
      // Job wurde bereits verarbeitet oder abgebrochen
      this.processQueue(); // Versuche nächsten Job
      return;
    }
    
    // Job starten
    this.startJob(jobId);
    
    // Rekursiv weitere Jobs starten falls noch Worker frei sind
    if (this.runningJobs.size < this.maxConcurrentJobs && this.queue.length > 0) {
      this.processQueue();
    }
  }

  // Job starten
  async startJob(jobId) {
    const job = this.jobs.get(jobId);
    if (!job) return;
    
    job.status = JOB_STATUS.RUNNING;
    job.startedAt = new Date().toISOString();
    this.runningJobs.add(jobId);
    this.stats.currentlyRunning++;
    
    console.log(`Job ${jobId} (${job.type}) gestartet. Laufende Jobs: ${this.runningJobs.size}`);
    
    // Event emittieren
    this.emit('jobStarted', job);
    
    try {
      // Job ausführen (wird von Worker übernommen)
      this.emit('executeJob', job);
    } catch (error) {
      this.completeJob(jobId, { error: error.message });
    }
  }

  // Job abschließen
  completeJob(jobId, result = {}) {
    const job = this.jobs.get(jobId);
    if (!job) return;
    
    job.completedAt = new Date().toISOString();
    job.result = result.result || null;
    job.error = result.error || null;
    job.status = result.error ? JOB_STATUS.FAILED : JOB_STATUS.COMPLETED;
    
    this.runningJobs.delete(jobId);
    this.stats.currentlyRunning--;
    
    if (result.error) {
      this.stats.failedJobs++;
      console.log(`Job ${jobId} (${job.type}) fehlgeschlagen: ${result.error}`);
    } else {
      this.stats.completedJobs++;
      console.log(`Job ${jobId} (${job.type}) erfolgreich abgeschlossen`);
    }
    
    // Event emittieren
    this.emit('jobCompleted', job);
    
    // Nächste Jobs aus der Queue verarbeiten
    this.processQueue();
  }

  // Job abbrechen
  cancelJob(jobId) {
    const job = this.jobs.get(jobId);
    if (!job) return false;
    
    if (job.status === JOB_STATUS.PENDING) {
      // Job aus Queue entfernen
      const index = this.queue.indexOf(jobId);
      if (index > -1) {
        this.queue.splice(index, 1);
      }
      job.status = JOB_STATUS.CANCELLED;
      job.completedAt = new Date().toISOString();
      
      console.log(`Job ${jobId} (${job.type}) abgebrochen (war in Queue)`);
      this.emit('jobCancelled', job);
      return true;
    } else if (job.status === JOB_STATUS.RUNNING) {
      // Laufenden Job markieren (Worker muss abbrechen)
      job.status = JOB_STATUS.CANCELLED;
      console.log(`Job ${jobId} (${job.type}) zum Abbruch markiert (läuft gerade)`);
      this.emit('jobCancelled', job);
      return true;
    }
    
    return false;
  }

  // Job-Status abrufen
  getJob(jobId) {
    return this.jobs.get(jobId);
  }

  // Alle Jobs abrufen
  getAllJobs(limit = 50) {
    const allJobs = Array.from(this.jobs.values())
      .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
      .slice(0, limit);
    return allJobs;
  }

  // Queue-Status abrufen
  getQueueStatus() {
    return {
      queueLength: this.queue.length,
      runningJobs: this.runningJobs.size,
      maxConcurrentJobs: this.maxConcurrentJobs,
      stats: { ...this.stats }
    };
  }

  // Warteschlangenlänge abrufen
  getQueueLength() {
    return this.queue.length;
  }

  // Jobs nach Typ abrufen
  getJobsByType(type, limit = 20) {
    return Array.from(this.jobs.values())
      .filter(job => job.type === type)
      .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
      .slice(0, limit);
  }

  // Alte Jobs aufräumen (älter als 24 Stunden)
  cleanupOldJobs() {
    const cutoffTime = new Date(Date.now() - 24 * 60 * 60 * 1000);
    let cleaned = 0;
    
    for (const [jobId, job] of this.jobs.entries()) {
      if (job.status === JOB_STATUS.COMPLETED || job.status === JOB_STATUS.FAILED) {
        if (new Date(job.createdAt) < cutoffTime) {
          this.jobs.delete(jobId);
          cleaned++;
        }
      }
    }
    
    if (cleaned > 0) {
      console.log(`${cleaned} alte Jobs aufgeräumt`);
    }
    
    return cleaned;
  }
}

// Singleton-Instanz
export const jobQueue = new JobQueue();

// Automatische Aufräumung alle 6 Stunden
setInterval(() => {
  jobQueue.cleanupOldJobs();
}, 6 * 60 * 60 * 1000);
