import { jobQueue } from '../../monitoring/job_queue.js';
import { logRESTAPIRequest } from '../../monitoring/log.js';

export function setupWorkerStatusRoutes(app, pythonWorkerPool) {
  
  // Queue-Status abrufen
  app.get('/api/worker/queue-status', async (req, res) => {
    try {
      logRESTAPIRequest('get-queue-status', {});
      
      const queueStatus = jobQueue.getQueueStatus();
      const poolStatus = pythonWorkerPool.getPoolStatus();
      
      res.json({
        queue: queueStatus,
        workerPool: poolStatus,
        combinedStatus: {
          totalJobs: queueStatus.stats.totalJobs,
          queueLength: queueStatus.queueLength,
          activeWorkers: poolStatus.busyWorkers,
          availableWorkers: poolStatus.availableWorkers,
          maxWorkers: poolStatus.maxWorkers
        }
      });
      
    } catch (error) {
      console.error('Fehler beim Abrufen des Queue-Status:', error);
      res.status(500).json({ error: error.message });
    }
  });

  // Alle Jobs abrufen
  app.get('/api/worker/jobs', async (req, res) => {
    try {
      logRESTAPIRequest('get-all-jobs', req.query);
      
      const limit = parseInt(req.query.limit) || 50;
      const jobs = jobQueue.getAllJobs(limit);
      
      res.json({ jobs });
      
    } catch (error) {
      console.error('Fehler beim Abrufen der Jobs:', error);
      res.status(500).json({ error: error.message });
    }
  });

  // Jobs nach Typ abrufen
  app.get('/api/worker/jobs/:type', async (req, res) => {
    try {
      logRESTAPIRequest('get-jobs-by-type', req.params);
      
      const { type } = req.params;
      const limit = parseInt(req.query.limit) || 20;
      const jobs = jobQueue.getJobsByType(type, limit);
      
      res.json({ jobs, type });
      
    } catch (error) {
      console.error('Fehler beim Abrufen der Jobs nach Typ:', error);
      res.status(500).json({ error: error.message });
    }
  });

  // Einzelnen Job abrufen
  app.get('/api/worker/job/:jobId', async (req, res) => {
    try {
      logRESTAPIRequest('get-job', req.params);
      
      const { jobId } = req.params;
      const job = jobQueue.getJob(jobId);
      
      if (!job) {
        return res.status(404).json({ error: 'Job nicht gefunden' });
      }
      
      res.json({ job });
      
    } catch (error) {
      console.error('Fehler beim Abrufen des Jobs:', error);
      res.status(500).json({ error: error.message });
    }
  });

  // Job abbrechen
  app.post('/api/worker/job/:jobId/cancel', async (req, res) => {
    try {
      logRESTAPIRequest('cancel-job', req.params);
      
      const { jobId } = req.params;
      const success = jobQueue.cancelJob(jobId);
      
      if (success) {
        res.json({ message: 'Job erfolgreich abgebrochen', jobId });
      } else {
        res.status(400).json({ error: 'Job konnte nicht abgebrochen werden' });
      }
      
    } catch (error) {
      console.error('Fehler beim Abbrechen des Jobs:', error);
      res.status(500).json({ error: error.message });
    }
  });

  // Worker-Pool-Statistiken
  app.get('/api/worker/stats', async (req, res) => {
    try {
      logRESTAPIRequest('get-worker-stats', {});
      
      const poolStatus = pythonWorkerPool.getPoolStatus();
      const queueStatus = jobQueue.getQueueStatus();
      
      const stats = {
        workerPool: {
          maxWorkers: poolStatus.maxWorkers,
          activeWorkers: poolStatus.busyWorkers,
          availableWorkers: poolStatus.availableWorkers,
          totalJobsProcessed: poolStatus.stats.totalJobsProcessed,
          totalErrors: poolStatus.stats.totalErrors,
          averageExecutionTime: Math.round(poolStatus.stats.averageExecutionTime)
        },
        queue: {
          currentQueueLength: queueStatus.queueLength,
          totalJobsCreated: queueStatus.stats.totalJobs,
          completedJobs: queueStatus.stats.completedJobs,
          failedJobs: queueStatus.stats.failedJobs,
          currentlyRunning: queueStatus.stats.currentlyRunning
        },
        performance: {
          successRate: queueStatus.stats.totalJobs > 0 
            ? Math.round((queueStatus.stats.completedJobs / queueStatus.stats.totalJobs) * 100) 
            : 0,
          failureRate: queueStatus.stats.totalJobs > 0 
            ? Math.round((queueStatus.stats.failedJobs / queueStatus.stats.totalJobs) * 100) 
            : 0
        }
      };
      
      res.json(stats);
      
    } catch (error) {
      console.error('Fehler beim Abrufen der Worker-Statistiken:', error);
      res.status(500).json({ error: error.message });
    }
  });

}
