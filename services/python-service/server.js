import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { PythonWorkerPool } from './services/execution/python_worker_pool.js';
import { jobQueue } from './services/monitoring/job_queue.js';
import { setupDataRoutes } from './routes/data_routes.js';
import { setupExecutionRoutes } from './routes/execution_routes.js';
import { sendJobCompletionWebhook } from './services/utils/webhook_client.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3003;

// CORS für Hauptserver
app.use(cors({
    origin: process.env.API_GATEWAY_URL || 'http://localhost:3001',
    credentials: true
}));

app.use(express.json({ limit: '50mb' }));

// Verzeichnisse
const scriptDir = process.env.SCRIPT_DIR || '/app/scripts';
const venvDir = process.env.VENV_DIR || '/opt/venv';
const uploadsDir = process.env.UPLOADS_DIR || '/app/uploads';
const modelsDir = process.env.MODELS_DIR || '/app/models';

// Python Worker Pool initialisieren
const pythonWorkerPool = new PythonWorkerPool(scriptDir, venvDir, 5);

// Worker Pool Event-Handler
pythonWorkerPool.on('jobProgress', ({ jobId, progress, workerId }) => {
    console.log(`Worker ${workerId} Progress für Job ${jobId}:`, progress);
});

// Job-Completion-Handler - Sendet Webhook an API Gateway
jobQueue.on('jobCompleted', async (job) => {
    console.log(`Job ${job.id} abgeschlossen:`, job.status);

    // Webhook an API Gateway senden
    const projectId = job.data?.projectId;
    if (projectId) {
        try {
            await sendJobCompletionWebhook(
                job.id,
                job.type,
                projectId,
                job.result,
                job.status
            );
        } catch (error) {
            console.error(`Fehler beim Senden des Webhooks für Job ${job.id}:`, error.message);
        }
    }
});

// Health-Check Endpoint
app.get('/health', (req, res) => {
    const poolStatus = pythonWorkerPool.getPoolStatus();
    res.json({
        status: 'healthy',
        service: 'python-service',
        port: PORT,
        pool: {
            totalWorkers: poolStatus.totalWorkers,
            availableWorkers: poolStatus.availableWorkers,
            busyWorkers: poolStatus.busyWorkers
        }
    });
});

// API-Routen
setupDataRoutes(app, venvDir, uploadsDir);
setupExecutionRoutes(app, pythonWorkerPool, scriptDir, venvDir, modelsDir);

// Fehlerbehandlung
app.use((err, req, res, next) => {
    console.error('Fehler:', err);
    res.status(500).json({
        error: err.message || 'Interner Serverfehler',
        service: 'python-service'
    });
});

// Server starten
app.listen(PORT, () => {
    console.log(`Python-Service läuft auf Port ${PORT}`);
    console.log(`Script-Dir: ${scriptDir}`);
    console.log(`Venv-Dir: ${venvDir}`);
    console.log(`Uploads-Dir: ${uploadsDir}`);
});

// Graceful Shutdown
process.on('SIGTERM', async () => {
    console.log('SIGTERM empfangen, fahre Service herunter...');
    await pythonWorkerPool.shutdown();
    process.exit(0);
});

process.on('SIGINT', async () => {
    console.log('SIGINT empfangen, fahre Service herunter...');
    await pythonWorkerPool.shutdown();
    process.exit(0);
});
