import {
    executePythonScript,
    extractMetricsFromOutput,
    generatePredictionScript,
    predictWithModel,
    cleanupOldPredictScripts
} from '../services/execution/code_exec.js';
import { JOB_TYPES, JOB_STATUS, jobQueue } from '../services/monitoring/job_queue.js';
import fs from 'fs/promises';
import path from 'path';

export function setupExecutionRoutes(app, pythonWorkerPool, scriptDir, venvDir, modelsDir) {
    // Training-Job starten
    app.post('/api/execution/train', async (req, res) => {
        try {
            const { projectId, pythonCode } = req.body;

            if (!projectId || !pythonCode) {
                return res.status(400).json({ error: 'projectId und pythonCode sind erforderlich' });
            }

            console.log(`Training-Job gestartet für Projekt: ${projectId}`);

            // Job zur Queue hinzufügen
            const jobId = pythonWorkerPool.addTrainingJob(projectId, pythonCode, 1);

            res.json({
                success: true,
                jobId,
                projectId,
                status: 'pending',
                message: 'Training-Job zur Queue hinzugefügt'
            });

        } catch (error) {
            console.error('Fehler beim Starten des Training-Jobs:', error);
            res.status(500).json({
                error: `Fehler beim Starten des Training-Jobs: ${error.message}`
            });
        }
    });

    // Retraining-Job starten
    app.post('/api/execution/retrain', async (req, res) => {
        try {
            const { projectId, pythonCode } = req.body;

            if (!projectId || !pythonCode) {
                return res.status(400).json({ error: 'projectId und pythonCode sind erforderlich' });
            }

            console.log(`Retraining-Job gestartet für Projekt: ${projectId}`);

            // Job zur Queue hinzufügen
            const jobId = pythonWorkerPool.addRetrainingJob(projectId, pythonCode, 1);

            res.json({
                success: true,
                jobId,
                projectId,
                status: 'pending',
                message: 'Retraining-Job zur Queue hinzugefügt'
            });

        } catch (error) {
            console.error('Fehler beim Starten des Retraining-Jobs:', error);
            res.status(500).json({
                error: `Fehler beim Starten des Retraining-Jobs: ${error.message}`
            });
        }
    });

    // Python-Code direkt ausführen
    app.post('/api/execution/execute', async (req, res) => {
        try {
            const { code, projectId } = req.body;

            if (!code) {
                return res.status(400).json({ error: 'code ist erforderlich' });
            }

            const scriptPath = path.join(scriptDir, `execute_${projectId || Date.now()}.py`);

            try {
                // Code in temporäre Datei schreiben
                await fs.writeFile(scriptPath, code);

                // Code ausführen
                const result = await executePythonScript(scriptPath, scriptDir, venvDir);

                // Temporäre Datei löschen
                try {
                    await fs.unlink(scriptPath);
                } catch (deleteError) {
                    console.warn('Konnte temporäre Datei nicht löschen:', deleteError.message);
                }

                res.json({
                    success: true,
                    output: result.stdout,
                    stderr: result.stderr
                });

            } catch (error) {
                // Versuche temporäre Datei zu löschen
                try {
                    await fs.unlink(scriptPath);
                } catch (deleteError) {
                    // Ignoriere Löschfehler
                }

                throw error;
            }

        } catch (error) {
            console.error('Fehler bei Code-Ausführung:', error);
            res.status(500).json({
                error: `Fehler bei Code-Ausführung: ${error.message}`
            });
        }
    });

    // Job-Status abfragen
    app.get('/api/execution/jobs/:id', (req, res) => {
        try {
            const { id } = req.params;

            const job = jobQueue.getJob(id);

            if (!job) {
                return res.status(404).json({ error: 'Job nicht gefunden' });
            }

            res.json({
                id: job.id,
                type: job.type,
                status: job.status,
                createdAt: job.createdAt,
                startedAt: job.startedAt,
                completedAt: job.completedAt,
                error: job.error,
                result: job.result
            });

        } catch (error) {
            console.error('Fehler beim Abfragen des Job-Status:', error);
            res.status(500).json({
                error: `Fehler beim Abfragen des Job-Status: ${error.message}`
            });
        }
    });

    // Worker-Status abfragen
    app.get('/api/execution/status', (req, res) => {
        try {
            const poolStatus = pythonWorkerPool.getPoolStatus();
            const queueStatus = jobQueue.getQueueStatus();

            res.json({
                pool: poolStatus,
                queue: queueStatus
            });

        } catch (error) {
            console.error('Fehler beim Abfragen des Worker-Status:', error);
            res.status(500).json({
                error: `Fehler beim Abfragen des Worker-Status: ${error.message}`
            });
        }
    });

    // Prediction ausführen
    app.post('/api/execution/predict', async (req, res) => {
        try {
            const { project, inputFeatures } = req.body;

            if (!project || !inputFeatures) {
                return res.status(400).json({ error: 'project und inputFeatures sind erforderlich' });
            }

            console.log(`Prediction gestartet für Projekt: ${project.id}`);

            const prediction = await predictWithModel(project, inputFeatures, scriptDir, venvDir, modelsDir);

            res.json({
                success: true,
                prediction
            });

        } catch (error) {
            console.error('Fehler bei Prediction:', error);
            res.status(500).json({
                error: `Fehler bei Prediction: ${error.message}`
            });
        }
    });
}

