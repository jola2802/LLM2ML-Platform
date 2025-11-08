import { parentPort, workerData } from 'worker_threads';
import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';
import path from 'path';
import { JOB_TYPES } from '../monitoring/job_queue.js';
import { executePythonScript as executeWithFix } from './code_exec.js';

const execAsync = promisify(exec);
const { workerId, scriptDir, venvDir } = workerData;

console.log(`Python Worker ${workerId} gestartet`);

// Worker-Hauptlogik
parentPort.on('message', async (message) => {
    if (message.type === 'executeJob') {
        await executeJob(message.job);
    }
});

// Job ausführen
async function executeJob(job) {
    const startTime = Date.now();

    try {
        console.log(`Worker ${workerId}: Starte Job ${job.id} (${job.type})`);

        let result;
        switch (job.type) {
            case JOB_TYPES.TRAINING:
                result = await executeTrainingJob(job);
                break;
            case JOB_TYPES.RETRAINING:
                result = await executeRetrainingJob(job);
                break;
            case JOB_TYPES.PREDICTION:
                result = await executePredictionJob(job);
                break;
            default:
                throw new Error(`Unbekannter Job-Typ: ${job.type}`);
        }

        const executionTime = Date.now() - startTime;

        // Erfolg melden
        parentPort.postMessage({
            type: 'jobCompleted',
            jobId: job.id,
            result,
            executionTime
        });

    } catch (error) {
        console.error(`Worker ${workerId}: Job ${job.id} fehlgeschlagen:`, error.message);

        // Fehler melden
        parentPort.postMessage({
            type: 'jobFailed',
            jobId: job.id,
            error: error.message
        });
    }
}

// Training-Job ausführen
async function executeTrainingJob(job) {
    const { projectId, pythonCode } = job.data;
    const scriptPath = path.join(scriptDir, `${projectId}.py`);

    try {
        // Python-Skript schreiben
        await fs.writeFile(scriptPath, pythonCode);
        console.log(`Worker ${workerId}: Training-Skript für Projekt ${projectId} geschrieben`);

        // Python-Skript ausführen
        const result = await executeWithFix(scriptPath, scriptDir, venvDir);

        // Skript nach erfolgreicher Ausführung löschen
        try {
            await fs.unlink(scriptPath);
            console.log(`Worker ${workerId}: Training-Skript ${scriptPath} gelöscht`);
        } catch (deleteError) {
            console.warn(`Worker ${workerId}: Konnte Training-Skript nicht löschen:`, deleteError.message);
        }

        return {
            type: 'training',
            projectId,
            output: result.stdout,
            stderr: result.stderr,
            success: true
        };

    } catch (error) {
        // Auch bei Fehler versuchen, das Skript zu löschen
        try {
            await fs.unlink(scriptPath);
        } catch (deleteError) {
            // Ignoriere Löschfehler
        }

        throw error;
    }
}

// Re-Training-Job ausführen
async function executeRetrainingJob(job) {
    const { projectId, pythonCode } = job.data;
    const scriptPath = path.join(scriptDir, `${projectId}.py`);

    try {
        // Python-Skript schreiben
        await fs.writeFile(scriptPath, pythonCode);
        console.log(`Worker ${workerId}: Re-Training-Skript für Projekt ${projectId} geschrieben`);

        // Python-Skript ausführen
        const result = await executeWithFix(scriptPath, scriptDir, venvDir);

        // Skript nach erfolgreicher Ausführung löschen
        try {
            await fs.unlink(scriptPath);
            console.log(`Worker ${workerId}: Re-Training-Skript ${scriptPath} gelöscht`);
        } catch (deleteError) {
            console.warn(`Worker ${workerId}: Konnte Re-Training-Skript nicht löschen:`, deleteError.message);
        }

        return {
            type: 'retraining',
            projectId,
            output: result.stdout,
            stderr: result.stderr,
            success: true
        };

    } catch (error) {
        // Auch bei Fehler versuchen, das Skript zu löschen
        try {
            await fs.unlink(scriptPath);
        } catch (deleteError) {
            // Ignoriere Löschfehler
        }

        throw error;
    }
}

// Prediction-Job ausführen
async function executePredictionJob(job) {
    const { projectId, predictionScript } = job.data;
    const scriptPath = path.join(scriptDir, `predict_${projectId}_${Date.now()}.py`);

    try {
        // Python-Skript schreiben
        await fs.writeFile(scriptPath, predictionScript);
        console.log(`Worker ${workerId}: Prediction-Skript für Projekt ${projectId} geschrieben`);

        // Python-Skript ausführen
        const result = await executeWithFix(scriptPath, scriptDir, venvDir);

        // Vorhersage-Ergebnis extrahieren
        const predictionMatch = result.stdout.match(/PREDICTION_RESULT: (.+)/);
        if (!predictionMatch) {
            throw new Error('Konnte Vorhersage-Ergebnis nicht aus der Ausgabe extrahieren');
        }

        const prediction = predictionMatch[1].trim();

        // Skript nach erfolgreicher Ausführung löschen
        try {
            await fs.unlink(scriptPath);
            console.log(`Worker ${workerId}: Prediction-Skript ${scriptPath} gelöscht`);
        } catch (deleteError) {
            console.warn(`Worker ${workerId}: Konnte Prediction-Skript nicht löschen:`, deleteError.message);
        }

        return {
            type: 'prediction',
            projectId,
            prediction,
            output: result.stdout,
            stderr: result.stderr,
            success: true
        };

    } catch (error) {
        // Auch bei Fehler versuchen, das Skript zu löschen
        try {
            await fs.unlink(scriptPath);
        } catch (deleteError) {
            // Ignoriere Löschfehler
        }

        throw error;
    }
}

// Graceful Shutdown
process.on('SIGINT', () => {
    console.log(`Worker ${workerId} wird heruntergefahren...`);
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log(`Worker ${workerId} wird beendet...`);
    process.exit(0);
});

