import fs from 'fs/promises';
import path from 'path';

// Bereinigung alter Predict-Skripte (wird vom Server regelmäßig aufgerufen)
export async function cleanupOldPredictScripts(scriptDir, maxAgeHours = 168) { // 7 Tage Standard
    try {
        const files = await fs.readdir(scriptDir);
        const predictFiles = files.filter(file =>
            file.startsWith('predict_') && (file.endsWith('.py') || file.endsWith('_metadata.json'))
        );

        // Auch projekt-spezifische Target-Encoder berücksichtigen
        const targetEncoderFiles = files.filter(file =>
            file.startsWith('target_encoder_') && file.endsWith('.pkl')
        );

        let cleaned = 0;
        const cutoffTime = new Date(Date.now() - maxAgeHours * 60 * 60 * 1000);

        for (const file of predictFiles) {
            try {
                // Prüfe nur Metadata-Dateien für Zeitstempel
                if (file.endsWith('_metadata.json')) {
                    const metadataPath = path.join(scriptDir, file);
                    const metadataContent = await fs.readFile(metadataPath, 'utf-8');
                    const metadata = JSON.parse(metadataContent);

                    const lastUsed = new Date(metadata.lastUsed || metadata.createdAt);

                    if (lastUsed < cutoffTime) {
                        // Lösche sowohl Metadata als auch das entsprechende Python-Skript und Target-Encoder
                        const projectId = metadata.projectId;
                        const scriptPath = path.join(scriptDir, `predict_${projectId}.py`);
                        const targetEncoderPath = path.join(scriptDir, `target_encoder_${projectId}.pkl`);

                        await fs.unlink(metadataPath);

                        try {
                            await fs.unlink(scriptPath);
                        } catch (scriptError) {
                            // Python-Datei existiert möglicherweise nicht
                        }

                        // Projekt-spezifischen Target-Encoder auch löschen
                        try {
                            await fs.unlink(targetEncoderPath);
                            console.log(`Target-Encoder für Projekt ${projectId} auch bereinigt`);
                        } catch (encoderError) {
                            // Target-Encoder existiert möglicherweise nicht
                        }

                        cleaned++;
                        console.log(`Altes Predict-Skript für Projekt ${projectId} bereinigt (zuletzt verwendet: ${lastUsed.toISOString()})`);
                    }
                }
            } catch (fileError) {
                console.log(`Fehler beim Bereinigen der Datei ${file}:`, fileError.message);
            }
        }

        if (cleaned > 0) {
            console.log(`${cleaned} alte Predict-Skripte erfolgreich bereinigt`);
        }

        return cleaned;
    } catch (error) {
        console.error('Fehler bei der Bereinigung alter Predict-Skripte:', error.message);
        return 0;
    }
}

