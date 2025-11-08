import fs from 'fs/promises';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';
import { codeTemplate } from './template_code.js';
import { predictionScript } from './template_predict.js';

import fsSync from 'fs';
import crypto from 'crypto';
const execAsync = promisify(exec);

// Hash-Generierung für Projekt-Änderungserkennung
function generateProjectHash(project) {
    // Erstelle einen Hash aus den relevanten Projekt-Eigenschaften
    // die sich auf das Predict-Skript auswirken könnten
    const relevantData = {
        id: project.id,
        modelType: project.modelType,
        algorithm: project.algorithm,
        hyperparameters: project.hyperparameters,
        pythonCode: project.pythonCode,
        originalPythonCode: project.originalPythonCode,
        modelPath: project.modelPath,
        features: project.features,
        targetVariable: project.targetVariable
    };

    const dataString = JSON.stringify(relevantData, Object.keys(relevantData).sort());
    return crypto.createHash('sha256').update(dataString).digest('hex');
}

// Zentrale Fehlerbehandlungsfunktion
function handleError(error, context, attempt = null) {
    const errorInfo = {
        message: error.message,
        context: context,
        attempt: attempt,
        timestamp: new Date().toISOString(),
        stack: error.stack
    };

    console.error(`[${context}] Fehler${attempt ? ` (Versuch ${attempt})` : ''}:`, errorInfo);

    // Log spezifische Fehlertypen
    if (error.message.includes('ENOENT')) {
        console.error(`[${context}] Datei nicht gefunden`);
    } else if (error.message.includes('EACCES')) {
        console.error(`[${context}] Keine Berechtigung`);
    } else if (error.message.includes('ECONNREFUSED')) {
        console.error(`[${context}] Verbindung verweigert`);
    } else if (error.message.includes('timeout')) {
        console.error(`[${context}] Timeout`);
    }

    return errorInfo;
}

// Überprüfe und validiere den Python-Code (ohne LLM)
export async function validatePythonCode(pythonCode) {
    try {
        // Sicherstellen, dass wichtige Imports vorhanden sind
        if (!pythonCode.includes('import pandas')) {
            pythonCode = 'import pandas as pd\n' + pythonCode;
        }
        if (!pythonCode.includes('import joblib')) {
            pythonCode = pythonCode.replace('import pandas as pd', 'import pandas as pd\nimport joblib');
        }

        // Sicherstellen, dass die Datei nicht ```python oder ```json enthält
        pythonCode = pythonCode.replace('```python', '').replace('```json', '').replace('```', '').replace('`', '');

        return pythonCode.trim();
    } catch (error) {
        handleError(error, 'validatePythonCode');
        throw error;
    }
}

// Wendet deterministische Korrekturen für bekannte Fehlerbilder an
function applyDeterministicFixes(pythonCode, errorText) {
    try {
        let fixed = pythonCode;

        // 1) LabelEncoder fälschlich in Feature-Pipeline/ColumnTransformer verwendet
        // Ersetze LabelEncoder() durch OneHotEncoder(...) sofern NICHT für das Target (Zeilen mit target_encoder ignorieren)
        if (/LabelEncoder\.fit_transform\(\)/.test(errorText) || /LabelEncoder\(\)/.test(fixed)) {
            const lines = fixed.split('\n');
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                if (line.includes('LabelEncoder(') && !line.includes('target_encoder')) {
                    // Nur in offensichtlichen Feature-Kontexten ersetzen
                    if (line.includes("('cat'") || line.includes('categorical') || line.includes('ColumnTransformer')) {
                        lines[i] = line.replace(/LabelEncoder\(\)/g, "OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')");
                    }
                }
            }
            fixed = lines.join('\n');
        }

        return fixed;
    } catch {
        return pythonCode;
    }
}

// Python-Script ausführen (ohne LLM-Korrektur)
export async function executePythonScript(scriptPath, scriptDir, venvDir, maxRetries = 3) {
    let currentCode = await fs.readFile(scriptPath, 'utf8');
    let attempt = 0;

    while (attempt < maxRetries) {
        try {
            attempt++;
            console.log(`Python Script Ausführung - Versuch ${attempt}/${maxRetries}`);

            // Code validieren und korrigieren
            try {
                const validatedCode = await validatePythonCode(currentCode);
                if (validatedCode !== currentCode) {
                    await fs.writeFile(scriptPath, validatedCode);
                    currentCode = validatedCode;
                    console.log(`Code wurde automatisch korrigiert (Versuch ${attempt})`);
                }
            } catch (validationError) {
                handleError(validationError, 'Code-Validierung', attempt);
                // Weiter mit dem ursprünglichen Code
            }

            // Wenn Windows, dann den Pfad zum Python-Interpreter im virtuellen Environment anpassen
            let venvPath;
            if (process.platform === 'win32') {
                venvPath = path.join(venvDir, 'Scripts', 'python.exe');
            } else if (process.platform === 'linux' || process.platform === 'darwin') {
                venvPath = path.join(venvDir, 'bin', 'python');
            } else {
                throw new Error('Unsupported operating system');
            }

            const { stdout, stderr } = await execAsync(`${venvPath} "${scriptPath}"`, {
                cwd: scriptDir,
                timeout: 600000 // 10 Minuten Timeout
            });

            // Kombiniere stdout und stderr für die vollständige Ausgabe
            const fullOutput = (stdout || '') + (stderr || '');

            // Prüfe auf echte Fehler (nicht nur Warnungen)
            const hasRealError = stderr && stderr.trim() && (
                stderr.includes('Error:') ||
                stderr.includes('Exception:') ||
                stderr.includes('Traceback') ||
                stderr.includes('SyntaxError') ||
                stderr.includes('ImportError') ||
                stderr.includes('ModuleNotFoundError') ||
                stderr.includes('FileNotFoundError') ||
                stderr.includes('PermissionError') ||
                stderr.includes('exit(1)') ||
                stderr.includes('sys.exit(1)')
            );

            // Prüfe ob es echte Fehler gibt (nur deterministische Korrekturen, keine LLM)
            if (hasRealError) {
                console.log(`Echter Fehler bei Ausführung (Versuch ${attempt}):`, stderr);
                // Versuche deterministische Korrekturen
                const deterministicallyFixed = applyDeterministicFixes(currentCode, stderr);
                if (deterministicallyFixed && deterministicallyFixed !== currentCode) {
                    await fs.writeFile(scriptPath, deterministicallyFixed);
                    currentCode = deterministicallyFixed;
                    console.log(`Deterministische Korrektur angewendet (Versuch ${attempt})`);
                    continue; // erneut versuchen
                }

                if (attempt >= maxRetries) {
                    // Letzter Versuch fehlgeschlagen
                    console.log(`Alle ${maxRetries} Versuche fehlgeschlagen`);
                    return { stdout: fullOutput, stderr };
                }
            }

            // Erfolgreiche Ausführung
            return { stdout: fullOutput, stderr: '' };

        } catch (error) {
            handleError(error, 'Python-Ausführung', attempt);

            // Versuche deterministische Korrekturen auch bei geworfenen Fehlern
            const deterministicallyFixed = applyDeterministicFixes(currentCode, error.message || '');
            if (deterministicallyFixed && deterministicallyFixed !== currentCode && attempt < maxRetries) {
                await fs.writeFile(scriptPath, deterministicallyFixed);
                currentCode = deterministicallyFixed;
                console.log(`Deterministische Korrektur (Exception) angewendet (Versuch ${attempt})`);
                continue;
            }

            if (attempt >= maxRetries) {
                // Letzter Versuch fehlgeschlagen
                console.log(`Alle ${maxRetries} Versuche fehlgeschlagen`);
                throw new Error(`Python execution failed after ${maxRetries} attempts: ${error.message}`);
            }
        }
    }

    // Fallback: Gib den letzten bekannten Fehler zurück
    throw new Error(`Python execution failed after ${maxRetries} attempts`);
}

// Metriken aus Python-Ausgabe extrahieren
export function extractMetricsFromOutput(output, modelType) {
    const metrics = {};
    const foundMetrics = new Set(); // Track bereits gefundene Metriken

    // Erweiterte Metriken-Extraktionsmuster mit Priorität für ausgeschriebene Namen
    const metricPatterns = [
        {
            primaryName: 'mean_absolute_error',
            aliases: ['mae'],
            regexes: [
                /(?:MAE):\s*([\d.]+)/i
            ]
        },
        {
            primaryName: 'mean_squared_error',
            aliases: ['mse'],
            regexes: [
                /(?:MSE):\s*([\d.]+)/i
            ]
        },
        {
            primaryName: 'root_mean_squared_error',
            aliases: ['rmse'],
            regexes: [
                /(?:RMSE):\s*([\d.]+)/i
            ]
        },
        {
            primaryName: 'r_squared',
            aliases: ['r2'],
            regexes: [
                /(?:R2):\s*([\d.]+)/i
            ]
        },
        {
            primaryName: 'accuracy',
            aliases: [],
            regexes: [
                /(?:Accuracy):\s*([\d.]+)/i
            ]
        },
        {
            primaryName: 'precision',
            aliases: [],
            regexes: [
                /(?:Precision)(?:\s*\(Precision\))?:\s*([\d.]+)/i,
                /(?:Precision):\s*([\d.]+)/i
            ]
        },
        {
            primaryName: 'recall',
            aliases: [],
            regexes: [
                /(?:Recall)(?:\s*\(Recall\))?:\s*([\d.]+)/i,
                /(?:Recall):\s*([\d.]+)/i
            ]
        },
        {
            primaryName: 'f1_score',
            aliases: ['f1'],
            regexes: [
                /(?:F1 Score|F1)(?:\s*\(F1\))?:\s*([\d.]+)/i,
                /(?:F1 Score|F1):\s*([\d.]+)/i
            ]
        }
    ];

    // Extrahiere Metriken mit Priorität für ausgeschriebene Namen
    metricPatterns.forEach(metricGroup => {
        // Prüfe ob diese Metrik bereits gefunden wurde
        const allNames = [metricGroup.primaryName, ...metricGroup.aliases];
        const alreadyFound = allNames.some(name => foundMetrics.has(name));

        if (alreadyFound) {
            return; // Überspringe diese Metrik-Gruppe
        }

        for (const regex of metricGroup.regexes) {
            const match = output.match(regex);
            if (match) {
                const value = parseFloat(match[1]);

                // Validiere den Wert
                if (isNaN(value)) {
                    continue; // Überspringe ungültige Werte
                }

                // Speichere nur den primären Namen (ausgeschrieben)
                metrics[metricGroup.primaryName] = value;

                // Markiere alle Aliase als gefunden
                allNames.forEach(name => foundMetrics.add(name));

                break; // Stoppe nach dem ersten erfolgreichen Match
            }
        }
    });

    return metrics;
}

function convertInputFeatures(inputFeatures) {
    if (typeof inputFeatures === 'object' && inputFeatures !== null && !Array.isArray(inputFeatures)) {
        const firstKey = Object.keys(inputFeatures)[0];
        if (typeof inputFeatures[firstKey] === 'object' && inputFeatures[firstKey] !== null && !Array.isArray(inputFeatures[firstKey])) {
            return inputFeatures[firstKey];
        }
    }
    return inputFeatures;
}

// Prediction-Script Generator
export async function generatePredictionScript(project, inputFeatures, scriptDir, modelsDir) {

    // Überprüfe, ob bereits eine Prediction-Script vorhanden ist
    const predictionScriptPath = path.join(scriptDir, `predict_${project.id}.py`);
    if (fsSync.existsSync(predictionScriptPath)) {
        // Lese das Script
        const predictionScriptContent = await fs.readFile(predictionScriptPath, 'utf8');
        // Finde die Zeile mit input_data, lösche diese und füge die neuen Input-Features hinzu
        const updatedPredictionScript = predictionScriptContent.replace(/input_data = .+/, `input_data = ${JSON.stringify(inputFeatures)}`);
        // Speichere das Script
        // await fs.writeFile(predictionScriptPath, updatedPredictionScript);
        return { predictionScript: updatedPredictionScript };
    }

    // Absolute Model-Pfade verwenden
    // project.modelPath ist relativ (z.B. "models/model_123.pkl")
    // Extrahiere den Dateinamen und erstelle absoluten Pfad
    const modelFileName = path.basename(project.modelPath || `model_${project.id}.pkl`);
    const modelPath = path.join(modelsDir, modelFileName);

    // Absolute Pfade für Encoder und Scaler
    const encoderPath = path.join(modelsDir, `model_${project.id}_encoder.pkl`);
    const scalerPath = path.join(modelsDir, `model_${project.id}_scaler.pkl`);

    let updatedPredictionScript = predictionScript.replace('PROJECT_ID', project.id);
    updatedPredictionScript = updatedPredictionScript.replace('MODEL_PATH', modelPath);
    updatedPredictionScript = updatedPredictionScript.replace('INPUT_FEATURES', JSON.stringify(inputFeatures));
    updatedPredictionScript = updatedPredictionScript.replace('PROBLEM_TYPE', project.modelType.toLowerCase());

    // Ersetze relative Pfade durch absolute Pfade
    updatedPredictionScript = updatedPredictionScript.replace(
        /'\.\.\/models\/' \+ 'model_'\+ project_id \+ '_encoder\.pkl'/g,
        `r'${encoderPath}'`
    );
    updatedPredictionScript = updatedPredictionScript.replace(
        /'\.\.\/models\/' \+ 'model_'\+ project_id \+ '_scaler\.pkl'/g,
        `r'${scalerPath}'`
    );

    return { predictionScript: updatedPredictionScript };
}

export async function predictWithModel(project, inputFeatures, scriptDir, venvDir, modelsDir) {
    try {
        const scriptPath = path.join(scriptDir, `predict_${project.id}.py`);
        const metadataPath = path.join(scriptDir, `predict_${project.id}_metadata.json`);

        let shouldRegenerateScript = true;

        // Prüfe ob bereits ein Predict-Skript existiert
        try {
            await fs.access(scriptPath);
            await fs.access(metadataPath);

            // Lade Metadata um zu prüfen, ob das Skript noch aktuell ist
            const metadataContent = await fs.readFile(metadataPath, 'utf-8');
            const metadata = JSON.parse(metadataContent);

            // Prüfe ob das Projekt seit der letzten Skript-Generierung verändert wurde
            const currentProjectHash = generateProjectHash(project);

            if (metadata.projectHash === currentProjectHash && metadata.version === '1.0') {
                shouldRegenerateScript = false;
            }
        } catch (accessError) {
            // Script existiert nicht
        }

        // Generiere neues Skript nur wenn nötig
        if (shouldRegenerateScript) {
            const { predictionScript } = await generatePredictionScript(project, convertInputFeatures(inputFeatures), scriptDir);

            // Speichere das neue Skript
            await fs.writeFile(scriptPath, predictionScript);

            // Speichere Metadata für zukünftige Wiederverwendung
            const metadata = {
                projectId: project.id,
                projectHash: generateProjectHash(project),
                version: '1.0',
                createdAt: new Date().toISOString(),
                lastUsed: new Date().toISOString()
            };

            await fs.writeFile(metadataPath, JSON.stringify(metadata, null, 2));
        } else {
            // Aktualisiere "lastUsed" Timestamp bei Wiederverwendung
            try {
                const metadataContent = await fs.readFile(metadataPath, 'utf-8');
                const metadata = JSON.parse(metadataContent);
                metadata.lastUsed = new Date().toISOString();
                await fs.writeFile(metadataPath, JSON.stringify(metadata, null, 2));
            } catch (updateError) {
                console.log(`Warnung: Konnte Metadata nicht aktualisieren: ${updateError.message}`);
            }
        }

        // Führe das Predict-Skript aus
        const { stdout, stderr } = await executePythonScript(scriptPath, scriptDir, venvDir);

        const predictionMatch = stdout.match(/PREDICTION_RESULT: (.+)/);
        if (!predictionMatch) {
            throw new Error('Could not extract prediction result from output');
        }

        const prediction = predictionMatch[1].trim();

        return prediction;
    } catch (error) {
        console.error('Prediction error:', error);

        // Bei Fehler das potentiell defekte Skript löschen, damit es beim nächsten Mal neu generiert wird
        try {
            const scriptPath = path.join(scriptDir, `predict_${project.id}.py`);
            const metadataPath = path.join(scriptDir, `predict_${project.id}_metadata.json`);
            const targetEncoderPath = path.join(scriptDir, `target_encoder_${project.id}.pkl`);

            await fs.unlink(scriptPath);
            await fs.unlink(metadataPath);

            // Auch projekt-spezifischen Target-Encoder löschen
            try {
                await fs.unlink(targetEncoderPath);
            } catch (encoderError) {
                // Target-Encoder existiert möglicherweise nicht
            }

            console.log(`Defekte Predict-Dateien für Projekt ${project.id} gelöscht - werden beim nächsten Mal neu generiert`);
        } catch (deleteError) {
            console.log(`Predict-Dateien konnten nach Fehler nicht gelöscht werden: ${deleteError.message}`);
        }

        throw error;
    }
}

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

