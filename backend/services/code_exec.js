import fs from 'fs/promises';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';
import { callLLMAPI } from './llm.js';

import fsSync from 'fs';
const execAsync = promisify(exec);

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

// Überprüfe und validiere den Python-Code mit LLM
export async function validatePythonCodeWithLLM(pythonCode) {
  try {
    const prompt = `Du bist ein extrem erfahrener Machine Learning Engineer, Python Programmierer und Fehlerbehebungs-Experte. Überprüfe den folgenden Python-Code auf Fehler und optimiere ihn: ${pythonCode}`;
    let response = await callLLMAPI(prompt, null, 'gemini-2.5-flash-lite');

    // Stelle sicher, dass wir den Python-Code als String zurückgeben
    let codeToValidate;
    if (response && response.result) {
      codeToValidate = response.result;
    } else if (typeof response === 'string') {
      codeToValidate = response;
    } else {
      throw new Error('LLM gab keinen gültigen Python-Code zurück');
    }

    return validatePythonCode(codeToValidate);
  } catch (error) {
    handleError(error, 'validatePythonCodeWithLLM');
    throw error;
  }
}

// Überprüfe und validiere den Python-Code
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

// LLM-basierte Code-Korrektur bei Ausführungsfehlern
export async function fixPythonCodeWithLLM(pythonCode, errorMessage, attempt) {
  try {
    const prompt = `Du bist ein extrem erfahrener Machine Learning Engineer und Python Debugging-Experte. Der folgende Python-Code hat einen Fehler bei der Ausführung. Analysiere den Fehler und korrigiere den Code entsprechend.

FEHLER (Versuch ${attempt}):
${errorMessage}

VERWENDETER PYTHON-CODE:
${pythonCode}

KORRIGIERTER CODE (gib nur den korrigierten Code zurück, ohne Erklärungen):`;

    console.log(`LLM-Korrektur - Versuch ${attempt}:`, errorMessage.substring(0, 100) + '...');
    
    const response = await callLLMAPI(prompt, null, 'gemini-2.5-flash-lite');
    
    // Stelle sicher, dass wir den Python-Code als String zurückgeben
    let correctedCode;
    if (response && response.result) {
      correctedCode = response.result;
    } else if (typeof response === 'string') {
      correctedCode = response;
    } else {
      throw new Error('LLM gab keinen gültigen Python-Code zurück');
    }
    
    correctedCode = correctedCode.trim();
    
    // Prüfe ob der korrigierte Code tatsächlich anders ist
    if (correctedCode && correctedCode !== pythonCode) {
      console.log(`LLM-Korrektur erfolgreich (Versuch ${attempt})`);
      return correctedCode;
    } else {
      console.log(`LLM-Korrektur fehlgeschlagen (Versuch ${attempt}) - Keine Änderungen`);
      return null;
    }
  } catch (error) {
    handleError(error, 'fixPythonCodeWithLLM', attempt);
    return null; // Fallback: Keine Korrektur
  }
}

// Python-Script ausführen
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
      
      console.log(`Operating System: ${process.platform}, venvPath: ${venvPath}`);
      
      const { stdout, stderr } = await execAsync(`${venvPath} "${scriptPath}"`, {
        cwd: scriptDir,
        timeout: 300000 // 5 Minuten Timeout
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
      
      // Prüfe ob es echte Fehler gibt (nicht nur Warnungen)
      if (hasRealError) {
        console.log(`Echter Fehler bei Ausführung (Versuch ${attempt}):`, stderr);
        
        if (attempt < maxRetries) {
          // Versuche Code mit LLM zu korrigieren
          console.log(`Versuche Code-Korrektur mit LLM (Versuch ${attempt})...`);
          const correctedCode = await fixPythonCodeWithLLM(currentCode, stderr, attempt);
          
          if (correctedCode && correctedCode !== currentCode) {
            await fs.writeFile(scriptPath, correctedCode);
            currentCode = correctedCode;
            console.log(`Code wurde mit LLM korrigiert (Versuch ${attempt})`);
            continue; // Nächster Versuch
          } else {
            console.log(`LLM konnte Code nicht korrigieren (Versuch ${attempt})`);
            break; // Keine weitere Korrektur möglich
          }
        } else {
          // Letzter Versuch fehlgeschlagen
          console.log(`Alle ${maxRetries} Versuche fehlgeschlagen`);
          return { stdout: fullOutput, stderr };
        }
      }
      
      // Erfolgreiche Ausführung - verwende die kombinierte Ausgabe
      console.log(`Python Script erfolgreich ausgeführt (Versuch ${attempt})`);
      console.log(`Vollständige Ausgabe:`, fullOutput.substring(0, 200) + '...');
      return { stdout: fullOutput, stderr: '' };
      
    } catch (error) {
      handleError(error, 'Python-Ausführung', attempt);
      
      if (attempt < maxRetries) {
        // Versuche Code mit LLM zu korrigieren
        console.log(`Versuche Code-Korrektur mit LLM (Versuch ${attempt})...`);
        const correctedCode = await fixPythonCodeWithLLM(currentCode, error.message, attempt);
        
        if (correctedCode && correctedCode !== currentCode) {
          await fs.writeFile(scriptPath, correctedCode);
          currentCode = correctedCode;
          console.log(`Code wurde mit LLM korrigiert (Versuch ${attempt})`);
          continue; // Nächster Versuch
        } else {
          console.log(`LLM konnte Code nicht korrigieren (Versuch ${attempt})`);
          break; // Keine weitere Korrektur möglich
        }
      } else {
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
        /(?:Mean Absolute Error|MAE)(?:\s*\(MAE\))?:\s*([\d.]+)/i,
        /(?:Mean Absolute Error|MAE):\s*([\d.]+)/i,
        /(?:MAE)(?:\s*\(MAE\))?:\s*([\d.]+)/i,
        /(?:MAE):\s*([\d.]+)/i
      ]
    },
    { 
      primaryName: 'mean_squared_error',
      aliases: ['mse'],
      regexes: [
        /(?:Mean Squared Error|MSE)(?:\s*\(MSE\))?:\s*([\d.]+)/i,
        /(?:Mean Squared Error|MSE):\s*([\d.]+)/i,
        /(?:MSE)(?:\s*\(MSE\))?:\s*([\d.]+)/i,
        /(?:MSE):\s*([\d.]+)/i
      ]
    },
    { 
      primaryName: 'root_mean_squared_error',
      aliases: ['rmse'],
      regexes: [
        /(?:Root Mean Squared Error|RMSE)(?:\s*\(RMSE\))?:\s*([\d.]+)/i,
        /(?:Root Mean Squared Error|RMSE):\s*([\d.]+)/i,
        /(?:RMSE)(?:\s*\(RMSE\))?:\s*([\d.]+)/i,
        /(?:RMSE):\s*([\d.]+)/i
      ]
    },
    { 
      primaryName: 'r_squared',
      aliases: ['r2'],
      regexes: [
        /(?:R-squared|R2|R²)(?:\s*\(R2\))?:\s*([\d.]+)/i,
        /(?:R-squared|R2|R²):\s*([\d.]+)/i,
        /(?:R²)(?:\s*\(R2\))?:\s*([\d.]+)/i,
        /(?:R²):\s*([\d.]+)/i
      ]
    },
    { 
      primaryName: 'accuracy',
      aliases: [],
      regexes: [
        /^Accuracy:\s*([\d.]+)/im,
        /(?:^|\n)Accuracy:\s*([\d.]+)/i,
        /(?:Accuracy)(?:\s*\(Accuracy\))?:\s*([\d.]+)/i,
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
export async function generatePredictionScript(project, inputFeatures, scriptDir) {

  // Überprüfe, ob bereits eine Prediction-Script vorhanden ist
  const predictionScriptPath = path.join(scriptDir, `predict_${project.id}.py`);
  if (fsSync.existsSync(predictionScriptPath)) {
    // Lese das Script
    const predictionScript = await fs.readFile(predictionScriptPath, 'utf8');
    // Finde die Zeile mit input_data, lösche diese und füge die neuen Input-Features hinzu
    const updatedPredictionScript = predictionScript.replace(/input_data = .+/, `input_data = ${JSON.stringify(inputFeatures)}`);
    // Speichere das Script
    // await fs.writeFile(predictionScriptPath, updatedPredictionScript);
    return { predictionScript: updatedPredictionScript };
  }

  // Wenn scriptDir auf scripts endet, dann eine Ebene höher springen
  if (scriptDir.endsWith('scripts')) {
    scriptDir = path.join(scriptDir, '..');
  }
  const modelPath = path.join(scriptDir, project.modelPath);

  const predictionScript = `
import pandas as pd
import numpy as np
import joblib
import json

# Input-Features verarbeiten
input_data = ${JSON.stringify(inputFeatures)}
print(f"Input-Features: {input_data}")

# DataFrame erstellen - konsistent mit Training
input_df = pd.DataFrame([input_data])
print(f"Input DataFrame Shape: {input_df.shape}")

# Direkt numpy array verwenden (konsistent mit Training)
input_array = input_df.values
print(f"Input Array Shape: {input_array.shape}")

# Model laden
try:
    model = joblib.load(r"${modelPath}")
    print("Model erfolgreich geladen")
    
    # Prüfe ob es eine Pipeline ist oder ein direktes Modell
    if hasattr(model, 'named_steps'):
        print("Pipeline-Modell erkannt")
        # Für Pipeline: Input muss durch die Pipeline gehen
        prediction = model.predict(input_array)[0]
    else:
        print("Direktes Modell erkannt")
        # Für direktes Modell: Input muss skaliert werden falls nötig
        try:
            # Versuche Scaler zu laden
            scaler = joblib.load('scaler.pkl')
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
        except Exception as scaler_error:
            print(f"Scaler nicht gefunden oder Fehler: {scaler_error}")
            # Fallback: Direkte Prediction ohne Skalierung
            # Aber zuerst Feature-Namen entfernen falls vorhanden
            if hasattr(input_array, 'columns'):
                input_array = input_array.values
            prediction = model.predict(input_array)[0]
    
    print(f"Raw Prediction: {prediction}")
    
    # Falls Label-Encoder existiert (für Klassifikation)
    try:
        target_encoder = joblib.load('target_encoder.pkl')
        prediction = target_encoder.inverse_transform([int(prediction)])[0]
        print(f"Decoded Prediction: {prediction}")
    except:
        pass
    
    # Ergebnis ausgeben (wird vom Node.js-Server geparst)
    print(f"PREDICTION_RESULT: {prediction}")
    
except Exception as e:
    print(f"Prediction error: {e}")
    exit(1)
`.trim();

  return { predictionScript };
}

export async function predictWithModel(project, inputFeatures, scriptDir, venvDir) {
  try {
    const { predictionScript } = await generatePredictionScript(project, convertInputFeatures(inputFeatures), scriptDir);
    const scriptPath = path.join(scriptDir, `predict_${project.id}.py`);
    
    await fs.writeFile(scriptPath, predictionScript);
    
    const { stdout, stderr } = await executePythonScript(scriptPath, scriptDir, venvDir);

    const predictionMatch = stdout.match(/PREDICTION_RESULT: (.+)/);
    if (!predictionMatch) {
      // Lösche das Script nach der Ausführung
      try {
        await fs.unlink(scriptPath);
      } catch (err) {
        console.error('Could not delete script file:', err.message);
      }
      throw new Error('Could not extract prediction result from output');
    }

    const prediction = predictionMatch[1].trim();
    //console.log(prediction);
    return prediction;
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
}