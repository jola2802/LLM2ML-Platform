import fs from 'fs/promises';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';
import { callLLMAPI } from './llm.js';

import fsSync from 'fs';
const execAsync = promisify(exec);

// Überprüfe und validiere den Python-Code mit LLM
export async function validatePythonCodeWithLLM(pythonCode) {
  const prompt = `Du bist ein extrem erfahrener Machine Learning Engineer, Python Programmierer und Fehlerbehebungs-Experte. Überprüfe den folgenden Python-Code auf Fehler und optimiere ihn: ${pythonCode}`;
  let response = await callLLMAPI(prompt);

  return validatePythonCode(response);  
}

// Überprüfe und validiere den Python-Code
export async function validatePythonCode(pythonCode) {
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
}

// Python-Script ausführen
export async function executePythonScript(scriptPath, scriptDir, venvDir) {
  try {
    const pythonCode = await fs.readFile(scriptPath, 'utf8');
    const validatedCode = await validatePythonCode(pythonCode);
    if (validatedCode !== pythonCode) {
      await fs.writeFile(scriptPath, validatedCode);
    }
    const venvPath = path.join( venvDir, 'Scripts', 'python.exe');
    // console.log(venvPath);
    const { stdout, stderr } = await execAsync(`${venvPath} "${scriptPath}"`, {
      cwd: scriptDir
    });
    
    return { stdout, stderr };
  } catch (error) {
    throw new Error(`Python execution failed: ${error.message}`);
  }
}

// Metriken aus Python-Ausgabe extrahieren
export function extractMetricsFromOutput(output, modelType) {
  const metrics = {};
  
  // Erweiterte Metriken-Extraktionsmuster mit flexibleren Regex
  const metricPatterns = [
    { 
      names: ['mae', 'mean_absolute_error'], 
      regexes: [
        /(?:Mean Absolute Error|MAE)(?:\s*\(MAE\))?:\s*([\d.]+)/i,
        /(?:Mean Absolute Error|MAE):\s*([\d.]+)/i
      ]
    },
    { 
      names: ['mse', 'mean_squared_error'], 
      regexes: [
        /(?:Mean Squared Error|MSE)(?:\s*\(MSE\))?:\s*([\d.]+)/i,
        /(?:Mean Squared Error|MSE):\s*([\d.]+)/i
      ]
    },
    { 
      names: ['rmse', 'root_mean_squared_error'], 
      regexes: [
        /(?:Root Mean Squared Error|RMSE)(?:\s*\(RMSE\))?:\s*([\d.]+)/i,
        /(?:Root Mean Squared Error|RMSE):\s*([\d.]+)/i
      ]
    },
    { 
      names: ['r2', 'r_squared'], 
      regexes: [
        /(?:R-squared|R2|R²)(?:\s*\(R2\))?:\s*([\d.]+)/i,
        /(?:R-squared|R2|R²):\s*([\d.]+)/i
      ]
    },
    { 
      names: ['accuracy'], 
      regexes: [
        /^Accuracy:\s*([\d.]+)/im,
        /(?:^|\n)Accuracy:\s*([\d.]+)/i,
        /(?:Accuracy)(?:\s*\(Accuracy\))?:\s*([\d.]+)/i,
        /(?:Accuracy):\s*([\d.]+)/i
      ]
    },
    { 
      names: ['precision'], 
      regexes: [
        /(?:Precision)(?:\s*\(Precision\))?:\s*([\d.]+)/i,
        /(?:Precision):\s*([\d.]+)/i
      ]
    },
    { 
      names: ['recall'], 
      regexes: [
        /(?:Recall)(?:\s*\(Recall\))?:\s*([\d.]+)/i,
        /(?:Recall):\s*([\d.]+)/i
      ]
    },
    { 
      names: ['f1', 'f1_score'], 
      regexes: [
        /(?:F1 Score|F1)(?:\s*\(F1\))?:\s*([\d.]+)/i,
        /(?:F1 Score|F1):\s*([\d.]+)/i
      ]
    }
  ];

  // Extrahiere Metriken mit flexiblen Mustern
  metricPatterns.forEach(metricGroup => {
    for (const regex of metricGroup.regexes) {
      const match = output.match(regex);
      if (match) {
        // Speichere den Wert für jeden möglichen Namen
        metricGroup.names.forEach(name => {
          metrics[name] = parseFloat(match[1]);
        });
        break; // Stoppe nach dem ersten erfolgreichen Match
      }
    }
  });
  
  return metrics;
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

# Model laden
try:
    model = joblib.load(r"${modelPath}")
    print("Model erfolgreich geladen")
except Exception as e:
    print(f"Fehler beim Laden des Models: {e}")
    exit(1)

# Input-Features verarbeiten
input_data = ${JSON.stringify(inputFeatures)}
print(f"Input-Features: {input_data}")

# DataFrame erstellen
input_df = pd.DataFrame([input_data])
print(f"Input DataFrame Shape: {input_df.shape}")
print(f"Input DataFrame Columns: {list(input_df.columns)}")

# Prediction
try:
    prediction = model.predict(input_df)[0]
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
    const { features, ...rest } = inputFeatures;
    const { predictionScript } = await generatePredictionScript(project, features, scriptDir);
    const scriptPath = path.join(scriptDir, `predict_${project.id}.py`);
    
    await fs.writeFile(scriptPath, predictionScript);
    
    const { stdout, stderr } = await executePythonScript(scriptPath, scriptDir, venvDir);
    if (stderr) {
      throw new Error(stderr);
      // Lösche das Script nach der Ausführung
      try {
        await fs.unlink(scriptPath);
      } catch (err) {
        console.error('Could not delete script file:', err.message);
      }
    }

    const predictionMatch = stdout.match(/PREDICTION_RESULT: (.+)/);
    if (!predictionMatch) {
      throw new Error('Could not extract prediction result');
    }

    const prediction = predictionMatch[1].trim();
    //console.log(prediction);
    return prediction;
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
}