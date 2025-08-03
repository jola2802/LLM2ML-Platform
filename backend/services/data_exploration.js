import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const venvDir = path.join(__dirname, 'python', 'venv');

/**
 * Führt automatische Datenexploration durch und erstellt eine strukturierte Übersicht
 * @param {string} filePath - Pfad zur zu analysierenden Datei
 * @returns {Promise<Object>} - Strukturierte Datenübersicht
 */
export async function performDataExploration(filePath) {
  return new Promise(async (resolve, reject) => {
    const pythonScript = path.join(__dirname, 'data_exploration.py');
    
    try {
      // Lese die ursprüngliche Python-Datei
      const originalContent = await fs.readFile(pythonScript, 'utf8');
      
      // Erstelle eine temporäre Kopie mit dem eingefügten Dateipfad
      const tempScriptPath = path.join(__dirname, `data_exploration_temp_${Date.now()}.py`);
      const updatedContent = originalContent.replace('{file_path}', filePath);
      
      // Schreibe die temporäre Datei
      await fs.writeFile(tempScriptPath, updatedContent);
      
      // Bestimme den korrekten Python-Pfad basierend auf venvDir
      let pythonExecutable = 'python';
      if (venvDir) {
        // Windows
        if (process.platform === 'win32') {
          pythonExecutable = path.join(venvDir, 'Scripts', 'python.exe');
        } else {
          // Unix/Linux/Mac
          pythonExecutable = path.join(venvDir, 'bin', 'python');
        }
      }
      
      // Führe die temporäre Python-Datei aus
      const pythonProcess = spawn(pythonExecutable, [tempScriptPath]);
      
      let stdout = '';
      let stderr = '';
      
      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      pythonProcess.on('close', async (code) => {
        // Lösche die temporäre Datei
        try {
          await fs.unlink(tempScriptPath);
        } catch (cleanupError) {
          console.warn('Konnte temporäre Datei nicht löschen:', cleanupError.message);
        }
        
        if (code === 0) {
          try {
            const result = JSON.parse(stdout);
            resolve(result);
          } catch (parseError) {
            console.error('Fehler beim Parsen der Python-Ausgabe:', parseError);
            console.error('Raw stdout:', stdout);
            reject(new Error('Ungültige Ausgabe von der Datenexploration'));
          }
        } else {
          console.error('Python-Script Fehler:', stderr);
          console.error('Exit code:', code);
          reject(new Error(`Datenexploration fehlgeschlagen: ${stderr}`));
        }
      });
      
      pythonProcess.on('error', async (error) => {
        // Lösche die temporäre Datei auch bei Fehlern
        try {
          await fs.unlink(tempScriptPath);
        } catch (cleanupError) {
          console.warn('Konnte temporäre Datei nicht löschen:', cleanupError.message);
        }
        
        console.error('Fehler beim Starten der Datenexploration:', error);
        reject(new Error(`Konnte Datenexploration nicht starten: ${error.message}`));
      });
      
    } catch (error) {
      console.error('Fehler beim Erstellen der temporären Python-Datei:', error);
      reject(new Error(`Fehler beim Vorbereiten der Datenexploration: ${error.message}`));
    }
  });
}

/**
 * Erstellt eine LLM-optimierte Zusammenfassung der Datenexploration
 * @param {Object} explorationResult - Ergebnis der Datenexploration
 * @returns {string} - Formatierte Zusammenfassung für das LLM
 */
export function createLLMSummary(explorationResult) {
  if (explorationResult.error) {
    return `Fehler bei der Datenanalyse: ${explorationResult.error}`;
  }
  
  const { dataset_info, columns, detailed_analysis, missing_values, correlations, outliers, sample_data } = explorationResult.full_analysis;
  
  let summary = `# AUTOMATISCHE DATENANALYSE

## DATASET-ÜBERSICHT
- **Zeilen**: ${dataset_info.rows}
- **Spalten**: ${dataset_info.columns}
- **Speicherverbrauch**: ${dataset_info.memory_usage_mb} MB
- **Duplikate**: ${dataset_info.duplicate_rows} (${dataset_info.duplicate_percentage.toFixed(2)}%)

## SPALTEN-ANALYSE
`;

  // Spalten-Details
  columns.forEach(column => {
    const analysis = detailed_analysis[column];
    if (!analysis) return;
    
    summary += `\n### ${column}
- **Typ**: ${analysis.dtype}
- **Eindeutige Werte**: ${analysis.unique_count}
- **Fehlende Werte**: ${analysis.missing_count} (${analysis.missing_percentage.toFixed(2)}%)
`;

    if (analysis.is_numeric) {
      summary += `- **Bereich**: ${analysis.min} bis ${analysis.max}
- **Mittelwert**: ${analysis.mean?.toFixed(4)}
- **Median**: ${analysis.median?.toFixed(4)}
- **Standardabweichung**: ${analysis.std?.toFixed(4)}
- **Schiefe**: ${analysis.skewness?.toFixed(4)}
`;
    } else if (analysis.top_values) {
      summary += `- **Top-Werte**: ${Object.entries(analysis.top_values).slice(0, 5).map(([k, v]) => `${k}(${v})`).join(', ')}
`;
    }
  });

  // Fehlende Werte
  if (missing_values.total_missing_cells > 0) {
    summary += `\n## FEHLENDE WERTE
- **Gesamt fehlende Zellen**: ${missing_values.total_missing_cells} (${missing_values.total_missing_percentage.toFixed(2)}%)
- **Spalten mit fehlenden Werten**: ${Object.keys(missing_values.columns_with_missing).join(', ')}
`;
  }

  // Korrelationen
  if (correlations.strong_correlations && correlations.strong_correlations.length > 0) {
    summary += `\n## STARKE KORRELATIONEN
`;
    correlations.strong_correlations.forEach(corr => {
      summary += `- ${corr.column1} ↔ ${corr.column2}: ${corr.correlation.toFixed(3)}
`;
    });
  }

  // Ausreißer
  if (Object.keys(outliers).length > 0) {
    summary += `\n## AUSREIßER
`;
    Object.entries(outliers).forEach(([column, info]) => {
      summary += `- ${column}: ${info.count} Ausreißer (${info.percentage.toFixed(2)}%)
`;
    });
  }

  // Beispieldaten
  if (sample_data && sample_data.length > 0) {
    summary += `\n## BEISPIEL-DATEN
`;
    sample_data.slice(0, 5).forEach((row, index) => {
      summary += `\n**Zeile ${row.row_index}**: ${Object.entries(row.data).map(([k, v]) => `${k}=${v}`).join(', ')}
`;
    });
  }

  return summary;
}

/**
 * Kombinierte Funktion: Führt Datenexploration durch und erstellt LLM-Zusammenfassung
 * @param {string} filePath - Pfad zur Datei
 * @returns {Promise<Object>} - Vollständige Analyse mit LLM-Zusammenfassung
 */
export async function analyzeDataForLLM(filePath) {
  try {    
    // Führe Datenexploration durch
    const explorationResult = await performDataExploration(filePath);
    
    // Erstelle LLM-Zusammenfassung
    const llmSummary = createLLMSummary(explorationResult);
    
    return {
      success: true,
      exploration: explorationResult,
      llm_summary: llmSummary,
      analysis_timestamp: new Date().toISOString()
    };
    
  } catch (error) {
    console.error('Fehler bei der automatischen Datenanalyse:', error);
    return {
      success: false,
      error: error.message,
      file_path: filePath,
      analysis_timestamp: new Date().toISOString()
    };
  }
}

/**
 * Cache für Datenanalysen (verhindert wiederholte Analysen)
 */
const analysisCache = new Map();

/**
 * Analysiert Daten mit Cache-Unterstützung
 * @param {string} filePath - Pfad zur Datei
 * @param {boolean} forceRefresh - Cache umgehen
 * @returns {Promise<Object>} - Analyseergebnis
 */
export async function getCachedDataAnalysis(filePath, forceRefresh = false) {
  // Prüfe Cache
  if (!forceRefresh && analysisCache.has(filePath)) {
    return analysisCache.get(filePath);
  }
  
  // Führe neue Analyse durch
  const result = await analyzeDataForLLM(filePath);
  
  // Speichere im Cache
  if (result.success) {
    analysisCache.set(filePath, result);
  }
  
  return result;
}

/**
 * Cache leeren
 */
export function clearAnalysisCache() {
  analysisCache.clear();
  console.log('Datenanalyse-Cache geleert');
}

/**
 * Cache-Status abrufen
 */
export function getAnalysisCacheStatus() {
  return {
    cached_analyses: Array.from(analysisCache.keys()),
    cache_size: analysisCache.size
  };
} 