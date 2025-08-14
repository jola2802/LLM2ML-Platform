import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// Korrekte Pfad zur Python Virtual Environment (services/python/venv/)
const venvDir = path.join(__dirname, '..', 'python', 'venv');

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
  
  // Nutze die kompakte, Pareto-optimierte Übersicht als Primärquelle
  const pareto = explorationResult.pareto_analysis;
  const full = explorationResult.full_analysis;
  
  if (!pareto || pareto.error) {
    // Fallback auf sehr kompakte Full-Analyse-Basisinfos
    const ds = full?.dataset_info || { rows: 'unbekannt', columns: 'unbekannt', memory_usage_mb: 'unbekannt', duplicate_rows: 0, duplicate_percentage: 0 };
    return [
      `AUTOMATISCHE DATENANALYSE (kompakt)`,
      `- Zeilen: ${ds.rows}`,
      `- Spalten: ${ds.columns}`,
      `- Speicher (MB): ${ds.memory_usage_mb}`,
      `- Duplikate: ${ds.duplicate_rows} (${Number(ds.duplicate_percentage).toFixed ? Number(ds.duplicate_percentage).toFixed(2) : ds.duplicate_percentage}%)`
    ].join('\n');
  }
  
  const lines = [];
  const s = pareto.summary || {};
  lines.push('AUTOMATISCHE DATENANALYSE (Pareto-kurzfassung)');
  lines.push(`- Zeilen: ${s.rows}`);
  lines.push(`- Spalten: ${s.columns}`);
  lines.push(`- Datei (MB): ${s.file_size_mb}`);
  lines.push(`- Speicher (MB): ${s.memory_mb}`);
  
  // Wichtigste Spalten kurz darstellen
  const keyCols = pareto.key_columns || {};
  const keyEntries = Object.entries(keyCols).slice(0, 5);
  if (keyEntries.length > 0) {
    lines.push('WICHTIGE SPALTEN:');
    keyEntries.forEach(([col, info]) => {
      const parts = [`${col} (dtype=${info.dtype}`];
      if (typeof info.missing_pct === 'number') parts.push(`missing=${info.missing_pct}%`);
      if (typeof info.unique_count === 'number') parts.push(`unique=${info.unique_count}`);
      if (typeof info.mean === 'number') parts.push(`mean=${info.mean}`);
      if (typeof info.std === 'number') parts.push(`std=${info.std}`);
      if (typeof info.min === 'number' && typeof info.max === 'number') parts.push(`range=${info.min}..${info.max}`);
      if (typeof info.outliers_pct === 'number') parts.push(`outliers=${info.outliers_pct}%`);
      if (info.top_values) parts.push(`top=${Object.entries(info.top_values).map(([k,v])=>`${k}(${v})`).join('|')}`);
      parts.push(')');
      lines.push(`- ${parts.join(', ')}`);
    });
  }
  
  // Starke Korrelationen (max 5)
  const corrs = pareto.strong_correlations || [];
  if (corrs.length > 0) {
    lines.push('STARKE KORRELATIONEN:');
    corrs.forEach(c => lines.push(`- ${c.columns?.join(' ↔ ')}: ${c.correlation}`));
  }
  
  // Datenqualitätsprobleme kurz nennen
  const issues = pareto.data_quality_issues || [];
  if (issues.length > 0) {
    lines.push('DATENQUALITÄT (kritisch):');
    issues.forEach(i => {
      if (i.type === 'high_missing_values') {
        lines.push(`- fehlende Werte >10% in: ${Array.isArray(i.columns) ? i.columns.join(', ') : ''} (gesamt≈${i.missing_pct}%)`);
      } else if (i.type === 'duplicates') {
        lines.push(`- Duplikate ≈${i.percentage}%`);
      } else if (i.type === 'constant_columns') {
        lines.push(`- konstante Spalten: ${Array.isArray(i.columns) ? i.columns.join(', ') : ''}`);
      }
    });
  }
  
  // Sehr kompakte Beispieldaten (bis 3 Zeilen)
  const samples = pareto.sample_data || [];
  if (samples.length > 0) {
    lines.push('BEISPIEL-DATEN (kompakt):');
    samples.slice(0, 3).forEach(r => {
      const kv = r.data ? Object.entries(r.data).slice(0, 8).map(([k, v]) => `${k}=${v}`).join(', ') : '';
      lines.push(`- ${r.position}: ${kv}`);
    });
  }
  
  return lines.join('\n');
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