import fs from 'fs/promises';
import { callLLMAPI } from '../llm/llm.js';
import { getCachedDataAnalysis } from './data_exploration.js';

const analysis_prompt = `Du bist ein extrem erfahrener Data Scientist. Analysiere die folgende automatische Datenübersicht und gib eine detaillierte Analyse der Daten zurück.

AUTOMATISCHE DATENÜBERSICHT:
{data_overview}

AUFGABE: Basierend auf der automatischen Datenanalyse, gib eine professionelle Einschätzung der Datenqualität, möglichen Herausforderungen und Empfehlungen für das Machine Learning zurück.

Fokussiere dich auf:
1. Datenqualität und -reinheit
2. Identifizierte Probleme (fehlende Werte, Ausreißer, etc.)
3. Beziehungen zwischen Variablen
4. Empfehlungen für Preprocessing
5. Potentielle ML-Anwendungsfälle

Antworte in einem strukturierten, professionellen Format.`;

// CSV-Datei-Analysefunktion
export async function analyzeCsvFile(filePath, withLLMAnalysis = true) {
  try {
    const csvContent = await fs.readFile(filePath, 'utf-8');
    const lines = csvContent.split('\n').filter(line => line.trim());
    
    if (lines.length < 2) {
      throw new Error('CSV-Datei muss mindestens einen Header und eine Datenzeile enthalten');
    }
    
    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
    const sampleRows = lines.slice(1, Math.min(6, lines.length)).map(line => 
      line.split(',').map(cell => cell.trim().replace(/"/g, ''))
    );
    
    // Datentypen erkennen
    const dataTypes = {};
    headers.forEach((header, index) => {
      const sampleValues = sampleRows.map(row => row[index]).filter(v => v && v !== '');
      if (sampleValues.length > 0) {
        const isNumeric = sampleValues.every(v => !isNaN(parseFloat(v)) && isFinite(v));
        dataTypes[header] = isNumeric ? 'numeric' : 'categorical';
      } else {
        dataTypes[header] = 'unknown';
      }
    });

    // Automatische Datenexploration durchführen
    const dataAnalysis = await getCachedDataAnalysis(filePath).catch(() => ({ success: false }));
    
    let llmAnalysis = null;
    
    // LLM-basierte Analyse nur wenn gewünscht und automatische Analyse erfolgreich
    if (withLLMAnalysis && dataAnalysis && dataAnalysis.success) {
      const prompt = analysis_prompt.replace('{data_overview}', dataAnalysis.llm_summary || 'Keine automatische Analyse verfügbar');
      llmAnalysis = await callLLMAPI(prompt).catch(() => null);
    }
    
    return {
      columns: headers,
      rowCount: lines.length - 1,
      dataTypes,
      sampleData: sampleRows.slice(0, 5),
      llm_analysis: llmAnalysis || null,
      automatic_analysis: dataAnalysis && dataAnalysis.success ? dataAnalysis.exploration : null,
      analysis_summary: dataAnalysis && dataAnalysis.success ? dataAnalysis.llm_summary : null
    };
    
  } catch (error) {
    console.error('Fehler bei der CSV-Analyse:', error);
    throw error;
  }
}

// JSON-Datei analysieren
export async function analyzeJsonFile(filePath, withLLMAnalysis = true) {
  try {
    const jsonContent = await fs.readFile(filePath, 'utf-8');
    const data = JSON.parse(jsonContent);
    
    let columns = [];
    let rowCount = 0;
    let sampleData = [];
    
    // Verschiedene JSON-Strukturen behandeln
    if (Array.isArray(data)) {
      // Array von Objekten
      if (data.length > 0) {
        columns = Object.keys(data[0]);
        rowCount = data.length;
        sampleData = data.slice(0, 5).map(item => Object.values(item));
      }
    } else if (typeof data === 'object') {
      // Einzelnes Objekt oder verschachtelte Struktur
      columns = Object.keys(data);
      rowCount = 1;
      sampleData = [Object.values(data)];
    }
    
    // Datentypen erkennen
    const dataTypes = {};
    columns.forEach((column, index) => {
      const sampleValues = sampleData.map(row => row[index]).filter(v => v !== undefined);
      if (sampleValues.length > 0) {
        const isNumeric = sampleValues.every(v => !isNaN(parseFloat(v)) && isFinite(v));
        dataTypes[column] = isNumeric ? 'numeric' : 'categorical';
      } else {
        dataTypes[column] = 'unknown';
      }
    });

    // Automatische Datenexploration durchführen
    const dataAnalysis = await getCachedDataAnalysis(filePath).catch(() => ({ success: false }));
    
    // LLM-basierte Analyse
    let llm_analysis = null;
    if (withLLMAnalysis) {
      const prompt = analysis_prompt.replace('{data_overview}', dataAnalysis && dataAnalysis.success ? dataAnalysis.llm_summary : 'Keine automatische Analyse verfügbar');
      llm_analysis = await callLLMAPI(prompt).catch(() => null);
    }
    
    return {
      columns,
      rowCount,
      dataTypes,
      sampleData,
      llm_analysis,
      automatic_analysis: dataAnalysis && dataAnalysis.success ? dataAnalysis.exploration : null,
      analysis_summary: dataAnalysis && dataAnalysis.success ? dataAnalysis.llm_summary : null
    };
  } catch (error) {
    throw new Error(`Fehler beim Analysieren der JSON-Datei: ${error.message}`);
  }
}

// Excel-Datei analysieren
export async function analyzeExcelFile(filePath, withLLMAnalysis = true) {
  try {
    // Automatische Datenexploration durchführen
    const dataAnalysis = await getCachedDataAnalysis(filePath).catch(() => ({ success: false }));
    
    let llm_analysis = null;
    if (withLLMAnalysis) {
      const prompt = analysis_prompt.replace('{data_overview}', dataAnalysis && dataAnalysis.success ? dataAnalysis.llm_summary : 'Keine automatische Analyse verfügbar');
      llm_analysis = await callLLMAPI(prompt).catch(() => null);
    }
    
    return {
      columns: dataAnalysis && dataAnalysis.success ? (dataAnalysis.exploration?.columns || []) : ['Excel-Spalten werden durch LLM analysiert'],
      rowCount: dataAnalysis && dataAnalysis.success ? (dataAnalysis.exploration?.dataset_info?.rows ?? 0) : 0,
      dataTypes: dataAnalysis && dataAnalysis.success ? (dataAnalysis.exploration?.data_types || {}) : {},
      sampleData: dataAnalysis && dataAnalysis.success ? (Array.isArray(dataAnalysis.exploration?.sample_data) ? dataAnalysis.exploration.sample_data.map(row => Object.values(row.data || row)) : []) : [],
      llm_analysis,
      automatic_analysis: dataAnalysis && dataAnalysis.success ? dataAnalysis.exploration : null,
      analysis_summary: dataAnalysis && dataAnalysis.success ? dataAnalysis.llm_summary : null
    };
  } catch (error) {
    throw new Error(`Fehler beim Analysieren der Excel-Datei: ${error.message}`);
  }
}

// Text-Datei analysieren
export async function analyzeTextFile(filePath, withLLMAnalysis = true) {
  try {
    const textContent = await fs.readFile(filePath, 'utf-8');
    const lines = textContent.split('\n').filter(line => line.trim());
    
    // Automatische Datenexploration durchführen
    const dataAnalysis = await getCachedDataAnalysis(filePath).catch(() => ({ success: false }));
    
    let llm_analysis = null;
    if (withLLMAnalysis) {
      const prompt = analysis_prompt.replace('{data_overview}', dataAnalysis && dataAnalysis.success ? dataAnalysis.llm_summary : 'Keine automatische Analyse verfügbar');
      llm_analysis = await callLLMAPI(prompt).catch(() => null);
    }
    
    return {
      columns: ['Text-Inhalt'],
      rowCount: lines.length,
      dataTypes: { 'Text-Inhalt': 'text' },
      sampleData: lines.slice(0, 5).map(line => [line]),
      llm_analysis,
      automatic_analysis: dataAnalysis.success ? dataAnalysis.exploration : null,
      analysis_summary: dataAnalysis.success ? dataAnalysis.llm_summary : null
    };
  } catch (error) {
    throw new Error(`Fehler beim Analysieren der Text-Datei: ${error.message}`);
  }
}

// Generische Datei analysieren
export async function analyzeGenericFile(filePath, fileExtension, withLLMAnalysis = true) {
  try {
    // Automatische Datenexploration durchführen
    const dataAnalysis = await getCachedDataAnalysis(filePath);
    
    let llm_analysis = null;
    if (withLLMAnalysis) {
      const prompt = analysis_prompt.replace('{data_overview}', dataAnalysis.success ? dataAnalysis.llm_summary : 'Keine automatische Analyse verfügbar');
      llm_analysis = await callLLMAPI(prompt);
    }
    
    return {
      columns: dataAnalysis && dataAnalysis.success ? (dataAnalysis.exploration?.columns || []) : [`${fileExtension.substring(1).toUpperCase()}-Inhalt`],
      rowCount: dataAnalysis && dataAnalysis.success ? (dataAnalysis.exploration?.dataset_info?.rows ?? 0) : 0,
      dataTypes: dataAnalysis && dataAnalysis.success ? (dataAnalysis.exploration?.data_types || {}) : { [`${fileExtension.substring(1).toUpperCase()}-Inhalt`]: 'mixed' },
      sampleData: dataAnalysis && dataAnalysis.success ? (Array.isArray(dataAnalysis.exploration?.sample_data) ? dataAnalysis.exploration.sample_data.map(row => Object.values(row.data || row)) : []) : [],
      llm_analysis,
      automatic_analysis: dataAnalysis && dataAnalysis.success ? dataAnalysis.exploration : null,
      analysis_summary: dataAnalysis && dataAnalysis.success ? dataAnalysis.llm_summary : null
    };
  } catch (error) {
    throw new Error(`Fehler beim Analysieren der ${fileExtension}-Datei: ${error.message}`);
  }
}
