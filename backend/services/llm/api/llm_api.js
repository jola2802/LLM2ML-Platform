import fs from 'fs/promises';
import path from 'path';
import { callLLMAPI } from './llm.js';
import { getCachedDataAnalysis } from '../../data/data_exploration.js';
import { getAgentModel, logAgentCall, getAgentConfig, WORKER_AGENTS } from '../agents/config_agent_network.js';
import { LLM_RECOMMENDATIONS_PROMPT, PERFORMANCE_EVALUATION_PROMPT, formatPrompt } from '../agents/prompts.js';

// LLM-Empfehlungen für Algorithmus und Features
export async function getLLMRecommendations(analysis, filePath = null, venvDir, selectedFeatures = null, excludedFeatures = null, userPreferences = null) {
  // Verwende automatische Datenexploration falls verfügbar
  let dataOverview = '';
  
  if (filePath) {
    try {
      const dataAnalysis = await getCachedDataAnalysis(filePath, false);
      if (dataAnalysis.success) {
        // Filtere die Datenexploration basierend auf ausgewählten/ausgeschlossenen Features
        dataOverview = filterDataOverviewForFeatures(dataAnalysis.llm_summary, selectedFeatures, excludedFeatures);
      }
    } catch (error) {
      console.error('Fehler bei automatischer Datenexploration:', error);
    }
  }
  
  // Fallback auf ursprüngliche Analyse falls keine automatische Analyse verfügbar
  if (!dataOverview) {
    // Filtere die ursprüngliche Analyse
    const filteredColumns = filterColumns(analysis.columns, selectedFeatures, excludedFeatures);
    const filteredDataTypes = filterDataTypes(analysis.dataTypes, selectedFeatures, excludedFeatures);
    const filteredSampleData = filterSampleData(analysis.sampleData, analysis.columns, selectedFeatures, excludedFeatures);
    
    dataOverview = `DATEN-INFORMATIONEN:
- Verfügbare Spalten: ${filteredColumns.join(', ')}
- Anzahl Zeilen: ${analysis.rowCount}
- Datentypen: ${Object.entries(filteredDataTypes).map(([col, type]) => `${col}: ${type}`).join(', ')}

BEISPIEL-DATEN (erste 10 Zeilen):
${filteredSampleData.map((row, i) => 
  `Zeile ${i+1}: ${filteredColumns.map((col, j) => `${col}=${row[j]}`).join(', ')}`
).join('\n')}`;
  }

  const prompt = formatPrompt(LLM_RECOMMENDATIONS_PROMPT, {
    dataOverview,
    userPreferences: userPreferences ? userPreferences : 'Keine speziellen Wünsche übermittelt.'
  });

  try {
    // Verwende den speziell konfigurierten Data Explorer Agent
    const agentModel = getAgentModel(WORKER_AGENTS.DATA_ANALYZER.key);
    const agentConfig = getAgentConfig(WORKER_AGENTS.DATA_ANALYZER.key);
    
    logAgentCall(WORKER_AGENTS.DATA_ANALYZER.key, agentModel, 'LLM-Empfehlungen für ML-Pipeline');
    console.log(`🤖 ${agentConfig.name} startet mit Modell: ${agentModel}`);
    
    const response = await callLLMAPI(prompt, null, agentModel, agentConfig.retries || 3);
    
    // Extrahiere und validiere JSON aus der Antwort
    const recommendations = extractAndValidateJSON(response);
    
    // Validiere die Features
    const availableColumns = filterColumns(analysis.columns, selectedFeatures, excludedFeatures);
    recommendations.features = recommendations.features.filter(f => f !== recommendations.targetVariable);
    recommendations.features = recommendations.features.filter(f => availableColumns.includes(f));
    
    // Sicherstellen, dass mindestens ein Feature vorhanden ist
    if (recommendations.features.length === 0) {
      throw new Error('Keine gültigen Features gefunden');
    }
    
    console.log('LLM-Empfehlungen erfolgreich extrahiert:', {
      targetVariable: recommendations.targetVariable,
      features: recommendations.features,
      algorithm: recommendations.algorithm,
      modelType: recommendations.modelType
    });
    
    return recommendations;
    
  } catch (error) {
    console.error('Fehler bei LLM-Empfehlungen:', error);

    // Versuche es noch einmal
    const response = await callLLMAPI(prompt, null, agentModel, agentConfig.retries || 3);
    const recommendations = extractAndValidateJSON(response);
    
    if (!error) { 
      return recommendations;
    } else {
      console.log('Verwende Fallback-Empfehlungen');
      return generateFallbackRecommendations(analysis, selectedFeatures, excludedFeatures);
    }
  }
}

function extractAndValidateJSON(response) {
  // Extrahiere JSON aus der Antwort - robuster für verschiedene Response-Formate
  let jsonText = '';
  if (response && response.result) {
    jsonText = response.result;
  } else if (typeof response === 'string') {
    jsonText = response;
  } else {
    throw new Error('Ungültige Response vom LLM');
  }
  
  // Entferne Markdown-Formatierung und führende/nachfolgende Leerzeichen
  jsonText = jsonText.replace(/```json/g, '').replace(/```/g, '').trim();
  
  // Entferne führende Anführungszeichen und Leerzeichen die manchmal in LLM-Antworten vorkommen
  jsonText = jsonText.replace(/^[\s"]*/, '').replace(/[\s"]*$/, '');
  
  // Konvertiere Python None zu JSON null
  jsonText = jsonText.replace(/:\s*None\b/g, ': null');
  
  // Versuche verschiedene JSON-Extraktionsmethoden
  let jsonMatch = null;
  
  // Methode 1: Suche nach JSON-Objekt mit geschweiften Klammern
  jsonMatch = jsonText.match(/\{[\s\S]*\}/);
  
  if (!jsonMatch) {
    // Methode 2: Suche nach JSON-Array falls das erste fehlschlägt
    jsonMatch = jsonText.match(/\[[\s\S]*\]/);
  }
  
  if (!jsonMatch) {
    // Methode 3: Versuche den gesamten Text als JSON zu parsen
    try {
      JSON.parse(jsonText);
      jsonMatch = [jsonText];
    } catch (parseError) {
      throw new Error('Konnte JSON nicht aus LLM-Antwort extrahieren');
    }
  }
  
  const recommendations = JSON.parse(jsonMatch[0]);
  
  // Validiere die Empfehlungen
  if (!recommendations.targetVariable || !recommendations.features || !recommendations.algorithm) {
    throw new Error('Unvollständige LLM-Empfehlungen');
  }
  
  return recommendations;
}

// Hilfsfunktionen zum Filtern der Daten
function filterColumns(allColumns, selectedFeatures, excludedFeatures) {
  if (selectedFeatures && selectedFeatures.length > 0) {
    // Nur ausgewählte Features verwenden
    return allColumns.filter(col => selectedFeatures.includes(col));
  } else if (excludedFeatures && excludedFeatures.length > 0) {
    // Ausgeschlossene Features entfernen
    return allColumns.filter(col => !excludedFeatures.includes(col));
  }
  // Keine Filterung
  return allColumns;
}

function filterDataTypes(allDataTypes, selectedFeatures, excludedFeatures) {
  const filteredColumns = filterColumns(Object.keys(allDataTypes), selectedFeatures, excludedFeatures);
  const filteredDataTypes = {};
  
  filteredColumns.forEach(col => {
    if (allDataTypes[col]) {
      filteredDataTypes[col] = allDataTypes[col];
    }
  });
  
  return filteredDataTypes;
}

function filterSampleData(sampleData, allColumns, selectedFeatures, excludedFeatures) {
  const filteredColumns = filterColumns(allColumns, selectedFeatures, excludedFeatures);
  const columnIndices = filteredColumns.map(col => allColumns.indexOf(col));
  
  return sampleData.map(row => {
    const filteredRow = [];
    columnIndices.forEach(index => {
      if (index >= 0 && index < row.length) {
        filteredRow.push(row[index]);
      }
    });
    return filteredRow;
  });
}

function filterDataOverviewForFeatures(dataOverview, selectedFeatures, excludedFeatures) {
  // Einfache Filterung der Datenübersicht basierend auf verfügbaren Features
  // Dies ist eine Basis-Implementierung - könnte erweitert werden für komplexere JSON-Strukturen
  
  if (!selectedFeatures && !excludedFeatures) {
    return dataOverview;
  }
  
  // Für jetzt geben wir einen Hinweis zurück, dass gefiltert wurde
  const availableColumns = filterColumns([], selectedFeatures, excludedFeatures);
  
  return `GEFILTERTE DATENÜBERSICHT:
Verfügbare Features: ${selectedFeatures ? selectedFeatures.join(', ') : 'Alle außer: ' + excludedFeatures.join(', ')}
${dataOverview}`;
}

// Fallback-Empfehlungen wenn LLM nicht funktioniert
export function generateFallbackRecommendations(analysis, selectedFeatures = null, excludedFeatures = null) {
  const { columns, dataTypes } = analysis;
  
  // Filtere Spalten basierend auf UI-Auswahl
  const availableColumns = filterColumns(columns, selectedFeatures, excludedFeatures);
  const availableDataTypes = filterDataTypes(dataTypes, selectedFeatures, excludedFeatures);
  
  // Zielvariable erraten (oft letzte Spalte oder enthält "target", "label", "class", "y")
  let targetVariable = availableColumns[availableColumns.length - 1];
  for (const col of availableColumns) {
    if (col.toLowerCase().includes('target') || 
        col.toLowerCase().includes('label') || 
        col.toLowerCase().includes('class') ||
        col.toLowerCase().includes('y') ||
        col.toLowerCase().includes('price') ||
        col.toLowerCase().includes('salary')) {
      targetVariable = col;
      break;
    }
  }
  
  const features = availableColumns.filter(col => col !== targetVariable);
  const targetType = availableDataTypes[targetVariable];
  
  const modelType = targetType === 'numeric' ? 'Regression' : 'Classification';
  const algorithm = modelType === 'Classification' ? 'RandomForestClassifier' : 'RandomForestRegressor';
  
  return {
    targetVariable,
    features,
    modelType,
    algorithm,
    hyperparameters: {
      n_estimators: 100,
      random_state: 42
    },
    reasoning: "Automatische Fallback-Empfehlung basierend auf gefilterten Daten",
    dataSourceName: "Dataset"
  };
}

// Intelligente Performance-Evaluation mit LLM
export async function evaluatePerformanceWithLLM(project) {
  try {
    // Kontext für die Evaluation sammeln
    const context = {
      projectName: project.name,
      algorithm: project.algorithm || 'RandomForest',
      modelType: project.modelType,
      targetVariable: project.targetVariable,
      features: project.features,
      dataSourceName: project.dataSourceName,
      performanceMetrics: project.performanceMetrics,
      recommendations: project.llmRecommendations,
      datasetSize: 'Unbekannt' // Könnte aus Training-Logs extrahiert werden
    };

    // Erstelle dynamische metricsInterpretation für alle verfügbaren Metriken
    const metricsInterpretationTemplate = {};
    if (context.performanceMetrics) {
      Object.keys(context.performanceMetrics).forEach(metricKey => {
        metricsInterpretationTemplate[metricKey] = {
          "value": context.performanceMetrics[metricKey],
          "interpretation": `Interpretation für ${metricKey}`,
          "benchmarkComparison": `Benchmark-Vergleich für ${metricKey}`
        };
      });
    }

    const prompt = formatPrompt(PERFORMANCE_EVALUATION_PROMPT, {
      projectName: context.projectName,
      algorithm: context.algorithm,
      modelType: context.modelType,
      targetVariable: context.targetVariable,
      features: context.features.join(', '),
      dataSourceName: context.dataSourceName,
      performanceMetrics: JSON.stringify(context.performanceMetrics, null, 2),
      recommendations: context.recommendations ? JSON.stringify(context.recommendations, null, 2) : 'Keine verfügbar',
      metricsInterpretationTemplate: JSON.stringify(metricsInterpretationTemplate, null, 2)
    });

    // Verwende den Performance Analyst Agent für Performance-Evaluation
    const agentModel = getAgentModel(WORKER_AGENTS.PERFORMANCE_ANALYZER.key);
    const agentConfig = getAgentConfig(WORKER_AGENTS.PERFORMANCE_ANALYZER.key);
    
    logAgentCall(WORKER_AGENTS.PERFORMANCE_ANALYZER.key, agentModel, 'Performance-Evaluation');
    console.log(`🤖 ${agentConfig.name} startet Performance-Analyse mit Modell: ${agentModel}`);
    
    const response = await callLLMAPI(prompt, null, agentModel);
    
    // Sicherstellen, dass response.result ein String ist
    if (!response || !response.result) {
      throw new Error('Keine gültige Antwort vom LLM erhalten');
    }

    // Verbesserte JSON-Extraktion und Validierung  
    const cleanAndExtractJSON = (text) => {    
      // Sicherstellen, dass text ein String ist
      if (typeof text !== 'string') {
        throw new Error('Text ist kein gültiger String');
      }
      
      // Entferne Markdown-Formatierung (```json, ```, etc.)
      let cleanedText = text
        .replace(/```json\s*/gi, '')  // Entferne ```json am Anfang
        .replace(/```\s*$/gi, '')     // Entferne ``` am Ende
        .replace(/^```\s*/gi, '')     // Entferne ``` am Anfang (falls kein json)
        .trim();                      // Entferne Whitespace
      
      const jsonMatch = cleanedText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('Kein gültiges JSON-Objekt gefunden');
      }
      
      const jsonString = jsonMatch[0];
      
      // try {
      //   const parsedJSON = JSON.parse(jsonString);
      //   // return parsedJSON;
      // } catch (parseError) {
      //   console.error('JSON-Parsing-Fehler:', parseError);
      //   console.error('Problematischer JSON-String:', jsonString);
      //   throw new Error(`Konnte JSON nicht parsen: ${parseError.message}`);
      // }
      
      try {
        // Versuche das JSON zu parsen
        const parsedJSON = JSON.parse(jsonString);
        
        // Zusätzliche Validierung der Schlüsselstrukturen
        const requiredKeys = [
          'overallScore', 
          'performanceGrade', 
          'summary', 
          'detailedAnalysis', 
          'metricsInterpretation', 
          'improvementSuggestions', 
          'businessImpact', 
          'nextSteps'
        ];
        
        requiredKeys.forEach(key => {
          if (!(key in parsedJSON)) {
            throw new Error(`Fehlendes Schlüssel: ${key}`);
          }
        });
        
        return parsedJSON;
      } catch (parseError) {
        // console.error('JSON-Parsing-Fehler:', parseError);
        // console.error('Problematischer JSON-String:', text);
        throw new Error(`Konnte JSON nicht parsen: ${parseError.message}`);
      }
    };
    
    // Verwende die neue Extraktionsmethode
    const insights = cleanAndExtractJSON(response.result);
    
    // Bestehende Metadaten-Ergänzung
    insights.evaluatedAt = new Date().toISOString();
    insights.evaluatedBy = 'LLM API';
    insights.version = '1.0';
    
    return insights;
    
  } catch (error) {
    console.error('Fehler bei LLM Performance-Evaluation:', error);
    
    // Fallback-Evaluation bei Fehlern
    return generateFallbackPerformanceEvaluation(project);
  }
}

// Fallback Performance-Evaluation 
export function generateFallbackPerformanceEvaluation(project) {
  const metrics = project.performanceMetrics;
  const isClassification = project.modelType === 'Classification';
  
  let overallScore = 5.0;
  let performanceGrade = 'Fair';
  let summary = 'Automatische Basis-Evaluation durchgeführt.';
  
  if (isClassification && metrics.accuracy) {
    if (metrics.accuracy >= 0.9) {
      overallScore = 9.0;
      performanceGrade = 'Excellent';
      summary = 'Sehr hohe Genauigkeit erreicht - exzellente Performance.';
    } else if (metrics.accuracy >= 0.8) {
      overallScore = 7.5;
      performanceGrade = 'Good';
      summary = 'Gute Genauigkeit erreicht - solide Performance.';
    } else if (metrics.accuracy >= 0.7) {
      overallScore = 6.0;
      performanceGrade = 'Fair';
      summary = 'Akzeptable Genauigkeit - Verbesserungen möglich.';
    } else {
      overallScore = 4.0;
      performanceGrade = 'Poor';
      summary = 'Niedrige Genauigkeit - deutliche Verbesserungen erforderlich.';
    }
  } else if (!isClassification && metrics.r2) {
    if (metrics.r2 >= 0.8) {
      overallScore = 8.5;
      performanceGrade = 'Excellent';
      summary = 'Sehr gute Vorhersagekraft - exzellente Performance.';
    } else if (metrics.r2 >= 0.6) {
      overallScore = 7.0;
      performanceGrade = 'Good';
      summary = 'Gute Vorhersagekraft - solide Performance.';
    } else if (metrics.r2 >= 0.4) {
      overallScore = 5.5;
      performanceGrade = 'Fair';
      summary = 'Moderate Vorhersagekraft - Optimierungen empfohlen.';
    } else {
      overallScore = 3.5;
      performanceGrade = 'Poor';
      summary = 'Schwache Vorhersagekraft - grundlegende Überarbeitung nötig.';
    }
  }
  
  return {
    overallScore,
    performanceGrade,
    summary,
    detailedAnalysis: {
      strengths: ['Fallback-Evaluation verfügbar'],
      weaknesses: ['Detaillierte LLM-Analyse nicht verfügbar'],
      keyFindings: ['Basis-Metriken wurden extrahiert']
    },
    metricsInterpretation: Object.fromEntries(
      Object.entries(metrics).map(([key, value]) => [
        key,
        {
          value,
          interpretation: `${key}: ${value}`,
          benchmarkComparison: 'Benchmark-Vergleich nicht verfügbar'
        }
      ])
    ),
    improvementSuggestions: [
      {
        category: 'General',
        suggestion: 'Detaillierte Analyse mit verfügbarer LLM-Verbindung durchführen',
        expectedImpact: 'Medium',
        implementation: 'API-Konfiguration prüfen und erneut evaluieren'
      }
    ],
    businessImpact: {
      readiness: 'Needs Improvement',
      riskAssessment: 'Medium',
      recommendation: 'Weitere Analyse erforderlich vor Produktions-Einsatz'
    },
    nextSteps: [
      'LLM-Verbindung überprüfen',
      'Detaillierte Performance-Analyse durchführen',
      'Modell-Optimierung basierend auf verfügbaren Metriken'
    ],
    evaluatedAt: new Date().toISOString(),
    evaluatedBy: 'Fallback System',
    version: '1.0'
  };
}
