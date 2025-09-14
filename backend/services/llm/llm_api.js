import fs from 'fs/promises';
import path from 'path';
import { callLLMAPI } from './llm.js';
import { getCachedDataAnalysis } from '../data/data_exploration.js';
import { getAgentModel, logAgentCall, AGENTS, getAgentConfig } from './agent_config.js';

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

BEISPIEL-DATEN (erste 5 Zeilen):
${filteredSampleData.map((row, i) => 
  `Zeile ${i+1}: ${filteredColumns.map((col, j) => `${col}=${row[j]}`).join(', ')}`
).join('\n')}`;
  }

  const prompt = `Du bist ein erfahrener Machine Learning Experte. Analysiere diese automatische Datenübersicht und gib PRÄZISE Empfehlungen zurück.

AUTOMATISCHE DATENÜBERSICHT (NUR ERLAUBTE FEATURES):
${dataOverview}

NUTZERWÜNSCHE (falls vorhanden):
${userPreferences ? userPreferences : 'Keine speziellen Wünsche übermittelt.'}

AUFGABE: Analysiere die Daten und gib EXAKT die folgenden Empfehlungen zurück im JSON-Format:

{
  "targetVariable": "[Name der Zielvariable - die Spalte die vorhergesagt werden soll]", // WICHTIG: NUR die Spalte die vorhergesagt werden soll, keine sonstigen Namen sind erlaubt; 
  "features": ["[Liste der Features/Eingangsvariablen ohne die Zielvariable - NUR aus den verfügbaren Spalten]"],
  "modelType": "[Classification oder Regression]",
  "algorithm": "[Bester Algorithmus: RandomForestClassifier, LogisticRegression, SVM, XGBoostClassifier, RandomForestRegressor, LinearRegression, SVR, XGBoostRegressor, MLPClassifier, MLPRegressor]",
  "hyperparameters": {
    "[Parameter1]": "[Wert1]",
    "[Parameter2]": "[Wert2]"
  },
  "reasoning": "[Kurze Begründung der Entscheidungen]",
  "dataSourceName": "[Aussagekräftiger Name für das Dataset]"
}

 WICHTIGE REGELN:
1. Identifiziere die wahrscheinlichste Zielvariable aus den verfügbaren Spalten
2. Verwende NUR die verfügbaren Spalten als Features (ausgeschlossene Spalten sind nicht verfügbar)
3. IMPORTANT: Schließe sinnlose Features wie "ID", "Name" aus; Schließe auch Features aus, die nichts mit der Aufgabe zu tun haben
4. Bestimme ob es sich um Classification (kategorische Zielvariable) oder Regression (numerische Zielvariable) handelt
5. Wähle den besten Algorithmus basierend auf den verfügbaren Daten
6. Gib sinnvolle Hyperparameter passend zu dem Datensatz, dem Algorithmus und der Aufgabe an
7. Antworte NUR mit dem JSON-Objekt, keine zusätzlichen Erklärungen außerhalb

 WICHTIG: Berücksichtige ausdrücklich die NUTZERWÜNSCHE, sofern diese nicht im Widerspruch zur Datenlage stehen (z. B. eine Zielvariable, die nicht existiert, darf ignoriert werden). Priorisiere valide Nutzerangaben wie gewünschte Zielvariable, bevorzugter Modelltyp/Algorithmus oder auszuschließende Features.

 WICHTIG: Gib NUR das JSON-Objekt zurück, keine Markdown-Formatierung oder zusätzlichen Text.`;

  try {
    // Verwende den speziell konfigurierten Data Explorer Agent
    const agentModel = getAgentModel(AGENTS.DATA_EXPLORER);
    const agentConfig = getAgentConfig(AGENTS.DATA_EXPLORER);
    
    logAgentCall(AGENTS.DATA_EXPLORER, agentModel, 'LLM-Empfehlungen für ML-Pipeline');
    console.log(`🤖 ${agentConfig.name} startet mit Modell: ${agentModel}`);
    
    const response = await callLLMAPI(prompt, null, agentModel, agentConfig.retries || 3);
    
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
    
    // Sicherstellen, dass targetVariable nicht in features ist
    recommendations.features = recommendations.features.filter(f => f !== recommendations.targetVariable);
    
    // Zusätzliche Validierung: Features müssen in den verfügbaren Spalten sein
    const availableColumns = filterColumns(analysis.columns, selectedFeatures, excludedFeatures);
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
    
    // Fallback-Empfehlungen basierend auf einfacher Heuristik
    console.log('Verwende Fallback-Empfehlungen');
    return generateFallbackRecommendations(analysis, selectedFeatures, excludedFeatures);
  }
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

    const prompt = `Du bist ein erfahrener Machine Learning Experte und Performance-Analyst. Bewerte die Performance-Metriken dieses ML-Modells umfassend und professionell.

PROJEKT-KONTEXT:
- Projektname: ${context.projectName}
- Algorithmus: ${context.algorithm}
- Model-Typ: ${context.modelType}
- Zielvariable: ${context.targetVariable}
- Features: ${context.features.join(', ')}
- Datenquelle: ${context.dataSourceName}

PERFORMANCE-METRIKEN:
${JSON.stringify(context.performanceMetrics, null, 2)}

URSPRÜNGLICHE KI-EMPFEHLUNGEN:
${context.recommendations ? JSON.stringify(context.recommendations, null, 2) : 'Keine verfügbar'}

AUFGABE: Führe eine tiefgehende Performance-Analyse durch und erstelle einen professionellen Evaluationsbericht.

Antworte im folgenden JSON-Format:
{
  "overallScore": 0.0-10.0,
  "performanceGrade": "Excellent|Good|Fair|Poor|Critical",
  "summary": "Kurze, prägnante Zusammenfassung der Model-Performance in 1-2 Sätzen",
  "detailedAnalysis": {
    "strengths": ["Stärke 1", "Stärke 2", "Stärke 3"],
    "weaknesses": ["Schwäche 1", "Schwäche 2"],
    "keyFindings": ["Wichtiger Befund 1", "Wichtiger Befund 2"]
  },
  "metricsInterpretation": ${JSON.stringify(metricsInterpretationTemplate, null, 2)},
  "improvementSuggestions": [
    {
      "category": "Data Quality|Feature Engineering|Algorithm Tuning|Model Architecture",
      "suggestion": "Konkrete Verbesserungsempfehlung",
      "expectedImpact": "Low|Medium|High",
      "implementation": "Wie kann das umgesetzt werden?"
    }
  ],
  "businessImpact": {
    "readiness": "Production Ready|Needs Improvement|Not Ready",
    "riskAssessment": "Low|Medium|High",
    "recommendation": "Empfehlung für den Business-Einsatz"
  },
  "nextSteps": [
    "Nächster Schritt 1",
    "Nächster Schritt 2"
  ],
  "confidenceLevel": 0.0-1.0,
  "version": "1.0"
}
WICHTIG: 
- Interpretiere ALLE verfügbaren Metriken in metricsInterpretation
- Verwende die exakten Metrik-Namen und -Werte aus den Performance-Metriken
- Gib eine fundierte, datengetriebene Analyse ab
- Nur gültiges JSON zurückgeben, keine zusätzlichen Kommentare oder Texte
- Antworte NUR mit dem JSON-Objekt, keine zusätzlichen Erklärungen außerhalb
- Antworten müssen in deutscher Sprache sein`;

    // Verwende den Performance Analyst Agent für Performance-Evaluation
    const agentModel = getAgentModel(AGENTS.PERFORMANCE_ANALYST);
    const agentConfig = getAgentConfig(AGENTS.PERFORMANCE_ANALYST);
    
    logAgentCall(AGENTS.PERFORMANCE_ANALYST, agentModel, 'Performance-Evaluation');
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
