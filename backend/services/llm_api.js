import fs from 'fs/promises';
import path from 'path';
import { callLLMAPI } from './llm.js';

// CSV-Datei-Analysefunktion
export async function analyzeCsvFile(filePath) {
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

  const prompt = `Du bist ein extrem erfahrener Data Scientist. Analyse die folgende Datei und gib eine detaillierte Analyse der Daten zurück.`;
  
  // LLM-basierte Analyse
  const llm_analysis = await callLLMAPI(prompt, filePath);
  
  return {
    columns: headers,
    rowCount: lines.length - 1,
    dataTypes,
    sampleData: sampleRows.slice(0, 5),
    llm_analysis: llm_analysis
  };
}

// JSON-Datei analysieren
export async function analyzeJsonFile(filePath) {
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

    const prompt = `Du bist ein extrem erfahrener Data Scientist. Analysiere die folgende JSON-Datei und gib eine detaillierte Analyse der Daten zurück.`;
    
    // LLM-basierte Analyse
    const llm_analysis = await callLLMAPI(prompt, filePath);
    
    return {
      columns,
      rowCount,
      dataTypes,
      sampleData,
      llm_analysis
    };
  } catch (error) {
    throw new Error(`Fehler beim Analysieren der JSON-Datei: ${error.message}`);
  }
}

// Excel-Datei analysieren (noch nicht implementiert)
export async function analyzeExcelFile(filePath) {
  try {
    // Für Excel-Dateien verwenden wir eine einfache Text-Analyse
    // In einer echten Implementierung würdest du hier eine Excel-Parsing-Bibliothek verwenden
    const prompt = `Du bist ein extrem erfahrener Data Scientist. Analysiere die folgende Excel-Datei und gib eine detaillierte Analyse der Daten zurück.`;
    
    const llm_analysis = await callLLMAPI(prompt, filePath);
    
    return {
      columns: ['Excel-Spalten werden durch LLM analysiert'],
      rowCount: 'Unbekannt - wird durch LLM analysiert',
      dataTypes: {},
      sampleData: [],
      llm_analysis
    };
  } catch (error) {
    throw new Error(`Fehler beim Analysieren der Excel-Datei: ${error.message}`);
  }
}

// Text-Datei analysieren
export async function analyzeTextFile(filePath) {
  try {
    const textContent = await fs.readFile(filePath, 'utf-8');
    const lines = textContent.split('\n').filter(line => line.trim());
    
    const prompt = `Du bist ein extrem erfahrener Data Scientist. Analysiere die folgende Text-Datei und gib eine detaillierte Analyse der Daten zurück.`;
    
    const llm_analysis = await callLLMAPI(prompt, filePath);
    
    return {
      columns: ['Text-Inhalt'],
      rowCount: lines.length,
      dataTypes: { 'Text-Inhalt': 'text' },
      sampleData: lines.slice(0, 5).map(line => [line]),
      llm_analysis
    };
  } catch (error) {
    throw new Error(`Fehler beim Analysieren der Text-Datei: ${error.message}`);
  }
}

// Generische Datei analysieren
export async function analyzeGenericFile(filePath, fileExtension) {
  try {
    const prompt = `Du bist ein extrem erfahrener Data Scientist. Analysiere die folgende ${fileExtension}-Datei und gib eine detaillierte Analyse der Daten zurück.`;
    
    const llm_analysis = await callLLMAPI(prompt, filePath);
    
    return {
      columns: [`${fileExtension.substring(1).toUpperCase()}-Inhalt`],
      rowCount: 'Unbekannt',
      dataTypes: { [`${fileExtension.substring(1).toUpperCase()}-Inhalt`]: 'mixed' },
      sampleData: [],
      llm_analysis
    };
  } catch (error) {
    throw new Error(`Fehler beim Analysieren der ${fileExtension}-Datei: ${error.message}`);
  }
}

// LLM-Empfehlungen für Algorithmus und Features
export async function getLLMRecommendations(analysis, filePath) {
  const prompt = `Du bist ein erfahrener Machine Learning Experte. Analysiere diese CSV-Datei und gib PRÄZISE Empfehlungen zurück.

DATEN-INFORMATIONEN:
- Spalten: ${analysis.columns.join(', ')}
- Anzahl Zeilen: ${analysis.rowCount}
- Datentypen: ${Object.entries(analysis.dataTypes).map(([col, type]) => `${col}: ${type}`).join(', ')}

BEISPIEL-DATEN (erste 5 Zeilen):
${analysis.sampleData.map((row, i) => 
  `Zeile ${i+1}: ${analysis.columns.map((col, j) => `${col}=${row[j]}`).join(', ')}`
).join('\n')}

AUFGABE: Analysiere die Daten und gib EXAKT die folgenden Empfehlungen zurück im JSON-Format:

{
  "targetVariable": "[Name der Zielvariable - die Spalte die vorhergesagt werden soll]",
  "features": ["[Liste der Features/Eingangsvariablen ohne die Zielvariable]"],
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
1. Identifiziere die wahrscheinlichste Zielvariable
2. Alle anderen relevanten Spalten könnten Features sein
3. Bestimme ob es sich um Classification (kategorische Zielvariable) oder Regression (numerische Zielvariable) handelt
4. Wähle den besten Algorithmus basierend auf den Daten
5. Gib sinnvolle Hyperparameter passend zu dem Datensatz, dem Algorithmus und der Aufgabe an
6. Antworte NUR mit dem JSON-Objekt, keine zusätzlichen Erklärungen außerhalb`;

  try {
    const response = await callLLMAPI(prompt, filePath);
    
    // JSON aus der Antwort extrahieren
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      const recommendations = JSON.parse(jsonMatch[0]);
      
      // Validierung der Empfehlungen
      if (!recommendations.targetVariable || !recommendations.features || !recommendations.algorithm) {
        throw new Error('Unvollständige LLM-Empfehlungen');
      }
      
      // Sicherstellen, dass targetVariable nicht in features ist
      recommendations.features = recommendations.features.filter(f => f !== recommendations.targetVariable);
      
      return recommendations;
    } else {
      throw new Error('Konnte JSON nicht aus LLM-Antwort extrahieren');
    }
    
  } catch (error) {
    console.error('Fehler bei LLM-Empfehlungen:', error);
    
    // Fallback-Empfehlungen basierend auf einfacher Heuristik
    return generateFallbackRecommendations(analysis);
  }
}

// Fallback-Empfehlungen wenn LLM nicht funktioniert
export function generateFallbackRecommendations(analysis) {
  const { columns, dataTypes } = analysis;
  
  // Zielvariable erraten (oft letzte Spalte oder enthält "target", "label", "class", "y")
  let targetVariable = columns[columns.length - 1];
  for (const col of columns) {
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
  
  const features = columns.filter(col => col !== targetVariable);
  const targetType = dataTypes[targetVariable];
  
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
    reasoning: "Automatische Fallback-Empfehlung basierend auf Datenstruktur",
    dataSourceName: "CSV Dataset"
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
- Nur gültiges JSON zurückgeben, keine zusätzlichen Kommentare oder Texte`;

    const response = await callLLMAPI(prompt);
    
    // Verbesserte JSON-Extraktion und Validierung
    const cleanAndExtractJSON = (text) => {
      // Entferne Whitespace am Anfang und Ende
      text = text.trim();
      
      // Suche nach dem ersten { und dem letzten }
      const startIndex = text.indexOf('{');
      const endIndex = text.lastIndexOf('}');
      
      if (startIndex === -1 || endIndex === -1) {
        throw new Error('Kein gültiges JSON-Objekt gefunden');
      }
      
      const jsonString = text.slice(startIndex, endIndex + 1);
      
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
        console.error('JSON-Parsing-Fehler:', parseError);
        console.error('Problematischer JSON-String:', jsonString);
        throw new Error(`Konnte JSON nicht parsen: ${parseError.message}`);
      }
    };
    
    // Verwende die neue Extraktionsmethode
    const insights = cleanAndExtractJSON(response);
    
    // Bestehende Metadaten-Ergänzung
    insights.evaluatedAt = new Date().toISOString();
    insights.evaluatedBy = 'Gemini AI';
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
