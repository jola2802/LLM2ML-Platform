/**
 * Datenanalyse-Worker-Agent
 * 
 * Analysiert Datasets und erstellt detaillierte Insights.
 * Identifiziert Muster, Ausreißer, Korrelationen und andere wichtige Datencharakteristika.
 */

import { BaseWorker } from './base_worker.js';
import { getCachedDataAnalysis } from '../../data/data_exploration.js';

export class DataAnalyzerWorker extends BaseWorker {
  constructor() {
    super('DATA_ANALYZER');
  }

  async execute(pipelineState) {
    this.log('info', 'Starte Datenanalyse');
    
    const { project } = pipelineState;
    
    if (!project || !project.csvFilePath) {
      throw new Error('Kein Dataset-Pfad verfügbar für Datenanalyse');
    }

    try {
      // Lade vorhandene Datenanalyse oder führe neue durch
      let dataAnalysis = await getCachedDataAnalysis(project.csvFilePath);
      
      if (!dataAnalysis || !dataAnalysis.success) {
        this.log('warn', 'Keine gecachte Datenanalyse verfügbar, führe neue Analyse durch');
        dataAnalysis = await this.performDataAnalysis(project);
      } else {
        this.log('info', 'Verwende gecachte Datenanalyse');
      }

      // Erweitere die Datenanalyse mit zusätzlichen Insights
      const enhancedAnalysis = await this.enhanceDataAnalysis(dataAnalysis, project);
      
      this.log('success', 'Datenanalyse erfolgreich abgeschlossen');
      return enhancedAnalysis;

    } catch (error) {
      this.log('error', 'Datenanalyse fehlgeschlagen', error.message);
      throw error;
    }
  }

  async performDataAnalysis(project) {
    const prompt = `Führe eine umfassende Datenanalyse für das Dataset durch:

Dataset-Pfad: ${project.csvFilePath}
Projekt: ${project.name || 'Unbekannt'}
Algorithmus: ${project.algorithm || 'Nicht spezifiziert'}

Analysiere folgende Aspekte:
1. Dataset-Übersicht (Größe, Spalten, Datentypen)
2. Fehlende Werte und Datenqualität
3. Statistische Beschreibungen
4. Korrelationen zwischen Features
5. Ausreißer-Erkennung
6. Verteilung der Zielvariable (falls vorhanden)
7. Empfehlungen für Preprocessing

Gib eine strukturierte Analyse zurück.`;

    const response = await this.callLLM(prompt);
    const analysis = typeof response === 'string' ? response : response?.result || '';
    
    return {
      success: true,
      analysis,
      timestamp: new Date().toISOString(),
      dataset: project.csvFilePath
    };
  }

  async enhanceDataAnalysis(dataAnalysis, project) {
    const prompt = `Erweitere die folgende Datenanalyse mit zusätzlichen ML-spezifischen Insights:

Vorhandene Analyse:
${JSON.stringify(dataAnalysis, null, 2)}

Projekt-Kontext:
- Name: ${project.name}
- Algorithmus: ${project.algorithm || 'Nicht spezifiziert'}
- Features: ${Array.isArray(project.features) ? project.features.length : 0} ausgewählt

Füge folgende Erkenntnisse hinzu:
1. ML-Algorithmus-Empfehlungen basierend auf Datencharakteristika
2. Feature-Engineering-Vorschläge
3. Preprocessing-Empfehlungen
4. Potentielle Herausforderungen und Lösungsansätze
5. Erwartete Modell-Performance-Indikatoren

Gib eine erweiterte, strukturierte Analyse zurück.`;

    const response = await this.callLLM(prompt);
    const enhancedAnalysis = typeof response === 'string' ? response : response?.result || '';
    
    return {
      ...dataAnalysis,
      enhancedAnalysis,
      mlInsights: {
        algorithmRecommendations: this.extractAlgorithmRecommendations(enhancedAnalysis),
        preprocessingSuggestions: this.extractPreprocessingSuggestions(enhancedAnalysis),
        featureEngineering: this.extractFeatureEngineering(enhancedAnalysis)
      }
    };
  }

  extractAlgorithmRecommendations(analysis) {
    // Einfache Extraktion von Algorithmus-Empfehlungen
    const algorithms = [];
    if (analysis.toLowerCase().includes('random forest')) algorithms.push('Random Forest');
    if (analysis.toLowerCase().includes('gradient boosting')) algorithms.push('Gradient Boosting');
    if (analysis.toLowerCase().includes('svm')) algorithms.push('SVM');
    if (analysis.toLowerCase().includes('logistic regression')) algorithms.push('Logistic Regression');
    if (analysis.toLowerCase().includes('neural network')) algorithms.push('Neural Network');
    
    return algorithms.length > 0 ? algorithms : ['Random Forest', 'Gradient Boosting'];
  }

  extractPreprocessingSuggestions(analysis) {
    const suggestions = [];
    if (analysis.toLowerCase().includes('scaling')) suggestions.push('Feature Scaling');
    if (analysis.toLowerCase().includes('normalization')) suggestions.push('Normalization');
    if (analysis.toLowerCase().includes('encoding')) suggestions.push('Categorical Encoding');
    if (analysis.toLowerCase().includes('imputation')) suggestions.push('Missing Value Imputation');
    
    return suggestions.length > 0 ? suggestions : ['Feature Scaling', 'Categorical Encoding'];
  }

  extractFeatureEngineering(analysis) {
    const features = [];
    if (analysis.toLowerCase().includes('polynomial')) features.push('Polynomial Features');
    if (analysis.toLowerCase().includes('interaction')) features.push('Feature Interactions');
    if (analysis.toLowerCase().includes('selection')) features.push('Feature Selection');
    
    return features.length > 0 ? features : ['Feature Selection'];
  }
}
