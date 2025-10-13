/**
 * Datenanalyse-Worker-Agent
 * 
 * Analysiert Datasets und erstellt detaillierte Insights.
 * Identifiziert Muster, Ausreißer, Korrelationen und andere wichtige Datencharakteristika.
 */

import { BaseWorker } from './0_base_agent.js';
import { getCachedDataAnalysis } from '../../data/data_exploration.js';
import { DATA_ANALYSIS_PROMPT, ENHANCED_DATA_ANALYSIS_PROMPT, formatPrompt } from './prompts.js';

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
        return dataAnalysis;
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
    const prompt = formatPrompt(DATA_ANALYSIS_PROMPT, {
      csvFilePath: project.csvFilePath,
      projectName: project.name || 'Unbekannt',
      algorithm: project.algorithm || 'Nicht spezifiziert'
    });

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
    const prompt = formatPrompt(ENHANCED_DATA_ANALYSIS_PROMPT, {
      dataAnalysis: JSON.stringify(dataAnalysis, null, 2),
      projectName: project.name,
      algorithm: project.algorithm || 'Nicht spezifiziert',
      featuresCount: Array.isArray(project.features) ? project.features.length : 0
    });

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
