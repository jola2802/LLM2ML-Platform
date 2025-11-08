/**
 * Hyperparameter-Optimizer-Worker-Agent
 * 
 * Schlägt optimale Hyperparameter basierend auf Datenanalyse vor.
 * Berücksichtigt Datencharakteristika, Problemtyp und verfügbare Ressourcen.
 */

import { BaseWorker } from './0_base_agent.js';
import { HYPERPARAMETER_OPTIMIZATION_PROMPT, HYPERPARAMETER_OPTIMIZER_TEST_PROMPT, formatPrompt } from './prompts.js';

export class HyperparameterOptimizerWorker extends BaseWorker {
  constructor() {
    super('HYPERPARAMETER_OPTIMIZER');
  }

  async execute(pipelineState) {
    this.log('info', 'Starte Hyperparameter-Optimierung');
    
    const { project, results } = pipelineState;
    
    // Prüfe, ob Datenanalyse verfügbar ist
    if (!results.DATA_ANALYZER) {
      throw new Error('Datenanalyse erforderlich für Hyperparameter-Optimierung');
    }

    try {
      const hyperparameterSuggestions = await this.optimizeHyperparameters(
        results.DATA_ANALYZER, 
        project
      );
      
      this.log('success', 'Hyperparameter-Optimierung erfolgreich abgeschlossen');
      return hyperparameterSuggestions;

    } catch (error) {
      this.log('error', 'Hyperparameter-Optimierung fehlgeschlagen', error.message);
      throw error;
    }
  }

  async optimizeHyperparameters(dataAnalysis, project) {
    const prompt = formatPrompt(HYPERPARAMETER_OPTIMIZATION_PROMPT, {
      dataAnalysis: JSON.stringify(dataAnalysis, null, 2),
      projectName: project.name,
      algorithm: project.algorithm || 'Nicht spezifiziert',
      featuresCount: Array.isArray(project.features) ? project.features.length : 0,
      datasetSize: dataAnalysis.dataset ? 'Verfügbar' : 'Unbekannt'
    });

    const response = await this.callLLM(prompt);
    const text = typeof response === 'string' ? response : response?.result || '';
    
    // Extrahiere JSON aus der Antwort
    let suggestions = this.extractJSON(text);
    
    // Validiere und ergänze die Vorschläge
    suggestions = this.validateAndEnhanceSuggestions(suggestions, project, dataAnalysis);
    
    return suggestions;
  }

  validateAndEnhanceSuggestions(suggestions, project, dataAnalysis) {
    // Fallback-Hyperparameter falls JSON-Parsing fehlschlägt
    if (!suggestions || Object.keys(suggestions).length === 0) {
      this.log('warn', 'JSON-Parsing fehlgeschlagen, verwende Fallback-Hyperparameter');
      suggestions = this.getFallbackHyperparameters(project);
    }

    // Ergänze fehlende Felder
    if (!suggestions.primary_algorithm) {
      suggestions.primary_algorithm = project.algorithm || 'RandomForestClassifier';
    }

    if (!suggestions.hyperparameters) {
      suggestions.hyperparameters = {};
    }

    if (!suggestions.reasoning) {
      suggestions.reasoning = 'Hyperparameter basierend auf Standard-Best-Practices ausgewählt';
    }

    // Füge Metadaten hinzu
    suggestions.metadata = {
      timestamp: new Date().toISOString(),
      dataset: dataAnalysis.dataset || 'unknown',
      algorithm: suggestions.primary_algorithm,
      optimized_by: this.agentKey
    };

    return suggestions;
  }

  getFallbackHyperparameters(project) {
    const algorithm = project.algorithm || 'RandomForestClassifier';
    
    const fallbackParams = {
      RandomForestClassifier: {
        n_estimators: 100,
        max_depth: 10,
        min_samples_split: 2,
        min_samples_leaf: 1,
        random_state: 42
      },
      GradientBoostingClassifier: {
        n_estimators: 100,
        learning_rate: 0.1,
        max_depth: 3,
        random_state: 42
      },
      LogisticRegression: {
        C: 1.0,
        random_state: 42,
        max_iter: 1000
      },
      SVM: {
        C: 1.0,
        kernel: 'rbf',
        random_state: 42
      },
      XGBClassifier: {
        n_estimators: 100,
        max_depth: 6,
        learning_rate: 0.1,
        random_state: 42
      },
      XGBRegressor: {
        n_estimators: 100,
        max_depth: 6,
        learning_rate: 0.1,
        random_state: 42
      },
      NeuralNetworkClassifier: {
        hidden_layer_sizes: (100,0),
        max_iter: 1000,
        random_state: 42
      },
      NeuralNetworkRegressor: {
        hidden_layer_sizes: (100,0),
        max_iter: 1000,
        random_state: 42
      }
    };

    return {
      primary_algorithm: algorithm,
      hyperparameters: {
        [algorithm]: fallbackParams[algorithm] || fallbackParams.RandomForestClassifier
      },
      reasoning: 'Fallback-Hyperparameter basierend auf Standard-Best-Practices',
      expected_performance: 'Moderate bis gute Performance erwartet',
    };
  }

  async test() {
    const testPrompt = formatPrompt(HYPERPARAMETER_OPTIMIZER_TEST_PROMPT, {
      agentName: this.config.name
    });

    try {
      const response = await this.callLLM(testPrompt, null, 10);
      const result = typeof response === 'string' ? response : response?.result || '';
      return result.toLowerCase().includes('ok');
    } catch (error) {
      this.log('error', 'Test fehlgeschlagen', error.message);
      return false;
    }
  }
}
