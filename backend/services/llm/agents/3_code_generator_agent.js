/**
 * Code-Generator-Worker-Agent
 * 
 * Generiert optimierten Python-Code für ML-Training.
 * Verwendet moderne ML-Bibliotheken und folgt Best Practices.
 */

import { BaseWorker } from './0_base_agent.js';
import { CODE_GENERATION_PROMPT, CODE_GENERATOR_TEST_PROMPT, formatPrompt } from './prompts.js';
import { ALGORITHMS } from '../../execution/algorithms.js';
import { codeTemplate } from '../../execution/template_code.js';

export class CodeGeneratorWorker extends BaseWorker {
  constructor() {
    super('CODE_GENERATOR');
  }

  async execute(pipelineState) {
    this.log('info', 'Starte Code-Generierung');

    const { project, results } = pipelineState;

    // Füge die richtige Model-Library hinzu
    project.llmRecommendations.hyperparameters.library = ALGORITHMS[project.llmRecommendations.algorithm].library;

    // Prüfe, ob Hyperparameter verfügbar sind
    if (!results.HYPERPARAMETER_OPTIMIZER) {
      throw new Error('Hyperparameter-Vorschläge erforderlich für Code-Generierung');
    }

    try {
      const pythonCode = await this.generatePythonCode(
        project,
        results.HYPERPARAMETER_OPTIMIZER
      );

      this.log('success', 'Code-Generierung erfolgreich abgeschlossen');
      return pythonCode;

    } catch (error) {
      this.log('error', 'Code-Generierung fehlgeschlagen', error.message);
      throw error;
    }
  }

  // Use template_code.py to generate the code; therefore adapt the header in the template file
  async generatePythonCode(project, hyperparameterSuggestions) {
    // Get code from template_code.js
    const code = codeTemplate;

    // Validate the hyperparameters based on the algorithm
    // const validatedHyperparameters = this.validateHyperparameters(hyperparameterSuggestions, project.llmRecommendations.algorithm);

    // Adapt the header in the template file
    const adaptedCode = this.adaptHeader(code, project, hyperparameterSuggestions);

    // Return the adapted code
    return adaptedCode;
  }

  adaptHeader(code, project, hyperparameterSuggestions) {
    let adaptedCode = code.replace('PROJECT_NAME', "'" + project.name + "'");
    adaptedCode = adaptedCode.replace('FILE_PATH', "r'" + project.csvFilePath + "'");
    adaptedCode = adaptedCode.replace('TARGET_COLUMN', "'" + project.llmRecommendations.targetVariable + "'");
    adaptedCode = adaptedCode.replace('PROBLEM_TYPE', "'" + project.llmRecommendations.modelType + "'");
    adaptedCode = adaptedCode.replace('MODEL_TYPE', "'" + project.llmRecommendations.algorithm + "'");
    adaptedCode = adaptedCode.replace('MODEL_LIB', "'" + project.llmRecommendations.hyperparameters.library + "'");
    // Format hyperparameters as a proper string representation
    const hyperParams = project.llmRecommendations.hyperparameters.params || hyperparameterSuggestions.hyperparameters;
    // Extract inner params if nested under algorithm name
    const params = hyperParams[Object.keys(hyperParams)[0]] &&
      typeof hyperParams[Object.keys(hyperParams)[0]] === 'object' ?
      hyperParams[Object.keys(hyperParams)[0]] :
      hyperParams;
    const formattedParams = JSON.stringify(params, null, 4);

    adaptedCode = adaptedCode.replace('MODEL_PARAMS', formattedParams);
    adaptedCode = adaptedCode.replace('MODEL_SAVE_PATH', "model_" + project.id + ".pkl");
    return adaptedCode;
  }

  validateHyperparameters(hyperparameterSuggestions, algorithm) {
    // TODO: Implement this

  }

  async test() {
    const testPrompt = formatPrompt(CODE_GENERATOR_TEST_PROMPT, {
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
