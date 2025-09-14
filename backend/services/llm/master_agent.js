/**
 * Master-Agent: Koordiniert die ML-Pipeline
 * 
 * Der Master-Agent kennt die gesamte ML-Pipeline und verteilt Aufgaben
 * an spezialisierte Worker-Agents. Er entscheidet basierend auf dem
 * aktuellen Projekt-Status, welcher Agent als nächstes arbeiten soll.
 */

import { EventEmitter } from 'events';
import { callLLMAPI } from './llm.js';
import { getCachedDataAnalysis } from '../data/data_exploration.js';
import { 
  MASTER_AGENT,
  WORKER_AGENTS,
  PIPELINE_STEPS,
  getAgentConfig,
  getAgentModel,
  getNextPipelineStep,
  logAgentCall
} from './network_agent_config.js';

/**
 * Worker-Agent-Implementierungen
 */
import { DataAnalyzerWorker } from './workers/data_analyzer_worker.js';
import { HyperparameterOptimizerWorker } from './workers/hyperparameter_optimizer_worker.js';
import { CodeGeneratorWorker } from './workers/code_generator_worker.js';
import { CodeReviewerWorker } from './workers/code_reviewer_worker.js';
import { PerformanceAnalyzerWorker } from './workers/performance_analyzer_worker.js';

/**
 * Master-Agent-Klasse
 */
export class MasterAgent extends EventEmitter {
  constructor() {
    super();
    this.config = getAgentConfig(MASTER_AGENT.key);
    this.model = getAgentModel(MASTER_AGENT.key);
    
    // Worker-Agent-Instanzen
    this.workers = {
      DATA_ANALYZER: new DataAnalyzerWorker(),
      HYPERPARAMETER_OPTIMIZER: new HyperparameterOptimizerWorker(),
      CODE_GENERATOR: new CodeGeneratorWorker(),
      CODE_REVIEWER: new CodeReviewerWorker(),
      PERFORMANCE_ANALYZER: new PerformanceAnalyzerWorker()
    };
    
    // Pipeline-Status
    this.pipelineState = {
      project: null,
      completedSteps: [],
      currentStep: null,
      results: {},
      errors: []
    };
  }

  /**
   * Startet die ML-Pipeline für ein Projekt
   */
  async runPipeline(project) {
    const projectId = project.id || project.name || 'unknown';
    console.log(`\n🚀 === MASTER-AGENT STARTET PIPELINE für ${project.name} ===`);
    
    // Pipeline-Status initialisieren
    this.pipelineState = {
      project,
      completedSteps: [],
      currentStep: null,
      results: {},
      errors: []
    };

    this.emit('pipelineStarted', { projectId, project });

    try {
      // Pipeline-Schritte durchlaufen
      while (true) {
        const nextStep = getNextPipelineStep(this.pipelineState.completedSteps);
        
        if (!nextStep) {
          console.log('✅ Pipeline abgeschlossen - alle Schritte erledigt');
          break;
        }

        console.log(`\n📍 SCHRITT ${nextStep.step}: ${nextStep.name}`);
        console.log(`   Agent: ${getAgentConfig(nextStep.agent).name}`);
        
        this.pipelineState.currentStep = nextStep;
        this.emit('stepStarted', { projectId, step: nextStep });

        try {
          // Worker-Agent ausführen
          const result = await this.executeWorkerAgent(nextStep.agent, this.pipelineState);
          
          if (result) {
            // Schritt erfolgreich abgeschlossen
            this.pipelineState.completedSteps.push(nextStep);
            this.pipelineState.results[nextStep.agent] = result;
            
            console.log(`✅ ${nextStep.name} erfolgreich abgeschlossen`);
            this.emit('stepCompleted', { projectId, step: nextStep, result });
          } else {
            // Schritt fehlgeschlagen
            const error = new Error(`${nextStep.name} fehlgeschlagen - kein Ergebnis erhalten`);
            this.pipelineState.errors.push(error.message);
            
            if (nextStep.required) {
              console.error(`❌ Erforderlicher Schritt ${nextStep.name} fehlgeschlagen`);
              throw error;
            } else {
              console.warn(`⚠️ Optionaler Schritt ${nextStep.name} übersprungen`);
              this.pipelineState.completedSteps.push(nextStep); // Als abgeschlossen markieren
            }
          }
          
        } catch (error) {
          console.error(`❌ Fehler in Schritt ${nextStep.name}:`, error.message);
          this.pipelineState.errors.push(`${nextStep.name}: ${error.message}`);
          
          if (nextStep.required) {
            this.emit('pipelineFailed', { projectId, step: nextStep, error });
            throw error;
          } else {
            console.warn(`⚠️ Optionaler Schritt ${nextStep.name} übersprungen aufgrund Fehler`);
            this.pipelineState.completedSteps.push(nextStep); // Als abgeschlossen markieren
          }
        }
      }

      // Pipeline erfolgreich abgeschlossen
      const finalResult = this.getFinalResult();
      console.log(`\n✅ === PIPELINE ERFOLGREICH BEENDET ===`);
      console.log(`📝 Generierter Code: ${finalResult ? finalResult.length : 0} Zeichen`);
      
      if (this.pipelineState.errors.length > 0) {
        console.log(`⚠️ Warnungen: ${this.pipelineState.errors.length}`);
      }

      this.emit('pipelineCompleted', { projectId, result: finalResult, state: this.pipelineState });
      return finalResult;

    } catch (error) {
      console.error(`\n❌ === PIPELINE FEHLGESCHLAGEN ===`);
      console.error(`🚫 Fehler: ${error.message}`);
      
      this.emit('pipelineFailed', { projectId, error, state: this.pipelineState });
      throw error;
    }
  }

  /**
   * Führt einen Worker-Agent aus
   */
  async executeWorkerAgent(agentKey, pipelineState) {
    const worker = this.workers[agentKey];
    if (!worker) {
      throw new Error(`Worker-Agent '${agentKey}' nicht gefunden`);
    }

    console.log(`   🤖 Führe Worker-Agent aus: ${getAgentConfig(agentKey).name}`);
    
    try {
      const result = await worker.execute(pipelineState);
      return result;
    } catch (error) {
      console.error(`   ❌ Worker-Agent ${agentKey} fehlgeschlagen:`, error.message);
      throw error;
    }
  }

  /**
   * Ermittelt das finale Ergebnis der Pipeline
   */
  getFinalResult() {
    // Priorität: reviewed_code > python_code > hyperparameter_suggestions
    if (this.pipelineState.results.CODE_REVIEWER) {
      return this.pipelineState.results.CODE_REVIEWER;
    }
    if (this.pipelineState.results.CODE_GENERATOR) {
      return this.pipelineState.results.CODE_GENERATOR;
    }
    if (this.pipelineState.results.HYPERPARAMETER_OPTIMIZER) {
      return this.pipelineState.results.HYPERPARAMETER_OPTIMIZER;
    }
    return null;
  }

  /**
   * Ruft den aktuellen Pipeline-Status ab
   */
  getPipelineStatus() {
    return {
      ...this.pipelineState,
      progress: {
        completed: this.pipelineState.completedSteps.length,
        total: PIPELINE_STEPS.length,
        percentage: Math.round((this.pipelineState.completedSteps.length / PIPELINE_STEPS.length) * 100)
      }
    };
  }

  /**
   * Ruft verfügbare Worker-Agents ab
   */
  getAvailableWorkers() {
    return Object.keys(this.workers).map(key => ({
      key,
      config: getAgentConfig(key),
      available: true
    }));
  }

  /**
   * Testet die Verbindung zu einem Worker-Agent
   */
  async testWorkerAgent(agentKey) {
    const worker = this.workers[agentKey];
    if (!worker) {
      return { success: false, error: `Worker-Agent '${agentKey}' nicht gefunden` };
    }

    try {
      const result = await worker.test();
      return { success: true, result };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }
}

/**
 * Globale Master-Agent-Instanz
 */
export const masterAgent = new MasterAgent();

/**
 * Wrapper-Funktion für Kompatibilität
 */
export async function runNetworkAgentPipeline(project) {
  return await masterAgent.runPipeline(project);
}
