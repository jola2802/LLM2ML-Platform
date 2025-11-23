"""
Pipeline mit Loop-Logik: Data Analyzer -> Feature Engineer -> Hyperparameter Optimizer -> 
Code Generator -> Code Executor -> Performance Analyzer -> Decision (Loop wenn nötig)
"""

import asyncio
from typing import Dict, Any
from core.agents.config_agent_network import PIPELINE_STEPS, get_agent_config, log_agent_call

async def run_simple_pipeline(project: Dict[str, Any], max_iterations: int = 3) -> str:
    """Führt eine sequenzielle Pipeline mit Loop-Logik aus"""
    project_id = project.get('id') or project.get('name', 'unknown')
    print(f'\n🚀 === STARTE PIPELINE für {project.get("name")} ===')
    
    # Pipeline-Status initialisieren
    pipeline_state = {
        'project': project,
        'completedSteps': [],
        'results': {},
        'errors': [],
        'iteration': 0,
        'maxIterations': max_iterations
    }
    
    try:
        iteration = 0
        should_continue = True
        
        while should_continue and iteration < max_iterations:
            iteration += 1
            pipeline_state['iteration'] = iteration
            
            print(f'\n🔄 === ITERATION {iteration}/{max_iterations} ===')
            
            # Führe alle Pipeline-Schritte nacheinander aus
            for step in PIPELINE_STEPS:
                print(f'\n📍 SCHRITT {step["step"]}: {step["name"]}')
                print(f'   Agent: {get_agent_config(step["agent"])["name"]}')
                
                agent_key = step['agent']
                
                # Dynamischer Import der Worker
                try:
                    worker = _get_worker(agent_key)
                except ImportError as e:
                    error_msg = f'Worker-Agent "{agent_key}" konnte nicht geladen werden: {e}'
                    print(f'❌ {error_msg}')
                    
                    if step['required']:
                        raise Exception(error_msg)
                    else:
                        print(f'⚠️ Optionaler Schritt {step["name"]} übersprungen')
                        continue
                
                log_agent_call(agent_key, get_agent_config(agent_key)['model'], step['name'])
                
                try:
                    # Worker-Agent ausführen
                    result = await worker.execute(pipeline_state)
                    
                    # Prüfe ob Ergebnis erfolgreich war
                    is_success = _check_step_success(result, agent_key)
                    
                    if result and is_success:
                        pipeline_state['completedSteps'].append(step)
                        pipeline_state['results'][agent_key] = result
                        print(f'✅ {step["name"]} erfolgreich abgeschlossen')
                    else:
                        error_msg = f'{step["name"]} fehlgeschlagen'
                        if result and isinstance(result, dict):
                            if not result.get('success', True):
                                error_msg += f': {result.get("error", "Unbekannter Fehler")}'
                        else:
                            error_msg += ': kein Ergebnis erhalten'
                        
                        pipeline_state['errors'].append(error_msg)
                        
                        if step['required']:
                            raise Exception(error_msg)
                        else:
                            print(f'⚠️ Optionaler Schritt {step["name"]} übersprungen')
                            pipeline_state['completedSteps'].append(step)
                            # Speichere auch fehlgeschlagenes Ergebnis für Debugging
                            if result:
                                pipeline_state['results'][agent_key] = result
                            
                except Exception as error:
                    print(f'❌ Fehler in Schritt {step["name"]}: {error}')
                    pipeline_state['errors'].append(f'{step["name"]}: {error}')
                    
                    if step['required']:
                        raise
                    else:
                        print(f'⚠️ Optionaler Schritt {step["name"]} übersprungen aufgrund Fehler')
                        pipeline_state['completedSteps'].append(step)
            
            # Prüfe Decision-Ergebnis (wenn Decision-Agent ausgeführt wurde)
            decision_result = pipeline_state['results'].get('DECISION', {})
            should_continue = decision_result.get('shouldContinue', False)
            
            if should_continue:
                print(f'\n🔄 Loop wird fortgesetzt - Grund: {decision_result.get("reason", "")}')
                # Bereite nächste Iteration vor (behalte Datenanalyse und Bereinigung, aber aktualisiere andere Schritte)
                # Entferne Ergebnisse die neu berechnet werden müssen
                keys_to_remove = ['FEATURE_ENGINEER', 'HYPERPARAMETER_OPTIMIZER', 'CODE_GENERATOR', 
                                 'CODE_REVIEWER', 'CODE_EXECUTOR', 'PERFORMANCE_ANALYZER', 'DECISION']
                for key in keys_to_remove:
                    pipeline_state['results'].pop(key, None)
            else:
                print(f'\n✅ Loop beendet - Grund: {decision_result.get("reason", "Ergebnis gut genug")}')
        
        # Pipeline erfolgreich abgeschlossen
        final_result = get_final_result(pipeline_state)
        print(f'\n✅ === PIPELINE ERFOLGREICH BEENDET ===')
        print(f'📝 Generierter Code: {len(final_result) if final_result else 0} Zeichen')
        print(f'🔄 Durchgeführte Iterationen: {iteration}')
        
        if pipeline_state['errors']:
            print(f'⚠️ Warnungen: {len(pipeline_state["errors"])}')
        
        return final_result
        
    except Exception as error:
        print(f'\n❌ === PIPELINE FEHLGESCHLAGEN ===')
        print(f'🚫 Fehler: {error}')
        raise

def _get_worker(agent_key: str):
    """Holt Worker-Instanz basierend auf Agent-Key"""
    if agent_key == 'DATA_ANALYZER':
        from core.agents.data_analyzer_agent import DataAnalyzerWorker
        return DataAnalyzerWorker()
    elif agent_key == 'DATA_CLEANER':
        from core.agents.data_cleaner_agent import DataCleanerWorker
        return DataCleanerWorker()
    elif agent_key == 'FEATURE_ENGINEER':
        from core.agents.feature_engineer_agent import FeatureEngineerWorker
        return FeatureEngineerWorker()
    elif agent_key == 'HYPERPARAMETER_OPTIMIZER':
        from core.agents.hyperparameter_optimizer_agent import HyperparameterOptimizerWorker
        return HyperparameterOptimizerWorker()
    elif agent_key == 'CODE_GENERATOR':
        from core.agents.code_generator_agent import CodeGeneratorWorker
        return CodeGeneratorWorker()
    elif agent_key == 'CODE_REVIEWER':
        from core.agents.code_reviewer_agent import CodeReviewerWorker
        return CodeReviewerWorker()
    elif agent_key == 'CODE_EXECUTOR':
        from core.agents.code_executor_agent import CodeExecutorWorker
        return CodeExecutorWorker()
    elif agent_key == 'PERFORMANCE_ANALYZER':
        from core.agents.performance_analyzer_agent import PerformanceAnalyzerWorker
        return PerformanceAnalyzerWorker()
    elif agent_key == 'DECISION':
        from core.agents.decision_agent import DecisionWorker
        return DecisionWorker()
    else:
        raise ValueError(f'Worker-Agent "{agent_key}" nicht gefunden')

def _check_step_success(result: Any, agent_key: str) -> bool:
    """Prüft ob ein Pipeline-Schritt erfolgreich war"""
    if not result:
        return False
    
    # Für Code-Executor: Prüfe success-Flag
    if agent_key == 'CODE_EXECUTOR':
        if isinstance(result, dict):
            return result.get('success', False)
    
    # Für Performance-Analyzer: Prüfe ob Score vorhanden
    if agent_key == 'PERFORMANCE_ANALYZER':
        if isinstance(result, dict):
            return 'overallScore' in result
    
    # Für Feature-Engineer: Prüfe ob Features generiert wurden
    if agent_key == 'FEATURE_ENGINEER':
        if isinstance(result, dict):
            features = result.get('generatedFeatures', [])
            # Auch wenn keine Features generiert wurden, ist es ein Erfolg (Fallback)
            return True
    
    # Für Hyperparameter-Optimizer: Prüfe ob Hyperparameter vorhanden
    if agent_key == 'HYPERPARAMETER_OPTIMIZER':
        if isinstance(result, dict):
            return 'hyperparameters' in result
    
    # Für andere Agents: Wenn Ergebnis vorhanden, ist es erfolgreich
    return True

def get_final_result(pipeline_state: Dict[str, Any]) -> str:
    """Ermittelt das finale Ergebnis der Pipeline"""
    # Priorität: generierter Code > Performance-Analyse
    if 'CODE_GENERATOR' in pipeline_state['results']:
        result = pipeline_state['results']['CODE_GENERATOR']
        if isinstance(result, str):
            return result
        return str(result)
    if 'PERFORMANCE_ANALYZER' in pipeline_state['results']:
        return str(pipeline_state['results']['PERFORMANCE_ANALYZER'])
    if 'HYPERPARAMETER_OPTIMIZER' in pipeline_state['results']:
        return str(pipeline_state['results']['HYPERPARAMETER_OPTIMIZER'])
    return ''

