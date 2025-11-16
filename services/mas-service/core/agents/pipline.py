"""
Einfache sequenzielle Pipeline ohne Master-Agent
"""

import asyncio
from typing import Dict, Any
from core.agents.config_agent_network import PIPELINE_STEPS, get_agent_config, log_agent_call

async def run_simple_pipeline(project: Dict[str, Any]) -> str:
    """Führt eine sequenzielle Pipeline aus"""
    project_id = project.get('id') or project.get('name', 'unknown')
    print(f'\n🚀 === STARTE SEQUENZIELLE PIPELINE für {project.get("name")} ===')
    
    # Pipeline-Status initialisieren
    pipeline_state = {
        'project': project,
        'completedSteps': [],
        'results': {},
        'errors': []
    }
    
    try:
        # Führe alle Pipeline-Schritte nacheinander aus
        for step in PIPELINE_STEPS:
            print(f'\n📍 SCHRITT {step["step"]}: {step["name"]}')
            print(f'   Agent: {get_agent_config(step["agent"])["name"]}')
            
            agent_key = step['agent']
            
            # Dynamischer Import der Worker
            try:
                if agent_key == 'DATA_ANALYZER':
                    from core.agents.data_analyzer_agent import DataAnalyzerWorker
                    worker = DataAnalyzerWorker()
                elif agent_key == 'HYPERPARAMETER_OPTIMIZER':
                    from core.agents.hyperparameter_optimizer_agent import HyperparameterOptimizerWorker
                    worker = HyperparameterOptimizerWorker()
                elif agent_key == 'CODE_GENERATOR':
                    from core.agents.code_generator_agent import CodeGeneratorWorker
                    worker = CodeGeneratorWorker()
                elif agent_key == 'CODE_REVIEWER':
                    from core.agents.code_reviewer_agent import CodeReviewerWorker
                    worker = CodeReviewerWorker()
                elif agent_key == 'PERFORMANCE_ANALYZER':
                    from core.agents.performance_analyzer_agent import PerformanceAnalyzerWorker
                    worker = PerformanceAnalyzerWorker()
                else:
                    raise ValueError(f'Worker-Agent "{agent_key}" nicht gefunden')
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
                
                if result:
                    pipeline_state['completedSteps'].append(step)
                    pipeline_state['results'][agent_key] = result
                    print(f'✅ {step["name"]} erfolgreich abgeschlossen')
                else:
                    error_msg = f'{step["name"]} fehlgeschlagen - kein Ergebnis erhalten'
                    pipeline_state['errors'].append(error_msg)
                    
                    if step['required']:
                        raise Exception(error_msg)
                    else:
                        print(f'⚠️ Optionaler Schritt {step["name"]} übersprungen')
                        pipeline_state['completedSteps'].append(step)
                        
            except Exception as error:
                print(f'❌ Fehler in Schritt {step["name"]}: {error}')
                pipeline_state['errors'].append(f'{step["name"]}: {error}')
                
                if step['required']:
                    raise
                else:
                    print(f'⚠️ Optionaler Schritt {step["name"]} übersprungen aufgrund Fehler')
                    pipeline_state['completedSteps'].append(step)
        
        # Pipeline erfolgreich abgeschlossen
        final_result = get_final_result(pipeline_state)
        print(f'\n✅ === PIPELINE ERFOLGREICH BEENDET ===')
        print(f'📝 Generierter Code: {len(final_result) if final_result else 0} Zeichen')
        
        if pipeline_state['errors']:
            print(f'⚠️ Warnungen: {len(pipeline_state["errors"])}')
        
        return final_result
        
    except Exception as error:
        print(f'\n❌ === PIPELINE FEHLGESCHLAGEN ===')
        print(f'🚫 Fehler: {error}')
        raise

def get_final_result(pipeline_state: Dict[str, Any]) -> str:
    """Ermittelt das finale Ergebnis der Pipeline"""
    # Priorität: reviewed_code > python_code > hyperparameter_suggestions
    if 'CODE_REVIEWER' in pipeline_state['results']:
        return pipeline_state['results']['CODE_REVIEWER']
    if 'CODE_GENERATOR' in pipeline_state['results']:
        return pipeline_state['results']['CODE_GENERATOR']
    if 'HYPERPARAMETER_OPTIMIZER' in pipeline_state['results']:
        return str(pipeline_state['results']['HYPERPARAMETER_OPTIMIZER'])
    return ''

