"""
Agent-Route-Handler
"""

from flask import request
import asyncio
import time
from core.agents.config_agent_network import (
    ALL_AGENTS, get_agent_stats, get_agent_config,
    is_valid_agent, PIPELINE_STEPS, WORKER_AGENTS
)
from core.agents.pipline import run_simple_pipeline
from shared.utils.response import success_response, error_response, validation_error, not_found_error
from shared.utils.validators import validate_project_request

def handle_get_all_agents():
    """Handler für alle Agents abrufen"""
    try:
        agents = [{'key': key, **config} for key, config in ALL_AGENTS.items()]
        stats = get_agent_stats()
        return success_response({
            'agents': agents,
            'stats': stats,
            'totalAgents': len(agents)
        })
    except Exception as error:
        return error_response(f'Fehler beim Abrufen der Agents: {error}')

def handle_get_agent_config(agent_key: str):
    """Handler für Agent-Konfiguration abrufen"""
    try:
        if not is_valid_agent(agent_key):
            return not_found_error(f"Agent '{agent_key}'")
        
        config = get_agent_config(agent_key)
        return success_response({'agent': {'key': agent_key, **config}})
    except Exception as error:
        return error_response(f'Fehler beim Abrufen der Agent-Konfiguration: {error}')

def handle_run_pipeline(worker_pool):
    """Handler für Pipeline starten - generiert Code und startet Training direkt"""
    try:
        data = request.get_json()
        error = validate_project_request(data)
        if error:
            return validation_error(error)
        
        project = data['project']
        project_id = project.get('id')
        
        if not project_id:
            return validation_error('project.id ist erforderlich')
        
        print(f'🚀 Starte Pipeline für Projekt {project_id}')
        
        # 1. Pipeline ausführen - Code generieren
        python_code = asyncio.run(run_simple_pipeline(project))
        
        if not python_code or not isinstance(python_code, str) or len(python_code.strip()) == 0:
            return error_response('Pipeline hat keinen gültigen Code generiert')
        
        print(f'✅ Pipeline erfolgreich. Generierter Code: {len(python_code)} Zeichen')
        
        # 2. Training-Job direkt starten (ohne zurück ans Backend zu gehen)
        print(f'🏋️ Starte Training-Job direkt im unified-service für Projekt {project_id}')
        job_id = worker_pool.add_training_job(project_id, python_code, priority=1)
        
        print(f'✅ Training-Job {job_id} zur Queue hinzugefügt')
        
        return success_response({
            'result': python_code,
            'jobId': job_id,
            'projectId': project_id,
            'status': 'training_started',
            'message': 'Code generiert und Training-Job gestartet'
        })
    except Exception as error:
        print(f'❌ Pipeline-Fehler: {error}')
        import traceback
        print(f'Traceback: {traceback.format_exc()}')
        return error_response(f'Pipeline-Start fehlgeschlagen: {error}')

def handle_get_pipeline_status():
    """Handler für Pipeline-Status"""
    try:
        available_workers = [
            {'key': key, 'config': config, 'available': True}
            for key, config in WORKER_AGENTS.items()
        ]
        return success_response({
            'pipelineSteps': PIPELINE_STEPS,
            'availableWorkers': available_workers,
            'message': 'Sequenzielle Pipeline (ohne Master-Agent)'
        })
    except Exception as error:
        return error_response(f'Fehler beim Abrufen des Pipeline-Status: {error}')

def handle_test_worker(agent_key: str):
    """Handler für Worker-Test"""
    try:
        workers_map = {
            'DATA_ANALYZER': 'data_analyzer_agent',
            'HYPERPARAMETER_OPTIMIZER': 'hyperparameter_optimizer_agent',
            'CODE_GENERATOR': 'code_generator_agent',
            'CODE_REVIEWER': 'code_reviewer_agent',
            'PERFORMANCE_ANALYZER': 'performance_analyzer_agent'
        }
        
        if agent_key not in workers_map:
            return not_found_error(f"Worker-Agent '{agent_key}'")
        
        # Dynamischer Import
        module_name = f'llm.agents.{workers_map[agent_key]}'
        try:
            module = __import__(module_name, fromlist=[f'{agent_key}Worker'])
            worker_class = getattr(module, f'{agent_key}Worker')
            worker = worker_class()
            
            test_result = asyncio.run(worker.test())
            return success_response({
                'agentKey': agent_key,
                'testResult': test_result
            })
        except Exception as import_error:
            return error_response(f'Worker konnte nicht geladen werden: {import_error}')
    except Exception as error:
        return error_response(f'Fehler beim Testen des Worker-Agents: {error}')

