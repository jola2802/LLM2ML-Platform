"""
Execution-Route-Handler
"""

from flask import request
from core.execution.code_exec import (
    execute_python_script,
    predict_with_model
)
from infrastructure.queue.job_queue import job_queue
from shared.utils.response import success_response, error_response, validation_error, not_found_error
from shared.utils.validators import (
    validate_training_request,
    validate_prediction_request,
    validate_execution_request
)
import os
import tempfile

def handle_train(worker_pool, script_dir, venv_dir, models_dir):
    """Handler für Training-Job"""
    try:
        data = request.get_json() or {}
        error = validate_training_request(data)
        if error:
            return validation_error(error)
        
        project_id = data['projectId']
        python_code = data['pythonCode']
        
        print(f'Training-Job gestartet für Projekt: {project_id}')
        
        job_id = worker_pool.add_training_job(project_id, python_code, 1)
        
        return success_response({
            'jobId': job_id,
            'projectId': project_id,
            'status': 'pending',
            'message': 'Training-Job zur Queue hinzugefügt'
        })
    except Exception as error:
        print(f'Fehler beim Starten des Training-Jobs: {error}')
        return error_response(f'Fehler beim Starten des Training-Jobs: {error}')

def handle_retrain(worker_pool, script_dir, venv_dir, models_dir):
    """Handler für Retraining-Job"""
    try:
        data = request.get_json() or {}
        error = validate_training_request(data)
        if error:
            return validation_error(error)
        
        project_id = data['projectId']
        python_code = data['pythonCode']
        
        print(f'Retraining-Job gestartet für Projekt: {project_id}')
        
        job_id = worker_pool.add_retraining_job(project_id, python_code, 1)
        
        return success_response({
            'jobId': job_id,
            'projectId': project_id,
            'status': 'pending',
            'message': 'Retraining-Job zur Queue hinzugefügt'
        })
    except Exception as error:
        print(f'Fehler beim Starten des Retraining-Jobs: {error}')
        return error_response(f'Fehler beim Starten des Retraining-Jobs: {error}')

def handle_execute(script_dir, venv_dir):
    """Handler für Code-Ausführung"""
    try:
        data = request.get_json() or {}
        error = validate_execution_request(data)
        if error:
            return validation_error(error)
        
        code = data['code']
        project_id = data.get('projectId')
        
        script_path = os.path.join(script_dir, f'execute_{project_id or "temp"}.py')
        
        try:
            # Code in temporäre Datei schreiben
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Code ausführen
            result = execute_python_script(script_path, script_dir, venv_dir)
            
            # Temporäre Datei löschen
            try:
                os.remove(script_path)
            except Exception:
                pass
            
            return success_response({
                'output': result.get('stdout', ''),
                'stderr': result.get('stderr', '')
            })
        except Exception as error:
            # Versuche temporäre Datei zu löschen
            try:
                os.remove(script_path)
            except Exception:
                pass
            raise
    except Exception as error:
        print(f'Fehler bei Code-Ausführung: {error}')
        return error_response(f'Fehler bei Code-Ausführung: {error}')

def handle_get_job_status(job_id: str):
    """Handler für Job-Status"""
    try:
        job = job_queue.get_job(job_id)
        
        if not job:
            return not_found_error('Job')
        
        return success_response({
            'id': job.id,
            'type': job.type,
            'status': job.status,
            'createdAt': job.created_at,
            'startedAt': job.started_at,
            'completedAt': job.completed_at,
            'error': job.error,
            'result': job.result
        })
    except Exception as error:
        print(f'Fehler beim Abfragen des Job-Status: {error}')
        return error_response(f'Fehler beim Abfragen des Job-Status: {error}')

def handle_get_execution_status(worker_pool):
    """Handler für Execution-Status"""
    try:
        pool_status = worker_pool.get_pool_status()
        queue_status = job_queue.get_queue_status()
        
        return success_response({
            'pool': pool_status,
            'queue': queue_status
        })
    except Exception as error:
        print(f'Fehler beim Abfragen des Worker-Status: {error}')
        return error_response(f'Fehler beim Abfragen des Worker-Status: {error}')

def handle_predict(script_dir, venv_dir, models_dir):
    """Handler für Prediction"""
    try:
        data = request.get_json() or {}
        error = validate_prediction_request(data)
        if error:
            return validation_error(error)
        
        project = data['project']
        input_features = data['inputFeatures']
        
        print(f'Prediction gestartet für Projekt: {project.get("id")}')
        
        prediction = predict_with_model(project, input_features, script_dir, venv_dir, models_dir)
        
        return success_response({'prediction': prediction})
    except Exception as error:
        print(f'Fehler bei Prediction: {error}')
        return error_response(f'Fehler bei Prediction: {error}')

