"""
Execution-Routen für Flask
"""

from flask import Blueprint
from api.handlers import execution_handlers

execution_bp = Blueprint('execution', __name__)

def setup_execution_routes(app, worker_pool, script_dir, venv_dir, models_dir):
    """Registriere Execution-Routen"""
    
    # Training & Retraining
    app.add_url_rule(
        '/api/execution/train',
        'train',
        lambda: execution_handlers.handle_train(worker_pool, script_dir, venv_dir, models_dir),
        methods=['POST']
    )
    app.add_url_rule(
        '/api/execution/retrain',
        'retrain',
        lambda: execution_handlers.handle_retrain(worker_pool, script_dir, venv_dir, models_dir),
        methods=['POST']
    )
    
    # Code Execution
    app.add_url_rule(
        '/api/execution/execute',
        'execute',
        lambda: execution_handlers.handle_execute(script_dir, venv_dir),
        methods=['POST']
    )
    
    # Job Management
    app.add_url_rule(
        '/api/execution/jobs/<job_id>',
        'get_job_status',
        execution_handlers.handle_get_job_status,
        methods=['GET']
    )
    app.add_url_rule(
        '/api/execution/status',
        'get_execution_status',
        lambda: execution_handlers.handle_get_execution_status(worker_pool),
        methods=['GET']
    )
    
    # Prediction
    app.add_url_rule(
        '/api/execution/predict',
        'predict',
        lambda: execution_handlers.handle_predict(script_dir, venv_dir, models_dir),
        methods=['POST']
    )

