"""
Agent-Routen für Flask
"""

from flask import Blueprint
from api.handlers import agent_handlers

agent_bp = Blueprint('agents', __name__)

def setup_agent_routes(app, worker_pool):
    """Registriere Agent-Routen"""
    
    # Agent-Management
    app.add_url_rule('/api/agents', 'get_all_agents', agent_handlers.handle_get_all_agents, methods=['GET'])
    app.add_url_rule('/api/agents/<agent_key>', 'get_agent_config', agent_handlers.handle_get_agent_config, methods=['GET'])
    
    # Pipeline - mit Worker Pool für direktes Training
    app.add_url_rule(
        '/api/agents/pipeline/run',
        'run_pipeline',
        lambda: agent_handlers.handle_run_pipeline(worker_pool),
        methods=['POST']
    )
    app.add_url_rule('/api/agents/pipeline/status', 'get_pipeline_status', agent_handlers.handle_get_pipeline_status, methods=['GET'])
    
    # Worker-Test
    app.add_url_rule('/api/agents/worker/test/<agent_key>', 'test_worker', agent_handlers.handle_test_worker, methods=['POST'])
