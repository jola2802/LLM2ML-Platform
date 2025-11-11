"""
LLM-Routen für Flask
"""

from flask import Blueprint
from api.handlers import llm_handlers

llm_bp = Blueprint('llm', __name__)

def setup_llm_routes(app):
    """Registriere LLM-Routen"""
    
    # Test & Config
    app.add_url_rule('/api/llm/test', 'test_llm', llm_handlers.handle_llm_test, methods=['GET'])
    app.add_url_rule('/api/llm/config', 'get_llm_config', llm_handlers.handle_get_config, methods=['GET'])
    
    # Ollama-spezifische Endpoints
    app.add_url_rule('/api/llm/ollama/models', 'get_ollama_models', llm_handlers.handle_get_ollama_models, methods=['GET'])
    app.add_url_rule('/api/llm/ollama/test', 'test_ollama', llm_handlers.handle_test_ollama, methods=['POST'])
    app.add_url_rule('/api/llm/ollama/config', 'update_ollama_config', llm_handlers.handle_update_ollama_config, methods=['POST'])
    
    # Status
    app.add_url_rule('/api/llm/status', 'get_llm_status', llm_handlers.handle_get_llm_status, methods=['GET'])
    
    # LLM-API-Calls
    app.add_url_rule('/api/llm/call', 'call_llm', llm_handlers.handle_call_llm, methods=['POST'])
    app.add_url_rule('/api/llm/recommendations', 'get_recommendations', llm_handlers.handle_get_recommendations, methods=['POST'])
    app.add_url_rule('/api/llm/feature-engineering', 'feature_engineering', llm_handlers.handle_feature_engineering, methods=['POST'])
    app.add_url_rule('/api/llm/evaluate-performance', 'evaluate_performance', llm_handlers.handle_evaluate_performance, methods=['POST'])
