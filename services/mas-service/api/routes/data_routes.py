"""
Data-Routen für Flask
"""

from flask import Blueprint
from api.handlers import data_handlers

data_bp = Blueprint('data', __name__)

def setup_data_routes(app, venv_dir, uploads_dir):
    """Registriere Data-Routen"""
    
    app.add_url_rule('/api/data/explore', 'data_explore', data_handlers.handle_data_explore, methods=['POST'])
    app.add_url_rule('/api/data/analyze', 'data_analyze', data_handlers.handle_data_analyze, methods=['POST'])
    app.add_url_rule('/api/data/cache/status', 'cache_status', data_handlers.handle_cache_status, methods=['GET'])
    app.add_url_rule('/api/data/cache/clear', 'cache_clear', data_handlers.handle_cache_clear, methods=['POST'])

