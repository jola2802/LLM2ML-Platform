"""
Queue-Routen für Flask
"""

from flask import Blueprint
from api.handlers import queue_handlers

queue_bp = Blueprint('queue', __name__)

def setup_queue_routes(app):
    """Registriere Queue-Routen"""
    
    app.add_url_rule('/api/llm/queue/status', 'get_queue_status', queue_handlers.handle_get_queue_status, methods=['GET'])
    app.add_url_rule('/api/llm/queue/cancel/<int:request_id>', 'cancel_request', queue_handlers.handle_cancel_request, methods=['POST'])
