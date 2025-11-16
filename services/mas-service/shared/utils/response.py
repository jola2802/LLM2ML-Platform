"""
Response-Helper-Funktionen
"""

from flask import jsonify
from datetime import datetime
from typing import Any, Optional

def success_response(data: Any = None, message: Optional[str] = None) -> tuple:
    """Erstellt eine erfolgreiche JSON-Response"""
    response = {'success': True}
    if data is not None:
        if isinstance(data, dict):
            response.update(data)
        else:
            response['data'] = data
    if message:
        response['message'] = message
    response['timestamp'] = datetime.now().isoformat()
    return jsonify(response), 200

def error_response(error: str, status_code: int = 500) -> tuple:
    """Erstellt eine Fehler-JSON-Response"""
    return jsonify({
        'success': False,
        'error': error,
        'timestamp': datetime.now().isoformat()
    }), status_code

def validation_error(message: str) -> tuple:
    """Erstellt eine Validierungsfehler-Response"""
    return error_response(message, 400)

def not_found_error(resource: str) -> tuple:
    """Erstellt eine Not-Found-Response"""
    return error_response(f'{resource} nicht gefunden', 404)

