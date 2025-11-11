"""
Queue-Route-Handler
"""

from flask import request
from core.llm import llm
from shared.utils.response import success_response, error_response, not_found_error

def handle_get_queue_status():
    """Handler für Queue-Status"""
    try:
        status = llm.get_llm_queue_status()
        return success_response({'status': status})
    except Exception as error:
        return error_response(f'Failed to get queue status: {error}')

def handle_cancel_request(request_id: int):
    """Handler für Request abbrechen"""
    try:
        data = request.get_json() or {}
        reason = data.get('reason', 'User cancelled')
        
        cancelled = llm.cancel_llm_request(request_id, reason)
        if cancelled:
            return success_response({
                'message': f'Request {request_id} cancelled',
                'requestId': request_id
            })
        else:
            return not_found_error(f'Request {request_id}')
    except Exception as error:
        return error_response(f'Failed to cancel request: {error}')

