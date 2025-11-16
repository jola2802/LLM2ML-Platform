"""
LLM-Route-Handler
"""

from flask import request
from core.llm import llm
from core.llm import recommendations, feature_engineering, performance
from shared.utils.response import success_response, error_response, validation_error
from shared.utils.validators import validate_prompt_request, validate_analysis_request, validate_project_request, validate_required

def handle_llm_test():
    """Handler für LLM-Test"""
    try:
        return success_response({'message': 'LLM API funktioniert'})
    except Exception as error:
        return error_response(f'Test failed: {error}')

def handle_get_config():
    """Handler für LLM-Konfiguration abrufen"""
    try:
        config = llm.get_llm_config()
        return success_response({'config': config})
    except Exception as error:
        return error_response(f'Failed to get config: {error}')

def handle_get_ollama_models():
    """Handler für Ollama-Modelle abrufen"""
    try:
        result = llm.get_available_ollama_models()
        return success_response(result)
    except Exception as error:
        return error_response(f'Failed to get models: {error}')

def handle_test_ollama():
    """Handler für Ollama-Verbindungstest"""
    try:
        result = llm.test_ollama_connection()
        return success_response(result)
    except Exception as error:
        return error_response(f'Failed to test connection: {error}')

def handle_update_ollama_config():
    """Handler für Ollama-Konfiguration aktualisieren"""
    try:
        data = request.get_json()
        config = {}
        if 'host' in data:
            config['host'] = data['host']
        if 'defaultModel' in data:
            config['defaultModel'] = data['defaultModel']
        llm.update_ollama_config(config)
        return success_response({
            'message': 'Ollama-Konfiguration aktualisiert',
            'config': config
        })
    except Exception as error:
        return error_response(f'Failed to update config: {error}')

def handle_get_llm_status():
    """Handler für LLM-Status"""
    try:
        config = llm.get_llm_config()
        ollama_result = llm.test_ollama_connection()
        ollama_status = {
            'connected': ollama_result.get('connected', False),
            'available': ollama_result.get('success', False),
            'error': ollama_result.get('error'),
            'model': config['ollama']['defaultModel']
        }
        return success_response({
            'activeProvider': config['activeProvider'],
            'ollama': ollama_status
        })
    except Exception as error:
        return error_response(f'Failed to get status: {error}')

def handle_call_llm():
    """Handler für LLM-API-Call"""
    try:
        data = request.get_json()
        error = validate_prompt_request(data)
        if error:
            return validation_error(error)
        
        file_path = data.get('filePath')
        custom_model = data.get('customModel')
        max_retries = data.get('maxRetries', 3)
        
        result = llm.call_llm_api(data['prompt'], file_path, custom_model, max_retries)
        return success_response({'result': result})
    except Exception as error:
        return error_response(f'LLM-API-Call fehlgeschlagen: {error}')

def handle_get_recommendations():
    """Handler für LLM-Empfehlungen"""
    try:
        data = request.get_json() or {}
        error = validate_required(data, ['analysis'])
        if error:
            return validation_error(error)
        
        result = recommendations.get_llm_recommendations(
            data['analysis'],
            data.get('filePath'),
            data.get('selectedFeatures'),
            data.get('excludedFeatures'),
            data.get('userPreferences')
        )
        return success_response({'recommendations': result})
    except Exception as error:
        return error_response(f'LLM-Empfehlungen fehlgeschlagen: {error}')

def handle_feature_engineering():
    """Handler für Feature Engineering"""
    try:
        data = request.get_json()
        error = validate_analysis_request(data)
        if error:
            return validation_error(error)
        
        result = feature_engineering.get_feature_engineering_recommendations(
            data.get('analysis'),
            data.get('filePath'),
            data.get('selectedFeatures'),
            data.get('excludedFeatures') or data.get('excludedColumns') or [],
            data.get('userPreferences')
        )
        return success_response(result)
    except Exception as error:
        return error_response(f'Feature Engineering fehlgeschlagen: {error}')

def handle_evaluate_performance():
    """Handler für Performance-Evaluation"""
    try:
        data = request.get_json()
        error = validate_project_request(data)
        if error:
            return validation_error(error)
        
        result = performance.evaluate_performance_with_llm(data['project'])
        return success_response({'evaluation': result})
    except Exception as error:
        return error_response(f'Performance-Evaluation fehlgeschlagen: {error}')

