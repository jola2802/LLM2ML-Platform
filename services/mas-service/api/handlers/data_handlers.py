"""
Data-Route-Handler
"""

from flask import request
from core.data.data_exploration import (
    perform_data_exploration,
    get_cached_data_analysis,
    clear_analysis_cache,
    get_analysis_cache_status
)
from shared.utils.response import success_response, error_response, validation_error, not_found_error
from shared.utils.validators import validate_file_path

def handle_data_explore():
    """Handler für Data Exploration"""
    try:
        data = request.get_json() or {}
        error = validate_file_path(data.get('filePath', ''))
        if error:
            return validation_error(error)
        
        file_path = data['filePath']
        print(f'Data Exploration gestartet für: {file_path}')
        
        exploration_result = perform_data_exploration(file_path)
        
        return success_response({
            'exploration': exploration_result,
            'filePath': file_path
        })
    except Exception as error:
        print(f'Fehler bei Data Exploration: {error}')
        return error_response(f'Fehler bei Data Exploration: {error}')

def handle_data_analyze():
    """Handler für Data Analysis (mit LLM-Zusammenfassung)"""
    try:
        data = request.get_json() or {}
        error = validate_file_path(data.get('filePath', ''))
        if error:
            return validation_error(error)
        
        file_path = data['filePath']
        force_refresh = data.get('forceRefresh', False)
        
        print(f'Data Analysis gestartet für: {file_path} (forceRefresh: {force_refresh})')
        
        analysis_result = get_cached_data_analysis(file_path, force_refresh)
        
        return success_response(analysis_result)
    except Exception as error:
        print(f'Fehler bei Data Analysis: {error}')
        return error_response(f'Fehler bei Data Analysis: {error}')

def handle_cache_status():
    """Handler für Cache-Status"""
    try:
        status = get_analysis_cache_status()
        return success_response(status)
    except Exception as error:
        return error_response(f'Fehler beim Abrufen des Cache-Status: {error}')

def handle_cache_clear():
    """Handler für Cache leeren"""
    try:
        clear_analysis_cache()
        return success_response({'message': 'Cache geleert'})
    except Exception as error:
        return error_response(f'Fehler beim Leeren des Cache: {error}')

