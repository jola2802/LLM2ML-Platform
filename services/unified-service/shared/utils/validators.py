"""
Validierungs-Funktionen
"""

from typing import Optional, Dict, Any, List
import os

def validate_required(data: Dict[str, Any], fields: List[str]) -> Optional[str]:
    """Validiert, dass alle erforderlichen Felder vorhanden sind"""
    if not data:
        return 'Request-Body fehlt'
    missing = [field for field in fields if field not in data or data[field] is None]
    if missing:
        return f'Erforderliche Felder fehlen: {", ".join(missing)}'
    return None

def validate_file_path(file_path: str) -> Optional[str]:
    """Validiert, dass eine Datei existiert"""
    if not file_path:
        return 'filePath ist erforderlich'
    if not os.path.exists(file_path):
        return f'Datei nicht gefunden: {file_path}'
    return None

def validate_training_request(data: Dict[str, Any]) -> Optional[str]:
    """Validiert einen Training-Request"""
    return validate_required(data, ['projectId', 'pythonCode'])

def validate_prediction_request(data: Dict[str, Any]) -> Optional[str]:
    """Validiert einen Prediction-Request"""
    return validate_required(data, ['project', 'inputFeatures'])

def validate_execution_request(data: Dict[str, Any]) -> Optional[str]:
    """Validiert einen Code-Execution-Request"""
    return validate_required(data, ['code'])

def validate_prompt_request(data: Dict[str, Any]) -> Optional[str]:
    """Validiert einen Prompt-Request"""
    return validate_required(data, ['prompt'])

def validate_analysis_request(data: Dict[str, Any]) -> Optional[str]:
    """Validiert einen Analysis-Request"""
    return validate_required(data, ['analysis'])

def validate_project_request(data: Dict[str, Any]) -> Optional[str]:
    """Validiert einen Project-Request"""
    return validate_required(data, ['project'])

