"""
Code-Execution Service (Hauptmodul)
"""

from .code_executor import execute_python_script
from .metrics_extractor import extract_metrics_from_output
from .prediction import predict_with_model, generate_prediction_script, convert_input_features

__all__ = [
    'execute_python_script',
    'extract_metrics_from_output',
    'predict_with_model',
    'generate_prediction_script',
    'convert_input_features'
]

