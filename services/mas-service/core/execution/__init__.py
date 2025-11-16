"""
Execution Package
"""

from core.execution.code_executor import execute_python_script
from core.execution.metrics_extractor import extract_metrics_from_output
from core.execution.prediction import predict_with_model, generate_prediction_script

__all__ = [
    'execute_python_script',
    'extract_metrics_from_output',
    'predict_with_model',
    'generate_prediction_script'
]

