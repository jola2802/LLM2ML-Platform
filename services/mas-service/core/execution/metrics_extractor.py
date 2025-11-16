"""
Metriken-Extraktion aus Python-Ausgabe
"""

import re
from typing import Dict, Any, Optional

def extract_metrics_from_output(output: str, model_type: str) -> Dict[str, Any]:
    """Extrahiert Metriken aus Python-Ausgabe"""
    metrics = {}
    found_metrics = set()
    
    # Metriken-Patterns
    metric_patterns = [
        {
            'primary_name': 'mean_absolute_error',
            'aliases': ['mae'],
            'regexes': [r'(?:MAE):\s*([\d.]+)']
        },
        {
            'primary_name': 'mean_squared_error',
            'aliases': ['mse'],
            'regexes': [r'(?:MSE):\s*([\d.]+)']
        },
        {
            'primary_name': 'root_mean_squared_error',
            'aliases': ['rmse'],
            'regexes': [r'(?:RMSE):\s*([\d.]+)']
        },
        {
            'primary_name': 'r_squared',
            'aliases': ['r2'],
            'regexes': [r'(?:R2|R²|R-squared):\s*([\d.]+)']
        },
        {
            'primary_name': 'accuracy',
            'aliases': [],
            'regexes': [r'(?:Accuracy):\s*([\d.]+)']
        },
        {
            'primary_name': 'precision',
            'aliases': [],
            'regexes': [r'(?:Precision):\s*([\d.]+)']
        },
        {
            'primary_name': 'recall',
            'aliases': [],
            'regexes': [r'(?:Recall):\s*([\d.]+)']
        },
        {
            'primary_name': 'f1_score',
            'aliases': ['f1'],
            'regexes': [r'(?:F1|F1-Score|F1 Score):\s*([\d.]+)']
        }
    ]
    
    # Durchsuche Output nach Metriken
    for pattern in metric_patterns:
        if pattern['primary_name'] in found_metrics:
            continue
        
        for regex_str in pattern['regexes']:
            match = re.search(regex_str, output, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    metrics[pattern['primary_name']] = value
                    found_metrics.add(pattern['primary_name'])
                    print(f'[Metrics Extractor] Gefundene Metrik: {pattern["primary_name"]} = {value}')
                    break
                except (ValueError, IndexError) as e:
                    print(f'[Metrics Extractor] Fehler beim Parsen von {pattern["primary_name"]}: {e}')
                    continue
    
    if not metrics:
        print(f'[Metrics Extractor] WARNUNG: Keine Metriken gefunden im Output (Länge: {len(output)} Zeichen)')
        print(f'[Metrics Extractor] Output-Ausschnitt (letzte 500 Zeichen): {output[-500:]}')
    
    return metrics

