"""
Data Exploration Service - Python Implementation
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

# Cache für Data Analysis
_analysis_cache = {}

def safe_convert(value):
    """Sichere Konvertierung von numpy/pandas Werten"""
    if pd.isna(value):
        return None
    elif isinstance(value, (np.integer, np.int64)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, pd.Timestamp):
        return str(value)
    elif isinstance(value, pd.Series):
        return value.tolist()
    elif isinstance(value, pd.DataFrame):
        return value.to_dict()
    else:
        return value

def perform_data_exploration(file_path: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Führt automatische Datenexploration durch
    """
    try:
        # Lade Daten
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        else:
            # Versuche CSV als Fallback
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        
        # Basis-Informationen
        columns = df.columns.tolist()
        row_count = len(df)
        
        # Datentypen
        data_types = {}
        for col in columns:
            dtype = str(df[col].dtype)
            if 'int' in dtype or 'float' in dtype:
                data_types[col] = 'numeric'
            elif 'object' in dtype or 'string' in dtype:
                data_types[col] = 'categorical'
            else:
                data_types[col] = 'unknown'
        
        # Sample Data (erste 5 Zeilen)
        sample_data = df.head(5).values.tolist()
        
        # Statistische Informationen für numerische Spalten
        numeric_stats = {}
        for col in columns:
            if data_types[col] == 'numeric':
                numeric_stats[col] = {
                    'min': safe_convert(df[col].min()),
                    'max': safe_convert(df[col].max()),
                    'mean': safe_convert(df[col].mean()),
                    'median': safe_convert(df[col].median()),
                    'std': safe_convert(df[col].std()),
                    'missing': int(df[col].isnull().sum()),
                    'missing_percentage': float((df[col].isnull().sum() / row_count) * 100) if row_count > 0 else 0.0
                }
        
        # Kategorische Informationen
        categorical_stats = {}
        for col in columns:
            if data_types[col] == 'categorical':
                value_counts = df[col].value_counts().head(10)
                categorical_stats[col] = {
                    'unique_count': int(df[col].nunique()),
                    'top_values': {str(k): int(v) for k, v in value_counts.items()},
                    'missing': int(df[col].isnull().sum()),
                    'missing_percentage': float((df[col].isnull().sum() / row_count) * 100) if row_count > 0 else 0.0
                }
        
        # Erstelle Ergebnis
        result = {
            'columns': columns,
            'rowCount': row_count,
            'dataTypes': data_types,
            'sampleData': sample_data,
            'numericStats': numeric_stats,
            'categoricalStats': categorical_stats
        }
        
        return result
        
    except Exception as error:
        print(f'Fehler bei Datenexploration: {error}')
        print(f'File path: {file_path}')
        raise RuntimeError(f'Datenexploration fehlgeschlagen: {str(error)}')

def analyze_data_for_llm(file_path: str) -> Dict[str, Any]:
    """
    Analysiert Daten und erstellt LLM-Zusammenfassung
    """
    exploration_result = perform_data_exploration(file_path)
    
    # Erstelle LLM-Zusammenfassung (vereinfacht)
    llm_summary = _create_llm_summary(exploration_result)
    
    return {
        'success': True,
        'exploration': exploration_result,
        'llm_summary': llm_summary,
        'filePath': file_path
    }

def _create_llm_summary(exploration: Dict[str, Any]) -> str:
    """Erstellt eine LLM-Zusammenfassung aus Exploration-Ergebnissen"""
    # Vereinfachte Implementierung
    columns = exploration.get('columns', [])
    row_count = exploration.get('rowCount', 0)
    data_types = exploration.get('dataTypes', {})
    
    summary = f"Dataset mit {row_count} Zeilen und {len(columns)} Spalten.\n"
    summary += f"Spalten: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}\n"
    
    return summary

def get_cached_data_analysis(file_path: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Holt gecachte Data Analysis oder führt neue durch
    """
    cache_key = file_path
    
    if not force_refresh and cache_key in _analysis_cache:
        return _analysis_cache[cache_key]
    
    result = analyze_data_for_llm(file_path)
    _analysis_cache[cache_key] = result
    
    return result

def clear_analysis_cache():
    """Leert den Analysis-Cache"""
    global _analysis_cache
    _analysis_cache.clear()

def get_analysis_cache_status() -> Dict[str, Any]:
    """Gibt Cache-Status zurück"""
    return {
        'cachedFiles': len(_analysis_cache),
        'cacheKeys': list(_analysis_cache.keys())
    }
