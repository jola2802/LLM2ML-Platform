#!/usr/bin/env python3
"""
Automatische Datenexploration Service - Pareto-optimiert
Generiert kompakte, relevante Datenübersichten für LLM-Analyse
Fokussiert auf die wichtigsten 20% der Informationen
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Dateipfad wird von JavaScript eingefügt (als Raw-String)
FILE_PATH = r"{file_path}"

# Sichere JSON-Serialisierung für numpy/pandas Objekte
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        return super().default(obj)

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

# Ursprüngliche Funktionen für createLLMSummary
def analyze_data_types(df):
    """Analysiert Datentypen und gibt detaillierte Informationen zurück"""
    type_analysis = {}
    
    for column in df.columns:
        try:
            col_data = df[column]
            dtype = str(col_data.dtype)
            
            # Basis-Informationen
            info = {
                'dtype': dtype,
                'unique_count': safe_convert(col_data.nunique()),
                'missing_count': safe_convert(col_data.isnull().sum()),
                'missing_percentage': safe_convert((col_data.isnull().sum() / len(df)) * 100),
                'is_numeric': pd.api.types.is_numeric_dtype(col_data),
                'is_categorical': pd.api.types.is_categorical_dtype(col_data),
                'is_datetime': pd.api.types.is_datetime64_any_dtype(col_data),
                'is_object': col_data.dtype == 'object'
            }
            
            # Numerische Spalten
            if info['is_numeric']:
                try:
                    info.update({
                        'min': safe_convert(col_data.min()),
                        'max': safe_convert(col_data.max()),
                        'mean': safe_convert(col_data.mean()),
                        'median': safe_convert(col_data.median()),
                        'std': safe_convert(col_data.std()),
                        'q25': safe_convert(col_data.quantile(0.25)),
                        'q75': safe_convert(col_data.quantile(0.75)),
                        'skewness': safe_convert(col_data.skew()),
                        'kurtosis': safe_convert(col_data.kurtosis())
                    })
                except Exception as e:
                    info['numeric_error'] = str(e)
            
            # Kategorische/Objekt Spalten
            if info['is_categorical'] or info['is_object'] or (not info['is_numeric'] and info['unique_count'] < 100):
                try:
                    value_counts = col_data.value_counts()
                    info.update({
                        'top_values': {str(k): safe_convert(v) for k, v in value_counts.head(10).items()},
                        'value_distribution': {str(k): safe_convert(v) for k, v in (value_counts / len(df) * 100).head(10).items()}
                    })
                except Exception as e:
                    info['categorical_error'] = str(e)
            
            # Datetime Spalten
            if info['is_datetime']:
                try:
                    info.update({
                        'min_date': safe_convert(col_data.min()),
                        'max_date': safe_convert(col_data.max()),
                        'date_range_days': safe_convert((col_data.max() - col_data.min()).days)
                    })
                except Exception as e:
                    info['datetime_error'] = str(e)
            
            type_analysis[column] = info
            
        except Exception as e:
            type_analysis[column] = {
                'error': str(e),
                'dtype': str(df[column].dtype) if hasattr(df[column], 'dtype') else 'unknown'
            }
    
    return type_analysis

def detect_correlations(df):
    """Erkennt Korrelationen zwischen numerischen Spalten"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = df[numeric_cols].corr()
        
        # Finde starke Korrelationen (|r| > 0.7)
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                try:
                    corr_value = corr_matrix.iloc[i, j]
                    if not pd.isna(corr_value) and abs(corr_value) > 0.7:
                        strong_correlations.append({
                            'column1': str(corr_matrix.columns[i]),
                            'column2': str(corr_matrix.columns[j]),
                            'correlation': safe_convert(corr_value)
                        })
                except Exception as e:
                    continue
        
        return {
            'strong_correlations': strong_correlations,
            'correlation_matrix': {str(k): {str(k2): safe_convert(v2) for k2, v2 in v.items()} for k, v in corr_matrix.to_dict().items()}
        }
    except Exception as e:
        return {'error': str(e)}

def analyze_missing_values(df):
    """Analysiert fehlende Werte"""
    try:
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        return {
            'columns_with_missing': {str(k): safe_convert(v) for k, v in missing_data[missing_data > 0].items()},
            'missing_percentages': {str(k): safe_convert(v) for k, v in missing_percentage[missing_percentage > 0].items()},
            'total_missing_cells': safe_convert(missing_data.sum()),
            'total_missing_percentage': safe_convert((missing_data.sum() / (len(df) * len(df.columns))) * 100)
        }
    except Exception as e:
        return {'error': str(e)}

def detect_outliers(df):
    """Erkennt Ausreißer in numerischen Spalten"""
    outliers = {}
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_cols:
            try:
                col_data = df[column].dropna()
                if len(col_data) == 0:
                    continue
                    
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR == 0:
                    continue
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = len(col_data[(col_data < lower_bound) | (col_data > upper_bound)])
                
                if outliers_count > 0:
                    outliers[str(column)] = {
                        'count': safe_convert(outliers_count),
                        'percentage': safe_convert((outliers_count / len(col_data)) * 100),
                        'lower_bound': safe_convert(lower_bound),
                        'upper_bound': safe_convert(upper_bound)
                    }
            except Exception as e:
                outliers[str(column)] = {'error': str(e)}
        
        return outliers
    except Exception as e:
        return {'error': str(e)}

def generate_sample_data(df, max_rows=10):
    """Generiert Beispieldaten für LLM-Analyse"""
    sample_data = []
    
    try:
        # Erste Zeilen
        first_rows = df.head(max_rows//2)
        for idx, row in first_rows.iterrows():
            try:
                sample_data.append({
                    'row_index': safe_convert(idx),
                    'data': {str(k): safe_convert(v) for k, v in row.to_dict().items()}
                })
            except Exception as e:
                continue
        
        # Letzte Zeilen
        last_rows = df.tail(max_rows//2)
        for idx, row in last_rows.iterrows():
            try:
                sample_data.append({
                    'row_index': safe_convert(idx),
                    'data': {str(k): safe_convert(v) for k, v in row.to_dict().items()}
                })
            except Exception as e:
                continue
        
        return sample_data
    except Exception as e:
        return [{'error': str(e)}]

# Pareto-optimierte Funktionen
def get_column_importance_score(df, column):
    """Bewertet die Wichtigkeit einer Spalte basierend auf verschiedenen Faktoren"""
    try:
        col_data = df[column]
        score = 0
        
        # Numerische Spalten sind wichtiger
        if pd.api.types.is_numeric_dtype(col_data):
            score += 3
            
            # Spalten mit hoher Varianz sind interessanter
            if col_data.std() > col_data.mean() * 0.5:
                score += 2
                
            # Spalten mit Ausreißern sind wichtig
            Q1, Q3 = col_data.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            if IQR > 0:
                outliers = len(col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)])
                if outliers > len(col_data) * 0.05:  # >5% Ausreißer
                    score += 2
        
        # Kategorische Spalten mit vielen verschiedenen Werten
        elif col_data.dtype == 'object':
            unique_ratio = col_data.nunique() / len(col_data)
            if 0.01 < unique_ratio < 0.5:  # Nicht zu viele, nicht zu wenige Kategorien
                score += 2
        
        # Spalten mit fehlenden Werten sind wichtig zu erwähnen
        missing_ratio = col_data.isnull().sum() / len(col_data)
        if missing_ratio > 0.1:  # >10% fehlende Werte
            score += 1
        
        # Datetime Spalten sind oft wichtig
        if pd.api.types.is_datetime64_any_dtype(col_data):
            score += 2
        
        return score
    except:
        return 0

def analyze_key_columns(df, max_columns=5):
    """Analysiert nur die wichtigsten Spalten (Pareto-Prinzip)"""
    # Bewerte alle Spalten
    column_scores = {col: get_column_importance_score(df, col) for col in df.columns}
    
    # Wähle die Top-Spalten
    top_columns = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)[:max_columns]
    
    key_analysis = {}
    for column, score in top_columns:
        try:
            col_data = df[column]
            info = {
                'dtype': str(col_data.dtype),
                'missing_pct': round((col_data.isnull().sum() / len(df)) * 100, 1),
                'unique_count': col_data.nunique()
            }
            
            # Numerische Spalten
            if pd.api.types.is_numeric_dtype(col_data):
                info.update({
                    'min': round(float(col_data.min()), 2),
                    'max': round(float(col_data.max()), 2),
                    'mean': round(float(col_data.mean()), 2),
                    'std': round(float(col_data.std()), 2)
                })
                
                # Ausreißer-Info
                Q1, Q3 = col_data.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                if IQR > 0:
                    outliers = len(col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)])
                    if outliers > 0:
                        info['outliers_pct'] = round((outliers / len(col_data)) * 100, 1)
            
            # Kategorische Spalten
            elif col_data.dtype == 'object' and col_data.nunique() < 20:
                top_values = col_data.value_counts().head(3)
                info['top_values'] = {str(k): int(v) for k, v in top_values.items()}
            
            key_analysis[column] = info
            
        except Exception as e:
            key_analysis[column] = {'error': str(e)}
    
    return key_analysis

def detect_strong_correlations(df, threshold=0.7):
    """Findet nur starke Korrelationen (reduziert Noise)"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return []
        
        corr_matrix = df[numeric_cols].corr()
        strong_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if not pd.isna(corr_value) and abs(corr_value) > threshold:
                    strong_correlations.append({
                        'columns': [str(corr_matrix.columns[i]), str(corr_matrix.columns[j])],
                        'correlation': round(float(corr_value), 3)
                    })
        
        # Sortiere nach Stärke der Korrelation
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        return strong_correlations[:5]  # Nur Top 5 Korrelationen
        
    except Exception as e:
        return []

def get_data_quality_issues(df):
    """Identifiziert wichtige Datenqualitätsprobleme"""
    issues = []
    
    try:
        # Fehlende Werte
        missing_data = df.isnull().sum()
        high_missing_cols = missing_data[missing_data > len(df) * 0.1]  # >10% fehlend
        if len(high_missing_cols) > 0:
            issues.append({
                'type': 'high_missing_values',
                'columns': list(high_missing_cols.index),
                'missing_pct': round((high_missing_cols.sum() / (len(df) * len(df.columns))) * 100, 1)
            })
        
        # Duplikate
        duplicate_pct = (df.duplicated().sum() / len(df)) * 100
        if duplicate_pct > 5:
            issues.append({
                'type': 'duplicates',
                'percentage': round(duplicate_pct, 1)
            })
        
        # Spalten mit nur einem Wert (konstante Spalten)
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() == 1:
                constant_cols.append(col)
        
        if constant_cols:
            issues.append({
                'type': 'constant_columns',
                'columns': constant_cols
            })
        
        return issues
        
    except Exception as e:
        return [{'type': 'analysis_error', 'error': str(e)}]

def generate_compact_sample(df, max_rows=3):
    """Generiert kompakte Beispieldaten"""
    try:
        if len(df) == 0:
            return []
        
        # Erste Zeile
        first_row = df.iloc[0].to_dict()
        sample = [{'position': 'first', 'data': {str(k): safe_convert(v) for k, v in first_row.items()}}]
        
        # Mittlere Zeile (falls vorhanden)
        if len(df) > 2:
            middle_idx = len(df) // 2
            middle_row = df.iloc[middle_idx].to_dict()
            sample.append({'position': 'middle', 'data': {str(k): safe_convert(v) for k, v in middle_row.items()}})
        
        # Letzte Zeile
        if len(df) > 1:
            last_row = df.iloc[-1].to_dict()
            sample.append({'position': 'last', 'data': {str(k): safe_convert(v) for k, v in last_row.items()}})
        
        return sample
        
    except Exception as e:
        return [{'error': str(e)}]

def load_data_safely(file_path):
    """Lädt Daten sicher mit verschiedenen Methoden"""
    file_extension = Path(file_path).suffix.lower()
    
    try:
        if file_extension == '.csv':
            # Versuche verschiedene Encoding-Optionen
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    if len(df.columns) > 1:  # Mindestens 2 Spalten für gültige CSV
                        return df
                except Exception:
                    continue
            
            # Fallback: Verwende pandas mit automatischer Erkennung
            return pd.read_csv(file_path, encoding=None)
            
        elif file_extension in ['.xlsx', '.xls']:
            try:
                return pd.read_excel(file_path)
            except Exception as e:
                # Fallback: Versuche mit openpyxl
                return pd.read_excel(file_path, engine='openpyxl')
                
        elif file_extension == '.json':
            try:
                return pd.read_json(file_path)
            except Exception as e:
                # Fallback: Versuche mit lines=True für JSONL
                return pd.read_json(file_path, lines=True)
                
        elif file_extension == '.parquet':
            return pd.read_parquet(file_path)
            
        elif file_extension == '.feather':
            return pd.read_feather(file_path)
            
        elif file_extension == '.pickle' or file_extension == '.pkl':
            return pd.read_pickle(file_path)
            
        elif file_extension == '.txt':
            # Versuche als CSV zu lesen, dann als Text
            try:
                return pd.read_csv(file_path, sep=None, engine='python')
            except:
                # Als Text-Datei behandeln
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                return pd.DataFrame({'text': lines})
                
        else:
            # Generischer Ansatz: Versuche verschiedene Methoden
            try:
                return pd.read_csv(file_path, sep=None, engine='python')
            except:
                try:
                    return pd.read_excel(file_path)
                except:
                    # Als Text behandeln
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return pd.DataFrame({'content': [content]})
                    
    except Exception as e:
        raise ValueError(f"Konnte Datei nicht laden: {str(e)}")

def create_data_overview(file_path):
    """Erstellt eine umfassende Datenübersicht mit allen ursprünglichen Objekten"""
    try:
        # Datei laden
        df = load_data_safely(file_path)
        
        # Vollständige Übersicht mit allen ursprünglichen Objekten
        overview = {
            'file_info': {
                'file_path': str(file_path),
                'file_size_mb': safe_convert(round(os.path.getsize(file_path) / (1024 * 1024), 2)),
                'file_extension': Path(file_path).suffix.lower()
            },
            'dataset_info': {
                'rows': safe_convert(len(df)),
                'columns': safe_convert(len(df.columns)),
                'memory_usage_mb': safe_convert(round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)),
                'duplicate_rows': safe_convert(df.duplicated().sum()),
                'duplicate_percentage': safe_convert((df.duplicated().sum() / len(df)) * 100)
            },
            'columns': [str(col) for col in df.columns],
            'detailed_analysis': analyze_data_types(df),
            'missing_values': analyze_missing_values(df),
            'correlations': detect_correlations(df),
            'outliers': detect_outliers(df),
            'sample_data': generate_sample_data(df, 10)
        }
        
        return overview
        
    except Exception as e:
        return {
            'error': str(e),
            'file_path': str(file_path),
            'traceback': str(sys.exc_info())
        }

def create_pareto_data_overview(file_path):
    """Erstellt eine Pareto-optimierte, kompakte Datenübersicht"""
    try:
        # Datei laden
        df = load_data_safely(file_path)
        
        # Kompakte Übersicht mit nur den wichtigsten Informationen
        overview = {
            'summary': {
                'rows': len(df),
                'columns': len(df.columns),
                'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
                'memory_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            },
            'key_columns': analyze_key_columns(df, max_columns=5),  # Nur Top 5 Spalten
            'strong_correlations': detect_strong_correlations(df, threshold=0.7),  # Nur starke Korrelationen
            'data_quality_issues': get_data_quality_issues(df),  # Nur wichtige Probleme
            'sample_data': generate_compact_sample(df, max_rows=3)  # Nur 3 Zeilen
        }
        
        return overview
        
    except Exception as e:
        return {
            'error': str(e),
            'file_path': str(file_path)
        }

def main():
    """Hauptfunktion für Kommandozeilen-Aufruf"""
    # Verwende den von JavaScript eingefügten Dateipfad
    file_path = FILE_PATH
    
    if not os.path.exists(file_path):
        print(f"Datei nicht gefunden: {file_path}")
        sys.exit(1)
    
    try:
        # Erstelle beide Übersichten: vollständig und Pareto-optimiert
        full_overview = create_data_overview(file_path)
        pareto_overview = create_pareto_data_overview(file_path)
        
        # Kombiniere beide in einem Ergebnis
        result = {
            'full_analysis': full_overview,  # Für createLLMSummary
            'pareto_analysis': pareto_overview  # Für LLM-Token-Optimierung
        }
        
        # Verwende sichere JSON-Serialisierung
        print(json.dumps(result, indent=2, ensure_ascii=False, cls=SafeJSONEncoder))
    except Exception as e:
        error_result = {
            'error': str(e),
            'file_path': file_path
        }
        print(json.dumps(error_result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main() 