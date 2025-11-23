"""
Python-Client für direkte Funktionsaufrufe
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from core.data.data_exploration import get_cached_data_analysis
from shared.utils.csv_manipulator import CSVManipulator, manipulate_csv

class PythonClient:
    """Client für direkte Funktionsaufrufe"""
    
    def analyze_data(self, file_path: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Data Analysis (mit LLM-Zusammenfassung) - Direkter Funktionsaufruf"""
        try:
            result = get_cached_data_analysis(file_path, force_refresh)
            return result
        except Exception as error:
            print(f'Fehler bei Data Analysis: {error}')
            raise Exception(f'Data Analysis fehlgeschlagen: {error}')
    
    def get_csv_manipulator(self, file_path: str) -> CSVManipulator:
        """
        Erstellt einen CSV-Manipulator für die angegebene Datei
        
        Args:
            file_path: Pfad zur CSV-Datei
        
        Returns:
            CSVManipulator-Instanz
        """
        return CSVManipulator(file_path)
    
    def manipulate_csv(
        self,
        file_path: str,
        operations: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        create_backup: bool = True
    ) -> str:
        """
        Führt CSV-Manipulationen aus (Convenience-Methode)
        
        Args:
            file_path: Pfad zur CSV-Datei
            operations: Liste von Operationen (siehe csv_manipulator.manipulate_csv)
            output_path: Pfad zum Speichern (wenn None, überschreibt Original)
            create_backup: Erstellt Backup der Originaldatei
        
        Returns:
            Pfad zur gespeicherten Datei
        """
        return manipulate_csv(file_path, operations, output_path, create_backup)
    
    def clean_data(
        self,
        file_path: str,
        operations: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Führt Datenbereinigung basierend auf Cleaning-Operationen aus
        
        Args:
            file_path: Pfad zur CSV-Datei
            operations: Liste von Cleaning-Operationen:
                - dropMissingRows: Entfernt Zeilen mit fehlenden Werten
                - fillMissing: Füllt fehlende Werte (method: mean, median, mode, constant)
                - dropColumn: Entfernt Spalten
                - removeOutliers: Entfernt Ausreißer (method: iqr, zscore)
                - encodeCategorial: Kodiert kategorische Variablen (method: onehot, label)
            output_path: Pfad zum Speichern (wenn None, überschreibt Original)
        
        Returns:
            Dictionary mit success, cleanedPath, operations, summary
        """
        try:
            # Lade CSV
            df = pd.read_csv(file_path)
            original_shape = df.shape
            applied_operations = []
            
            # Erstelle Backup
            if output_path is None:
                backup_dir = os.path.join(os.path.dirname(file_path), '.backups')
                os.makedirs(backup_dir, exist_ok=True)
                file_name = Path(file_path).stem
                file_ext = Path(file_path).suffix
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                backup_path = os.path.join(
                    backup_dir,
                    f"{file_name}_backup_{timestamp}{file_ext}"
                )
                df.to_csv(backup_path, index=False)
            
            # Führe Operationen aus
            for op in operations:
                op_type = op.get('type', '').lower()
                columns = op.get('columns', [])
                
                if op_type == 'dropmissingrows':
                    # Entferne Zeilen mit fehlenden Werten
                    threshold = op.get('threshold', 1.0)
                    if threshold < 1.0:
                        # Entferne Zeilen wenn mehr als threshold% fehlende Werte
                        max_missing = int(len(columns) * threshold) if columns else len(df.columns) * threshold
                        df = df.dropna(subset=columns if columns else None, thresh=len(df.columns) - max_missing)
                    else:
                        # Entferne Zeilen mit irgendwelchen fehlenden Werten
                        df = df.dropna(subset=columns if columns else None)
                    applied_operations.append(f"dropMissingRows: {len(columns)} Spalten")
                
                elif op_type == 'fillmissing':
                    # Fülle fehlende Werte
                    method = op.get('method', 'mean')
                    value = op.get('value')
                    
                    for col in columns:
                        if col not in df.columns:
                            continue
                        
                        if method == 'mean' and df[col].dtype in [np.number]:
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif method == 'median' and df[col].dtype in [np.number]:
                            df[col].fillna(df[col].median(), inplace=True)
                        elif method == 'mode':
                            mode_value = df[col].mode()
                            if len(mode_value) > 0:
                                df[col].fillna(mode_value[0], inplace=True)
                        elif method == 'constant' and value is not None:
                            df[col].fillna(value, inplace=True)
                        elif method == 'drop':
                            df = df.dropna(subset=[col])
                    
                    applied_operations.append(f"fillMissing: {method} auf {len(columns)} Spalten")
                
                elif op_type == 'dropcolumn':
                    # Entferne Spalten
                    cols_to_drop = [col for col in columns if col in df.columns]
                    df = df.drop(columns=cols_to_drop)
                    applied_operations.append(f"dropColumn: {len(cols_to_drop)} Spalten entfernt")
                
                elif op_type == 'removeoutliers':
                    # Entferne Ausreißer
                    method = op.get('method', 'iqr')
                    
                    for col in columns:
                        if col not in df.columns or df[col].dtype not in [np.number]:
                            continue
                        
                        if method == 'iqr':
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                        elif method == 'zscore':
                            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                            df = df[z_scores < 3]
                    
                    applied_operations.append(f"removeOutliers: {method} auf {len(columns)} Spalten")
                
                elif op_type == 'encodecategorial' or op_type == 'encodecategorical':
                    # Kodiere kategorische Variablen
                    method = op.get('method', 'label')
                    
                    for col in columns:
                        if col not in df.columns:
                            continue
                        
                        if method == 'onehot':
                            # One-Hot Encoding
                            dummies = pd.get_dummies(df[col], prefix=col)
                            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                        elif method == 'label':
                            # Label Encoding
                            df[col] = pd.Categorical(df[col]).codes
                    
                    applied_operations.append(f"encodeCategorial: {method} auf {len(columns)} Spalten")
            
            # Speichere bereinigte Datei
            if output_path is None:
                output_path = file_path
            
            df.to_csv(output_path, index=False)
            
            # Erstelle Zusammenfassung
            rows_removed = original_shape[0] - df.shape[0]
            cols_removed = original_shape[1] - df.shape[1]
            summary = f"Datenbereinigung abgeschlossen: {len(applied_operations)} Operationen durchgeführt"
            if rows_removed > 0:
                summary += f", {rows_removed} Zeilen entfernt"
            if cols_removed > 0:
                summary += f", {cols_removed} Spalten entfernt"
            
            return {
                'success': True,
                'cleanedPath': output_path,
                'operations': applied_operations,
                'summary': summary,
                'originalShape': original_shape,
                'cleanedShape': df.shape
            }
            
        except Exception as error:
            return {
                'success': False,
                'error': str(error),
                'cleanedPath': file_path,
                'operations': [],
                'summary': f'Fehler bei Datenbereinigung: {error}'
            }

# Globale Instanz
python_client = PythonClient()

