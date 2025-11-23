"""
CSV-Manipulationstool für Agents
Ermöglicht sichere und effiziente CSV-Operationen
"""

import pandas as pd
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


class CSVManipulator:
    """
    Tool für CSV-Manipulationen
    Bietet sichere Operationen zum Bearbeiten von CSV-Dateien
    """
    
    def __init__(self, file_path: str):
        """
        Initialisiert den CSV-Manipulator mit einer CSV-Datei
        
        Args:
            file_path: Pfad zur CSV-Datei
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV-Datei nicht gefunden: {file_path}")
        
        self.file_path = file_path
        self.backup_path = None
        self._df = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Lädt die CSV-Datei in einen DataFrame"""
        try:
            self._df = pd.read_csv(self.file_path)
            self._original_df = self._df.copy()
        except Exception as e:
            raise ValueError(f"Fehler beim Laden der CSV-Datei: {e}")
    
    def _create_backup(self) -> None:
        """Erstellt ein Backup der Originaldatei"""
        if self.backup_path is None:
            backup_dir = os.path.join(os.path.dirname(self.file_path), '.backups')
            os.makedirs(backup_dir, exist_ok=True)
            
            file_name = Path(self.file_path).stem
            file_ext = Path(self.file_path).suffix
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            
            self.backup_path = os.path.join(
                backup_dir, 
                f"{file_name}_backup_{timestamp}{file_ext}"
            )
            
            self._original_df.to_csv(self.backup_path, index=False)
            print(f"📦 Backup erstellt: {self.backup_path}")
    
    def get_columns(self) -> List[str]:
        """Gibt eine Liste aller Spaltennamen zurück"""
        return list(self._df.columns)
    
    def get_shape(self) -> tuple:
        """Gibt die Dimensionen des DataFrames zurück (rows, columns)"""
        return self._df.shape
    
    def get_info(self) -> Dict[str, Any]:
        """Gibt Informationen über den DataFrame zurück"""
        return {
            'columns': list(self._df.columns),
            'shape': self._df.shape,
            'dtypes': self._df.dtypes.to_dict(),
            'memory_usage': self._df.memory_usage(deep=True).sum(),
            'null_counts': self._df.isnull().sum().to_dict()
        }
    
    def drop_columns(self, columns: Union[str, List[str]], inplace: bool = False) -> Optional[pd.DataFrame]:
        """
        Löscht eine oder mehrere Spalten
        
        Args:
            columns: Spaltenname oder Liste von Spaltennamen
            inplace: Wenn True, wird der DataFrame direkt modifiziert
        
        Returns:
            Modifizierter DataFrame (wenn inplace=False)
        """
        if isinstance(columns, str):
            columns = [columns]
        
        # Prüfe ob alle Spalten existieren
        missing_cols = [col for col in columns if col not in self._df.columns]
        if missing_cols:
            raise ValueError(f"Spalten nicht gefunden: {missing_cols}")
        
        if inplace:
            self._df.drop(columns=columns, inplace=True)
            return None
        else:
            return self._df.drop(columns=columns)
    
    def merge_columns(
        self, 
        columns: List[str], 
        new_column_name: str,
        operation: str = 'add',
        inplace: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Führt mehrere Spalten zusammen
        
        Args:
            columns: Liste der zu kombinierenden Spalten
            new_column_name: Name der neuen Spalte
            operation: Operation ('add', 'subtract', 'multiply', 'divide', 'concat', 'mean')
            inplace: Wenn True, wird der DataFrame direkt modifiziert
        
        Returns:
            Modifizierter DataFrame (wenn inplace=False)
        """
        # Prüfe ob alle Spalten existieren
        missing_cols = [col for col in columns if col not in self._df.columns]
        if missing_cols:
            raise ValueError(f"Spalten nicht gefunden: {missing_cols}")
        
        # Prüfe ob neue Spalte bereits existiert
        if new_column_name in self._df.columns:
            raise ValueError(f"Spalte '{new_column_name}' existiert bereits")
        
        # Führe Operation aus
        if operation == 'add':
            result = self._df[columns].sum(axis=1)
        elif operation == 'subtract':
            if len(columns) != 2:
                raise ValueError("Subtract benötigt genau 2 Spalten")
            result = self._df[columns[0]] - self._df[columns[1]]
        elif operation == 'multiply':
            result = self._df[columns].prod(axis=1)
        elif operation == 'divide':
            if len(columns) != 2:
                raise ValueError("Divide benötigt genau 2 Spalten")
            result = self._df[columns[0]] / self._df[columns[1]].replace(0, pd.NA)
        elif operation == 'concat':
            # String-Konkatenation
            result = self._df[columns].astype(str).agg('_'.join, axis=1)
        elif operation == 'mean':
            result = self._df[columns].mean(axis=1)
        else:
            raise ValueError(f"Unbekannte Operation: {operation}")
        
        if inplace:
            self._df[new_column_name] = result
            return None
        else:
            new_df = self._df.copy()
            new_df[new_column_name] = result
            return new_df
    
    def add_column(
        self,
        column_name: str,
        values: Union[List, pd.Series, str],
        inplace: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fügt eine neue Spalte hinzu
        
        Args:
            column_name: Name der neuen Spalte
            values: Werte (Liste, Series oder Formel-String wie 'col1 + col2')
            inplace: Wenn True, wird der DataFrame direkt modifiziert
        
        Returns:
            Modifizierter DataFrame (wenn inplace=False)
        """
        if column_name in self._df.columns:
            raise ValueError(f"Spalte '{column_name}' existiert bereits")
        
        if isinstance(values, str):
            # Formel-basierte Spalte (z.B. 'col1 + col2', 'col1 * 2')
            try:
                # Sicherheitsprüfung: Nur Spaltennamen und mathematische Operationen erlauben
                allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-*/()[]., ')
                if not all(c in allowed_chars for c in values):
                    raise ValueError("Formel enthält ungültige Zeichen")
                
                # Erstelle neue Spalte durch Auswertung der Formel
                result = self._df.eval(values)
            except Exception as e:
                raise ValueError(f"Fehler beim Auswerten der Formel '{values}': {e}")
        elif isinstance(values, (list, pd.Series)):
            if len(values) != len(self._df):
                raise ValueError(f"Anzahl der Werte ({len(values)}) stimmt nicht mit DataFrame-Länge ({len(self._df)}) überein")
            result = pd.Series(values) if isinstance(values, list) else values
        else:
            raise TypeError(f"Unbekannter Typ für values: {type(values)}")
        
        if inplace:
            self._df[column_name] = result
            return None
        else:
            new_df = self._df.copy()
            new_df[column_name] = result
            return new_df
    
    def filter_rows(
        self,
        condition: str,
        inplace: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Filtert Zeilen basierend auf einer Bedingung
        
        Args:
            condition: Bedingung als String (z.B. 'age > 30', 'income < 50000')
            inplace: Wenn True, wird der DataFrame direkt modifiziert
        
        Returns:
            Gefilterter DataFrame (wenn inplace=False)
        """
        try:
            mask = self._df.eval(condition)
            if inplace:
                self._df = self._df[mask]
                return None
            else:
                return self._df[mask]
        except Exception as e:
            raise ValueError(f"Fehler beim Filtern mit Bedingung '{condition}': {e}")
    
    def rename_columns(
        self,
        rename_map: Dict[str, str],
        inplace: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Benennt Spalten um
        
        Args:
            rename_map: Dictionary mit {alter_name: neuer_name}
            inplace: Wenn True, wird der DataFrame direkt modifiziert
        
        Returns:
            DataFrame mit umbenannten Spalten (wenn inplace=False)
        """
        missing_cols = [col for col in rename_map.keys() if col not in self._df.columns]
        if missing_cols:
            raise ValueError(f"Spalten nicht gefunden: {missing_cols}")
        
        if inplace:
            self._df.rename(columns=rename_map, inplace=True)
            return None
        else:
            return self._df.rename(columns=rename_map)
    
    def save(self, output_path: Optional[str] = None, create_backup: bool = True) -> str:
        """
        Speichert den modifizierten DataFrame
        
        Args:
            output_path: Pfad zum Speichern (wenn None, überschreibt Original)
            create_backup: Erstellt Backup der Originaldatei
        
        Returns:
            Pfad zur gespeicherten Datei
        """
        if create_backup and self.backup_path is None:
            self._create_backup()
        
        save_path = output_path or self.file_path
        
        try:
            self._df.to_csv(save_path, index=False)
            print(f"✅ CSV gespeichert: {save_path}")
            return save_path
        except Exception as e:
            raise IOError(f"Fehler beim Speichern der CSV-Datei: {e}")
    
    def reset(self) -> None:
        """Setzt den DataFrame auf den Originalzustand zurück"""
        self._df = self._original_df.copy()
        print("🔄 DataFrame auf Originalzustand zurückgesetzt")
    
    def preview(self, n_rows: int = 5) -> pd.DataFrame:
        """Gibt eine Vorschau des DataFrames zurück"""
        return self._df.head(n_rows)


# Convenience-Funktionen für einfache Verwendung
def manipulate_csv(
    file_path: str,
    operations: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    create_backup: bool = True
) -> str:
    """
    Führt mehrere CSV-Operationen in einer Funktion aus
    
    Args:
        file_path: Pfad zur CSV-Datei
        operations: Liste von Operationen, z.B.:
            [
                {'type': 'drop_columns', 'columns': ['col1', 'col2']},
                {'type': 'merge_columns', 'columns': ['col3', 'col4'], 
                 'new_column_name': 'combined', 'operation': 'add'},
                {'type': 'add_column', 'column_name': 'new_col', 
                 'values': 'col3 * 2'},
                {'type': 'filter_rows', 'condition': 'col3 > 100'},
                {'type': 'rename_columns', 'rename_map': {'old': 'new'}}
            ]
        output_path: Pfad zum Speichern (wenn None, überschreibt Original)
        create_backup: Erstellt Backup der Originaldatei
    
    Returns:
        Pfad zur gespeicherten Datei
    """
    manipulator = CSVManipulator(file_path)
    
    for op in operations:
        op_type = op.get('type')
        inplace = op.get('inplace', True)
        
        if op_type == 'drop_columns':
            manipulator.drop_columns(
                columns=op['columns'],
                inplace=inplace
            )
        elif op_type == 'merge_columns':
            manipulator.merge_columns(
                columns=op['columns'],
                new_column_name=op['new_column_name'],
                operation=op.get('operation', 'add'),
                inplace=inplace
            )
        elif op_type == 'add_column':
            manipulator.add_column(
                column_name=op['column_name'],
                values=op['values'],
                inplace=inplace
            )
        elif op_type == 'filter_rows':
            manipulator.filter_rows(
                condition=op['condition'],
                inplace=inplace
            )
        elif op_type == 'rename_columns':
            manipulator.rename_columns(
                rename_map=op['rename_map'],
                inplace=inplace
            )
        else:
            raise ValueError(f"Unbekannte Operation: {op_type}")
    
    return manipulator.save(output_path, create_backup)

