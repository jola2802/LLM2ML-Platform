"""
CSV-Manipulationstools als MCP Tools
"""

from typing import Dict, Any, List, Optional
from shared.utils.csv_manipulator import CSVManipulator, manipulate_csv
import json


def csv_drop_columns(file_path: str, columns: List[str], output_path: Optional[str] = None, create_backup: bool = True) -> Dict[str, Any]:
    """
    Löscht Spalten aus einer CSV-Datei
    
    Args:
        file_path: Pfad zur CSV-Datei
        columns: Liste der zu löschenden Spaltennamen
        output_path: Optional: Pfad zum Speichern (wenn None, überschreibt Original)
        create_backup: Erstellt Backup der Originaldatei
    
    Returns:
        Ergebnis-Dictionary mit Status und Informationen
    """
    try:
        manipulator = CSVManipulator(file_path)
        manipulator.drop_columns(columns, inplace=True)
        saved_path = manipulator.save(output_path, create_backup)
        
        return {
            'success': True,
            'message': f'Spalten {columns} erfolgreich gelöscht',
            'output_path': saved_path,
            'remaining_columns': manipulator.get_columns(),
            'shape': manipulator.get_shape()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f'Fehler beim Löschen der Spalten: {e}'
        }


def csv_merge_columns(
    file_path: str,
    columns: List[str],
    new_column_name: str,
    operation: str = 'add',
    output_path: Optional[str] = None,
    create_backup: bool = True
) -> Dict[str, Any]:
    """
    Führt mehrere Spalten zusammen
    
    Args:
        file_path: Pfad zur CSV-Datei
        columns: Liste der zu kombinierenden Spalten
        new_column_name: Name der neuen Spalte
        operation: Operation ('add', 'subtract', 'multiply', 'divide', 'concat', 'mean')
        output_path: Optional: Pfad zum Speichern
        create_backup: Erstellt Backup
    
    Returns:
        Ergebnis-Dictionary
    """
    try:
        manipulator = CSVManipulator(file_path)
        manipulator.merge_columns(columns, new_column_name, operation, inplace=True)
        saved_path = manipulator.save(output_path, create_backup)
        
        return {
            'success': True,
            'message': f'Spalten {columns} erfolgreich zu "{new_column_name}" zusammengeführt (Operation: {operation})',
            'output_path': saved_path,
            'new_column': new_column_name,
            'all_columns': manipulator.get_columns(),
            'shape': manipulator.get_shape()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f'Fehler beim Zusammenführen der Spalten: {e}'
        }


def csv_add_column(
    file_path: str,
    column_name: str,
    values: str,
    output_path: Optional[str] = None,
    create_backup: bool = True
) -> Dict[str, Any]:
    """
    Fügt eine neue Spalte hinzu
    
    Args:
        file_path: Pfad zur CSV-Datei
        column_name: Name der neuen Spalte
        values: Werte als Formel-String (z.B. 'col1 * 2 + col2') oder JSON-Array
        output_path: Optional: Pfad zum Speichern
        create_backup: Erstellt Backup
    
    Returns:
        Ergebnis-Dictionary
    """
    try:
        manipulator = CSVManipulator(file_path)
        
        # Versuche values als JSON-Array zu parsen, sonst als Formel-String
        try:
            parsed_values = json.loads(values)
            if isinstance(parsed_values, list):
                values = parsed_values
        except (json.JSONDecodeError, TypeError):
            # Behalte als String (Formel)
            pass
        
        manipulator.add_column(column_name, values, inplace=True)
        saved_path = manipulator.save(output_path, create_backup)
        
        return {
            'success': True,
            'message': f'Spalte "{column_name}" erfolgreich hinzugefügt',
            'output_path': saved_path,
            'new_column': column_name,
            'all_columns': manipulator.get_columns(),
            'shape': manipulator.get_shape()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f'Fehler beim Hinzufügen der Spalte: {e}'
        }


def csv_filter_rows(
    file_path: str,
    condition: str,
    output_path: Optional[str] = None,
    create_backup: bool = True
) -> Dict[str, Any]:
    """
    Filtert Zeilen basierend auf einer Bedingung
    
    Args:
        file_path: Pfad zur CSV-Datei
        condition: Bedingung als String (z.B. 'age > 30', 'income < 50000')
        output_path: Optional: Pfad zum Speichern
        create_backup: Erstellt Backup
    
    Returns:
        Ergebnis-Dictionary
    """
    try:
        manipulator = CSVManipulator(file_path)
        original_shape = manipulator.get_shape()
        manipulator.filter_rows(condition, inplace=True)
        new_shape = manipulator.get_shape()
        saved_path = manipulator.save(output_path, create_backup)
        
        rows_filtered = original_shape[0] - new_shape[0]
        
        return {
            'success': True,
            'message': f'{rows_filtered} Zeilen gefiltert (Bedingung: {condition})',
            'output_path': saved_path,
            'original_shape': original_shape,
            'new_shape': new_shape,
            'rows_filtered': rows_filtered
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f'Fehler beim Filtern: {e}'
        }


def csv_rename_columns(
    file_path: str,
    rename_map: Dict[str, str],
    output_path: Optional[str] = None,
    create_backup: bool = True
) -> Dict[str, Any]:
    """
    Benennt Spalten um
    
    Args:
        file_path: Pfad zur CSV-Datei
        rename_map: Dictionary mit {alter_name: neuer_name}
        output_path: Optional: Pfad zum Speichern
        create_backup: Erstellt Backup
    
    Returns:
        Ergebnis-Dictionary
    """
    try:
        manipulator = CSVManipulator(file_path)
        manipulator.rename_columns(rename_map, inplace=True)
        saved_path = manipulator.save(output_path, create_backup)
        
        return {
            'success': True,
            'message': f'Spalten erfolgreich umbenannt: {rename_map}',
            'output_path': saved_path,
            'renamed_columns': rename_map,
            'all_columns': manipulator.get_columns()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f'Fehler beim Umbenennen: {e}'
        }


def csv_get_info(file_path: str) -> Dict[str, Any]:
    """
    Gibt Informationen über eine CSV-Datei zurück
    
    Args:
        file_path: Pfad zur CSV-Datei
    
    Returns:
        Informationen über die CSV-Datei
    """
    try:
        manipulator = CSVManipulator(file_path)
        info = manipulator.get_info()
        
        return {
            'success': True,
            'file_path': file_path,
            'columns': info['columns'],
            'shape': info['shape'],
            'dtypes': {str(k): str(v) for k, v in info['dtypes'].items()},
            'null_counts': info['null_counts'],
            'memory_usage_bytes': info['memory_usage']
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f'Fehler beim Abrufen der Informationen: {e}'
        }


def csv_batch_operations(
    file_path: str,
    operations: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    create_backup: bool = True
) -> Dict[str, Any]:
    """
    Führt mehrere CSV-Operationen in einem Batch aus
    
    Args:
        file_path: Pfad zur CSV-Datei
        operations: Liste von Operationen (siehe csv_manipulator.manipulate_csv)
        output_path: Optional: Pfad zum Speichern
        create_backup: Erstellt Backup
    
    Returns:
        Ergebnis-Dictionary
    """
    try:
        saved_path = manipulate_csv(file_path, operations, output_path, create_backup)
        
        return {
            'success': True,
            'message': f'{len(operations)} Operationen erfolgreich ausgeführt',
            'output_path': saved_path,
            'operations_count': len(operations)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f'Fehler bei Batch-Operationen: {e}'
        }


# Tool-Definitionen für MCP
CSV_TOOLS = [
    {
        'name': 'csv_drop_columns',
        'description': 'Löscht eine oder mehrere Spalten aus einer CSV-Datei',
        'parameters': {
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'Pfad zur CSV-Datei'
                },
                'columns': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'Liste der zu löschenden Spaltennamen'
                },
                'output_path': {
                    'type': 'string',
                    'description': 'Optional: Pfad zum Speichern (wenn nicht angegeben, überschreibt Original)'
                },
                'create_backup': {
                    'type': 'boolean',
                    'description': 'Erstellt Backup der Originaldatei (Standard: true)',
                    'default': True
                }
            },
            'required': ['file_path', 'columns']
        },
        'implementation': csv_drop_columns
    },
    {
        'name': 'csv_merge_columns',
        'description': 'Führt mehrere Spalten zu einer neuen Spalte zusammen (add, subtract, multiply, divide, concat, mean)',
        'parameters': {
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'Pfad zur CSV-Datei'
                },
                'columns': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'Liste der zu kombinierenden Spaltennamen'
                },
                'new_column_name': {
                    'type': 'string',
                    'description': 'Name der neuen zusammengeführten Spalte'
                },
                'operation': {
                    'type': 'string',
                    'enum': ['add', 'subtract', 'multiply', 'divide', 'concat', 'mean'],
                    'description': 'Operation zum Zusammenführen (Standard: add)',
                    'default': 'add'
                },
                'output_path': {
                    'type': 'string',
                    'description': 'Optional: Pfad zum Speichern'
                },
                'create_backup': {
                    'type': 'boolean',
                    'description': 'Erstellt Backup (Standard: true)',
                    'default': True
                }
            },
            'required': ['file_path', 'columns', 'new_column_name']
        },
        'implementation': csv_merge_columns
    },
    {
        'name': 'csv_add_column',
        'description': 'Fügt eine neue Spalte hinzu (mit Formel oder Werten)',
        'parameters': {
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'Pfad zur CSV-Datei'
                },
                'column_name': {
                    'type': 'string',
                    'description': 'Name der neuen Spalte'
                },
                'values': {
                    'type': 'string',
                    'description': 'Werte als Formel-String (z.B. "col1 * 2 + col2") oder JSON-Array'
                },
                'output_path': {
                    'type': 'string',
                    'description': 'Optional: Pfad zum Speichern'
                },
                'create_backup': {
                    'type': 'boolean',
                    'description': 'Erstellt Backup (Standard: true)',
                    'default': True
                }
            },
            'required': ['file_path', 'column_name', 'values']
        },
        'implementation': csv_add_column
    },
    {
        'name': 'csv_filter_rows',
        'description': 'Filtert Zeilen basierend auf einer Bedingung (z.B. "age > 30")',
        'parameters': {
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'Pfad zur CSV-Datei'
                },
                'condition': {
                    'type': 'string',
                    'description': 'Bedingung als String (z.B. "age > 30", "income < 50000")'
                },
                'output_path': {
                    'type': 'string',
                    'description': 'Optional: Pfad zum Speichern'
                },
                'create_backup': {
                    'type': 'boolean',
                    'description': 'Erstellt Backup (Standard: true)',
                    'default': True
                }
            },
            'required': ['file_path', 'condition']
        },
        'implementation': csv_filter_rows
    },
    {
        'name': 'csv_rename_columns',
        'description': 'Benennt Spalten um',
        'parameters': {
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'Pfad zur CSV-Datei'
                },
                'rename_map': {
                    'type': 'object',
                    'description': 'Dictionary mit {alter_name: neuer_name}',
                    'additionalProperties': {'type': 'string'}
                },
                'output_path': {
                    'type': 'string',
                    'description': 'Optional: Pfad zum Speichern'
                },
                'create_backup': {
                    'type': 'boolean',
                    'description': 'Erstellt Backup (Standard: true)',
                    'default': True
                }
            },
            'required': ['file_path', 'rename_map']
        },
        'implementation': csv_rename_columns
    },
    {
        'name': 'csv_get_info',
        'description': 'Gibt Informationen über eine CSV-Datei zurück (Spalten, Shape, Datentypen, etc.)',
        'parameters': {
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'Pfad zur CSV-Datei'
                }
            },
            'required': ['file_path']
        },
        'implementation': csv_get_info
    },
    {
        'name': 'csv_batch_operations',
        'description': 'Führt mehrere CSV-Operationen in einem Batch aus',
        'parameters': {
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'Pfad zur CSV-Datei'
                },
                'operations': {
                    'type': 'array',
                    'description': 'Liste von Operationen (drop_columns, merge_columns, add_column, filter_rows, rename_columns)',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'type': {
                                'type': 'string',
                                'enum': ['drop_columns', 'merge_columns', 'add_column', 'filter_rows', 'rename_columns']
                            },
                            'columns': {'type': 'array', 'items': {'type': 'string'}},
                            'new_column_name': {'type': 'string'},
                            'operation': {'type': 'string'},
                            'column_name': {'type': 'string'},
                            'values': {'type': 'string'},
                            'condition': {'type': 'string'},
                            'rename_map': {'type': 'object'}
                        }
                    }
                },
                'output_path': {
                    'type': 'string',
                    'description': 'Optional: Pfad zum Speichern'
                },
                'create_backup': {
                    'type': 'boolean',
                    'description': 'Erstellt Backup (Standard: true)',
                    'default': True
                }
            },
            'required': ['file_path', 'operations']
        },
        'implementation': csv_batch_operations
    }
]

