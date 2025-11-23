"""
Beispiel-Verwendung des CSV-Manipulators
"""

from shared.utils.csv_manipulator import CSVManipulator, manipulate_csv
from infrastructure.clients.python_client import python_client

# Beispiel 1: Direkte Verwendung des CSVManipulators
def example_direct_usage():
    """Beispiel für direkte Verwendung"""
    # Erstelle Manipulator
    manipulator = CSVManipulator('data.csv')
    
    # Zeige Informationen
    print("Spalten:", manipulator.get_columns())
    print("Shape:", manipulator.get_shape())
    print("Info:", manipulator.get_info())
    
    # Lösche Spalten
    manipulator.drop_columns(['unwanted_col1', 'unwanted_col2'], inplace=True)
    
    # Führe Spalten zusammen
    manipulator.merge_columns(
        columns=['col1', 'col2'],
        new_column_name='combined',
        operation='add',
        inplace=True
    )
    
    # Füge neue Spalte hinzu
    manipulator.add_column(
        column_name='new_feature',
        values='col1 * 2 + col2',
        inplace=True
    )
    
    # Filtere Zeilen
    manipulator.filter_rows('age > 30', inplace=True)
    
    # Speichere
    manipulator.save(create_backup=True)


# Beispiel 2: Verwendung mit Operations-Liste
def example_operations_list():
    """Beispiel für Verwendung mit Operations-Liste"""
    operations = [
        {
            'type': 'drop_columns',
            'columns': ['id', 'unwanted_col']
        },
        {
            'type': 'merge_columns',
            'columns': ['income', 'savings'],
            'new_column_name': 'total_assets',
            'operation': 'add'
        },
        {
            'type': 'add_column',
            'column_name': 'income_per_age',
            'values': 'income / age'
        },
        {
            'type': 'filter_rows',
            'condition': 'age >= 18'
        },
        {
            'type': 'rename_columns',
            'rename_map': {'old_name': 'new_name'}
        }
    ]
    
    # Führe alle Operationen aus
    output_path = manipulate_csv(
        file_path='data.csv',
        operations=operations,
        output_path='data_processed.csv',
        create_backup=True
    )
    
    print(f"Verarbeitete Datei gespeichert: {output_path}")


# Beispiel 3: Verwendung über python_client
def example_via_client():
    """Beispiel für Verwendung über python_client"""
    operations = [
        {
            'type': 'drop_columns',
            'columns': ['temp_col']
        },
        {
            'type': 'merge_columns',
            'columns': ['feature1', 'feature2'],
            'new_column_name': 'combined_feature',
            'operation': 'multiply'
        }
    ]
    
    # Über python_client
    output_path = python_client.manipulate_csv(
        file_path='data.csv',
        operations=operations,
        create_backup=True
    )
    
    print(f"Verarbeitete Datei: {output_path}")


# Beispiel 4: In einem Agent verwenden
def example_in_agent():
    """Beispiel für Verwendung in einem Agent"""
    from core.agents.base_agent import BaseWorker
    
    class MyAgent(BaseWorker):
        async def execute(self, pipeline_state):
            project = pipeline_state.get('project', {})
            csv_path = project.get('csvFilePath', '')
            
            # Erstelle Manipulator
            manipulator = python_client.get_csv_manipulator(csv_path)
            
            # Führe Operationen aus
            manipulator.drop_columns(['unwanted_col'], inplace=True)
            manipulator.merge_columns(
                columns=['col1', 'col2'],
                new_column_name='combined',
                operation='add',
                inplace=True
            )
            
            # Speichere (mit Backup)
            new_path = manipulator.save(create_backup=True)
            
            # Aktualisiere Projekt-Pfad
            project['csvFilePath'] = new_path
            
            return {'success': True, 'new_path': new_path}

