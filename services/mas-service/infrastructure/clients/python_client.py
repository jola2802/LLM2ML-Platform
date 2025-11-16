"""
Python-Client für direkte Funktionsaufrufe
"""

from typing import Optional, Dict, Any
from core.data.data_exploration import get_cached_data_analysis

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

# Globale Instanz
python_client = PythonClient()

