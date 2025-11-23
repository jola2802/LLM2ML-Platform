"""
Datenanalyse-Worker-Agent
Analysiert Datasets mit LLM-basierter Analyse
"""

from typing import Dict, Any
from core.agents.base_agent import BaseWorker
from core.agents.prompts import (
    DATA_ANALYSIS_PROMPT,
    format_prompt
)
from infrastructure.clients.python_client import python_client
from shared.utils.data_processing import extract_and_validate_json

class DataAnalyzerWorker(BaseWorker):
    def __init__(self):
        super().__init__('DATA_ANALYZER')
    
    async def execute(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Führt Datenanalyse aus"""
        self.log('info', 'Starte Datenanalyse')
        
        project = pipeline_state.get('project', {})
        
        if not project or not project.get('csvFilePath'):
            raise ValueError('Kein Dataset-Pfad verfügbar für Datenanalyse')
        
        csv_file_path = project.get('csvFilePath')
        
        try:
            # Hole Basis-Datenexploration
            exploration_result = python_client.analyze_data(csv_file_path)
            
            if not exploration_result or not exploration_result.get('success'):
                self.log('warn', 'Keine Datenexploration verfügbar')
                raise ValueError('Datenexploration fehlgeschlagen')
            
            exploration = exploration_result.get('exploration', {})
            
            # Führe LLM-Analyse durch
            llm_analysis = await self._analyze_with_llm(
                file_path=csv_file_path,
                exploration=exploration
            )
            
            # Kombiniere Ergebnisse
            result = {
                'success': True,
                'exploration': exploration,
                'llmAnalysis': llm_analysis,
                'llmSummary': llm_analysis.get('summary', ''),
                'filePath': csv_file_path,
                'timestamp': __import__('datetime').datetime.now().isoformat()
            }
            
            self.log('success', 'Data analysis completed successfully')
            return result
            
        except Exception as error:
            self.log('error', f'Data analysis failed: {error}')
            raise
    
    async def _analyze_with_llm(self, file_path: str, exploration: Dict[str, Any]) -> Dict[str, Any]:
        """Führt LLM-Analyse durch"""
        
        # Formatiere Daten für Prompt
        columns = exploration.get('columns', [])
        row_count = exploration.get('rowCount', 0)
        data_types = exploration.get('dataTypes', {})
        numeric_stats = exploration.get('numericStats', {})
        categorical_stats = exploration.get('categoricalStats', {})
        sample_data = exploration.get('sampleData', [])
        
        # Formatiere Daten für Prompt
        data_types_str = self._format_data_types(data_types)
        numeric_stats_str = self._format_numeric_stats(numeric_stats)
        categorical_stats_str = self._format_categorical_stats(categorical_stats)
        sample_data_str = self._format_sample_data(sample_data, columns)
        
        # Erstelle Prompt
        prompt = format_prompt(DATA_ANALYSIS_PROMPT, {
            'filePath': file_path,
            'rowCount': row_count,
            'columnCount': len(columns),
            'columns': ', '.join(columns),
            'dataTypes': data_types_str,
            'numericStats': numeric_stats_str,
            'categoricalStats': categorical_stats_str,
            'sampleData': sample_data_str
        })
        
        try:
            # Rufe LLM auf
            self.log('info', 'Rufe LLM auf für Datenanalyse')
            response = await self.call_llm(prompt, None, self.config.get('maxTokens', 4096))
            
            # Extrahiere JSON aus Response
            result = extract_and_validate_json(response)
            
            return result
            
        except Exception as error:
            self.log('error', f'Fehler bei LLM-Analyse: {error}')
            
            # Fallback: Vereinfachte Analyse
            return {
                'summary': f'Dataset mit {row_count} Zeilen und {len(columns)} Spalten',
                'dataQuality': {
                    'missingValues': 'Nicht analysiert',
                    'dataCompleteness': 'Unbekannt',
                    'potentialIssues': []
                },
                'dataCharacteristics': {
                    'numericColumns': [col for col, dt in data_types.items() if dt == 'numeric'],
                    'categoricalColumns': [col for col, dt in data_types.items() if dt == 'categorical'],
                    'keyInsights': []
                },
                'recommendations': {
                    'preprocessing': [],
                    'featureEngineering': [],
                    'modelType': 'Unknown'
                },
                'targetVariableSuggestion': '',
                'reasoning': f'Fallback-Analyse: LLM-Analyse fehlgeschlagen ({error})'
            }
    
    def _format_data_types(self, data_types: Dict[str, str]) -> str:
        """Formatiert Datentypen für Prompt"""
        if not data_types:
            return "Keine Datentypen verfügbar"
        
        lines = []
        for col, dtype in data_types.items():
            lines.append(f"  - {col}: {dtype}")
        
        return "\n".join(lines) if lines else "Keine Datentypen verfügbar"
    
    def _format_numeric_stats(self, numeric_stats: Dict[str, Any]) -> str:
        """Formatiert numerische Statistiken für Prompt"""
        if not numeric_stats:
            return "Keine numerischen Statistiken verfügbar"
        
        lines = []
        for col, stats in numeric_stats.items():
            lines.append(f"\n{col}:")
            for key, value in stats.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines) if lines else "Keine numerischen Statistiken verfügbar"
    
    def _format_categorical_stats(self, categorical_stats: Dict[str, Any]) -> str:
        """Formatiert kategorische Statistiken für Prompt"""
        if not categorical_stats:
            return "Keine kategorischen Statistiken verfügbar"
        
        lines = []
        for col, stats in categorical_stats.items():
            lines.append(f"\n{col}:")
            lines.append(f"  Unique values: {stats.get('unique_count', 0)}")
            top_values = stats.get('top_values', {})
            if top_values:
                lines.append(f"  Top values: {', '.join(list(top_values.keys())[:5])}")
        
        return "\n".join(lines) if lines else "Keine kategorischen Statistiken verfügbar"
    
    def _format_sample_data(self, sample_data: list, columns: list) -> str:
        """Formatiert Sample-Daten für Prompt"""
        if not sample_data or not columns:
            return "Keine Sample-Daten verfügbar"
        
        lines = []
        lines.append("Columns: " + ", ".join(columns))
        lines.append("\nSample rows:")
        for i, row in enumerate(sample_data[:5], 1):
            lines.append(f"  Row {i}: {row}")
        
        return "\n".join(lines)
