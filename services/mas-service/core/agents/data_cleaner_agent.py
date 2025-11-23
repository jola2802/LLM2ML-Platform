"""
Data Cleaner Worker-Agent
Bereinigt Datasets basierend auf Datenanalyse-Ergebnissen
"""

from typing import Dict, Any
from core.agents.base_agent import BaseWorker
from core.agents.prompts import (
    DATA_CLEANING_PROMPT,
    format_prompt
)
from shared.utils.data_processing import extract_and_validate_json
from infrastructure.clients.python_client import python_client

class DataCleanerWorker(BaseWorker):
    def __init__(self):
        super().__init__('DATA_CLEANER')
    
    async def execute(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Führt Datenbereinigung aus"""
        self.log('info', 'Starte Datenbereinigung')
        
        project = pipeline_state.get('project', {})
        results = pipeline_state.get('results', {})
        
        # Hole Datenanalyse-Ergebnisse
        data_analysis = results.get('DATA_ANALYZER', {})
        if not data_analysis or not data_analysis.get('success'):
            self.log('warning', 'Keine Datenanalyse verfügbar - überspringe Bereinigung')
            return {
                'success': True,
                'cleaningPerformed': False,
                'reason': 'Keine Datenanalyse verfügbar'
            }
        
        csv_file_path = project.get('csvFilePath')
        if not csv_file_path:
            raise ValueError('Kein CSV-Dateipfad verfügbar')
        
        # Analysiere Datenqualität
        exploration = data_analysis.get('exploration', {})
        llm_analysis = data_analysis.get('llmAnalysis', {})
        data_quality = llm_analysis.get('dataQuality', {})
        
        # Bestimme ob Bereinigung notwendig ist
        needs_cleaning = self._assess_cleaning_needs(data_quality, exploration)
        
        if not needs_cleaning:
            self.log('info', 'Daten sind sauber - keine Bereinigung notwendig')
            return {
                'success': True,
                'cleaningPerformed': False,
                'reason': 'Daten bereits sauber',
                'originalPath': csv_file_path
            }
        
        # Generiere Bereinigungsplan mit LLM
        cleaning_plan = await self._create_cleaning_plan(
            data_quality=data_quality,
            exploration=exploration,
            file_path=csv_file_path
        )
        
        # Führe Bereinigung aus (via Python Client)
        cleaning_result = self._apply_cleaning_plan(
            cleaning_plan=cleaning_plan,
            file_path=csv_file_path
        )
        
        result = {
            'success': cleaning_result.get('success', True),
            'cleaningPerformed': True,
            'cleaningPlan': cleaning_plan,
            'originalPath': csv_file_path,
            'cleanedPath': cleaning_result.get('cleanedPath', csv_file_path),
            'operationsApplied': cleaning_result.get('operations', []),
            'summary': cleaning_result.get('summary', 'Datenbereinigung abgeschlossen')
        }
        
        self.log('success', f'Datenbereinigung erfolgreich - {len(result["operationsApplied"])} Operationen durchgeführt')
        return result
    
    def _assess_cleaning_needs(self, data_quality: Dict[str, Any], exploration: Dict[str, Any]) -> bool:
        """Prüft ob Datenbereinigung notwendig ist"""
        
        # Prüfe auf Missing Values
        potential_issues = data_quality.get('potentialIssues', [])
        if potential_issues and len(potential_issues) > 0:
            return True
        
        # Prüfe Missing Values in Exploration
        missing_values = exploration.get('missingValues', {})
        if missing_values and any(count > 0 for count in missing_values.values()):
            return True
        
        return False
    
    async def _create_cleaning_plan(
        self,
        data_quality: Dict[str, Any],
        exploration: Dict[str, Any],
        file_path: str
    ) -> Dict[str, Any]:
        """Erstellt Bereinigungsplan mit LLM"""
        
        # Formatiere Datenqualitäts-Informationen
        quality_summary = self._format_quality_issues(data_quality, exploration)
        
        # Erstelle Prompt
        prompt = format_prompt(DATA_CLEANING_PROMPT, {
            'filePath': file_path,
            'qualitySummary': quality_summary,
            'missingValues': str(exploration.get('missingValues', {})),
            'rowCount': exploration.get('rowCount', 0),
            'columnCount': len(exploration.get('columns', []))
        })
        
        try:
            # Rufe LLM auf
            self.log('info', 'Rufe LLM auf für Bereinigungsplan')
            response = await self.call_llm(prompt, None, self.config.get('maxTokens', 2048))
            
            # Extrahiere JSON aus Response
            result = extract_and_validate_json(response)
            
            return {
                'operations': result.get('operations', []),
                'reasoning': result.get('reasoning', ''),
                'priority': result.get('priority', 'medium')
            }
            
        except Exception as error:
            self.log('error', f'Fehler bei Bereinigungsplan-Erstellung: {error}')
            
            # Fallback: Basis-Bereinigungsplan
            return {
                'operations': [
                    {
                        'type': 'dropMissingRows',
                        'columns': list(exploration.get('missingValues', {}).keys()),
                        'threshold': 0.5
                    }
                ],
                'reasoning': 'Fallback: Entferne Zeilen mit vielen fehlenden Werten',
                'priority': 'low'
            }
    
    def _apply_cleaning_plan(
        self,
        cleaning_plan: Dict[str, Any],
        file_path: str
    ) -> Dict[str, Any]:
        """Führt Bereinigungsplan aus"""
        
        operations = cleaning_plan.get('operations', [])
        
        if not operations:
            return {
                'success': True,
                'cleanedPath': file_path,
                'operations': [],
                'summary': 'Keine Bereinigungsoperationen notwendig'
            }
        
        try:
            # Rufe Python Client für Datenbereinigung auf
            result = python_client.clean_data(file_path, operations)
            
            if result and result.get('success'):
                return {
                    'success': True,
                    'cleanedPath': result.get('cleanedPath', file_path),
                    'operations': operations,
                    'summary': result.get('summary', 'Datenbereinigung erfolgreich')
                }
            else:
                # Bereinigung fehlgeschlagen, verwende Original
                self.log('warning', 'Bereinigung fehlgeschlagen, verwende Original-Daten')
                return {
                    'success': True,
                    'cleanedPath': file_path,
                    'operations': [],
                    'summary': 'Bereinigung fehlgeschlagen, Original-Daten verwendet'
                }
                
        except Exception as error:
            self.log('error', f'Fehler bei Bereinigung: {error}')
            return {
                'success': True,
                'cleanedPath': file_path,
                'operations': [],
                'summary': f'Bereinigung fehlgeschlagen: {error}'
            }
    
    def _format_quality_issues(self, data_quality: Dict[str, Any], exploration: Dict[str, Any]) -> str:
        """Formatiert Qualitätsprobleme für Prompt"""
        lines = []
        
        # Missing Values
        missing_values = exploration.get('missingValues', {})
        if missing_values:
            lines.append("Missing Values:")
            for col, count in missing_values.items():
                if count > 0:
                    lines.append(f"  - {col}: {count} missing")
        
        # Potential Issues
        potential_issues = data_quality.get('potentialIssues', [])
        if potential_issues:
            lines.append("\nPotential Issues:")
            for issue in potential_issues:
                lines.append(f"  - {issue}")
        
        # Data Completeness
        completeness = data_quality.get('dataCompleteness', '')
        if completeness:
            lines.append(f"\nData Completeness: {completeness}")
        
        return "\n".join(lines) if lines else "Keine Qualitätsprobleme identifiziert"
