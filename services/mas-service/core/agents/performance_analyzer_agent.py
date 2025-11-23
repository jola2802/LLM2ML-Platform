"""
Performance-Analyzer-Worker-Agent
Analysiert Modell-Performance basierend auf Code-Ausführung
"""

import re
from typing import Dict, Any
from core.agents.base_agent import BaseWorker
from core.agents.prompts import (
    PERFORMANCE_EVALUATION_PROMPT,
    format_prompt
)
from core.execution.metrics_extractor import extract_metrics_from_output
from shared.utils.data_processing import extract_and_validate_json

class PerformanceAnalyzerWorker(BaseWorker):
    def __init__(self):
        super().__init__('PERFORMANCE_ANALYZER')
    
    async def execute(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analysiert Modell-Performance"""
        self.log('info', 'Starte Performance-Analyse')
        
        project = pipeline_state.get('project', {})
        results = pipeline_state.get('results', {})
        
        # Hole Code-Ausführungs-Ergebnisse
        code_execution = results.get('CODE_EXECUTOR', {})
        if not code_execution:
            raise ValueError('Keine Code-Ausführungs-Ergebnisse verfügbar')
        
        stdout = code_execution.get('stdout', '')
        stderr = code_execution.get('stderr', '')
        
        # Prüfe ob Ausführung erfolgreich war
        if not code_execution.get('success', False):
            self.log('error', 'Code-Ausführung fehlgeschlagen')
            return {
                'overallScore': 0.0,
                'performanceGrade': 'Poor',
                'summary': 'Code-Ausführung fehlgeschlagen',
                'detailedAnalysis': {
                    'strengths': [],
                    'weaknesses': ['Code-Ausführung fehlgeschlagen'],
                    'keyFindings': ['Fehler bei Code-Ausführung']
                },
                'improvementSuggestions': [],
                'metrics': {}
            }
        
        # Extrahiere Metriken aus Output
        llm_recommendations = project.get('llmRecommendations', {})
        model_type = llm_recommendations.get('modelType', 'Classification')
        algorithm = llm_recommendations.get('algorithm', '')
        target_variable = llm_recommendations.get('targetVariable', '')
        features = llm_recommendations.get('features', [])
        
        # Extrahiere Metriken
        metrics = extract_metrics_from_output(stdout + stderr, model_type)
        
        # Analysiere Performance mit LLM
        performance_analysis = await self._analyze_performance(
            project_name=project.get('name', 'Unknown'),
            algorithm=algorithm,
            model_type=model_type,
            target_variable=target_variable,
            features=features,
            performance_metrics=metrics,
            output=stdout
        )
        
        result = {
            'overallScore': performance_analysis.get('overallScore', 0.0),
            'performanceGrade': performance_analysis.get('performanceGrade', 'Poor'),
            'summary': performance_analysis.get('summary', ''),
            'detailedAnalysis': performance_analysis.get('detailedAnalysis', {}),
            'improvementSuggestions': performance_analysis.get('improvementSuggestions', []),
            'metrics': metrics
        }
        
        self.log('success', f'Performance-Analyse erfolgreich - Score: {result["overallScore"]:.2f}')
        return result
    
    async def _analyze_performance(
        self,
        project_name: str,
        algorithm: str,
        model_type: str,
        target_variable: str,
        features: list,
        performance_metrics: Dict[str, Any],
        output: str
    ) -> Dict[str, Any]:
        """Analysiert Performance mit LLM"""
        
        # Formatiere Metriken für Prompt
        metrics_str = self._format_metrics(performance_metrics, model_type)
        
        # Erstelle Prompt
        prompt = format_prompt(PERFORMANCE_EVALUATION_PROMPT, {
            'projectName': project_name,
            'algorithm': algorithm,
            'modelType': model_type,
            'targetVariable': target_variable,
            'features': ', '.join(features) if features else 'N/A',
            'performanceMetrics': metrics_str
        })
        
        try:
            # Rufe LLM auf
            self.log('info', 'Rufe LLM auf für Performance-Analyse')
            response = await self.call_llm(prompt, None, self.config.get('maxTokens', 2048))
            
            # Extrahiere JSON aus Response
            result = extract_and_validate_json(response)
            
            # Validiere und bereinige Ergebnis
            validated_result = self._validate_performance_analysis(result, performance_metrics)
            
            return validated_result
            
        except Exception as error:
            self.log('error', f'Fehler bei Performance-Analyse: {error}')
            
            # Fallback: Berechne Score basierend auf Metriken
            fallback_score = self._calculate_fallback_score(performance_metrics, model_type)
            
            return {
                'overallScore': fallback_score,
                'performanceGrade': self._score_to_grade(fallback_score),
                'summary': f'Fallback-Analyse: Score {fallback_score:.2f}',
                'detailedAnalysis': {
                    'strengths': [],
                    'weaknesses': ['LLM-Analyse fehlgeschlagen'],
                    'keyFindings': ['Automatische Score-Berechnung verwendet']
                },
                'improvementSuggestions': []
            }
    
    def _format_metrics(self, metrics: Dict[str, Any], model_type: str) -> str:
        """Formatiert Metriken für Prompt"""
        if not metrics:
            return "Keine Metriken verfügbar"
        
        lines = []
        for metric_name, value in metrics.items():
            lines.append(f"{metric_name}: {value:.4f}")
        
        return "\n".join(lines)
    
    def _validate_performance_analysis(self, result: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validiert Performance-Analyse-Ergebnis"""
        # Stelle sicher dass alle erforderlichen Felder vorhanden sind
        validated = {
            'overallScore': float(result.get('overallScore', 0.0)),
            'performanceGrade': result.get('performanceGrade', 'Poor'),
            'summary': result.get('summary', ''),
            'detailedAnalysis': result.get('detailedAnalysis', {}),
            'improvementSuggestions': result.get('improvementSuggestions', [])
        }
        
        # Validiere Score (0-10)
        validated['overallScore'] = max(0.0, min(10.0, validated['overallScore']))
        
        # Stelle sicher dass detailedAnalysis die richtige Struktur hat
        if not isinstance(validated['detailedAnalysis'], dict):
            validated['detailedAnalysis'] = {}
        
        for key in ['strengths', 'weaknesses', 'keyFindings']:
            if key not in validated['detailedAnalysis']:
                validated['detailedAnalysis'][key] = []
        
        return validated
    
    def _calculate_fallback_score(self, metrics: Dict[str, Any], model_type: str) -> float:
        """Berechnet Fallback-Score basierend auf Metriken"""
        if not metrics:
            return 0.0
        
        if model_type.lower() == 'classification':
            # Für Klassifikation: Fokus auf Accuracy, F1, Precision, Recall
            accuracy = metrics.get('accuracy', 0.0)
            f1 = metrics.get('f1_score', 0.0)
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            
            # Gewichteter Durchschnitt
            score = (accuracy * 0.4 + f1 * 0.3 + precision * 0.15 + recall * 0.15) * 10
        else:
            # Für Regression: Fokus auf R2, RMSE (invers)
            r2 = metrics.get('r_squared', 0.0)
            rmse = metrics.get('root_mean_squared_error', float('inf'))
            
            # Normalisiere RMSE (angenommen max RMSE = 100)
            normalized_rmse = max(0.0, min(1.0, 1.0 - (rmse / 100.0))) if rmse != float('inf') else 0.0
            
            # Gewichteter Durchschnitt
            score = (r2 * 0.7 + normalized_rmse * 0.3) * 10
        
        return max(0.0, min(10.0, score))
    
    def _score_to_grade(self, score: float) -> str:
        """Konvertiert Score zu Grade"""
        if score >= 8.5:
            return 'Excellent'
        elif score >= 7.0:
            return 'Good'
        elif score >= 5.0:
            return 'Fair'
        else:
            return 'Poor'

