"""
Decision Worker-Agent
Entscheidet ob Ergebnis gut genug ist oder Loop wiederholt werden soll
"""

from typing import Dict, Any
from core.agents.base_agent import BaseWorker
from core.agents.prompts import (
    DECISION_PROMPT,
    format_prompt
)
from shared.utils.data_processing import extract_and_validate_json

class DecisionWorker(BaseWorker):
    def __init__(self):
        super().__init__('DECISION')
    
    async def execute(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Entscheidet ob Loop wiederholt werden soll"""
        self.log('info', 'Starte Entscheidungsfindung')
        
        project = pipeline_state.get('project', {})
        results = pipeline_state.get('results', {})
        
        # Hole Performance-Analyse
        performance_analysis = results.get('PERFORMANCE_ANALYZER', {})
        if not performance_analysis:
            # Fallback: Keine Performance-Daten verfügbar
            self.log('warning', 'Keine Performance-Analyse verfügbar')
            return {
                'shouldContinue': False,
                'reason': 'Keine Performance-Daten verfügbar',
                'iteration': pipeline_state.get('iteration', 0)
            }
        
        # Hole bisherige Iterationen
        iteration = pipeline_state.get('iteration', 0)
        max_iterations = pipeline_state.get('maxIterations', 3)
        
        # Prüfe ob maximale Iterationen erreicht
        if iteration >= max_iterations:
            self.log('info', f'Maximale Iterationen ({max_iterations}) erreicht')
            return {
                'shouldContinue': False,
                'reason': f'Maximale Iterationen ({max_iterations}) erreicht',
                'iteration': iteration
            }
        
        # Hole Performance-Metriken
        overall_score = performance_analysis.get('overallScore', 0.0)
        performance_grade = performance_analysis.get('performanceGrade', 'Poor')
        
        # Entscheide mit LLM basierend auf Performance
        decision = await self._make_decision(
            project=project,
            performance_analysis=performance_analysis,
            iteration=iteration,
            max_iterations=max_iterations
        )
        
        should_continue = decision.get('shouldContinue', False)
        reason = decision.get('reason', '')
        
        if should_continue:
            self.log('info', f'Entscheidung: Loop fortsetzen - {reason}')
        else:
            self.log('success', f'Entscheidung: Loop beenden - {reason}')
        
        result = {
            'shouldContinue': should_continue,
            'reason': reason,
            'iteration': iteration,
            'overallScore': overall_score,
            'performanceGrade': performance_grade,
            'suggestions': decision.get('suggestions', [])
        }
        
        return result
    
    async def _make_decision(
        self,
        project: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        iteration: int,
        max_iterations: int
    ) -> Dict[str, Any]:
        """Trifft Entscheidung mit LLM"""
        
        # Erstelle Prompt
        prompt = format_prompt(DECISION_PROMPT, {
            'projectName': project.get('name', 'Unknown'),
            'iteration': iteration,
            'maxIterations': max_iterations,
            'overallScore': performance_analysis.get('overallScore', 0.0),
            'performanceGrade': performance_analysis.get('performanceGrade', 'Poor'),
            'summary': performance_analysis.get('summary', ''),
            'strengths': str(performance_analysis.get('detailedAnalysis', {}).get('strengths', [])),
            'weaknesses': str(performance_analysis.get('detailedAnalysis', {}).get('weaknesses', [])),
            'improvementSuggestions': str(performance_analysis.get('improvementSuggestions', []))
        })
        
        try:
            # Rufe LLM auf
            self.log('info', 'Rufe LLM auf für Entscheidungsfindung')
            response = await self.call_llm(prompt, None, self.config.get('maxTokens', 2048))
            
            # Extrahiere JSON aus Response
            result = extract_and_validate_json(response)
            
            return {
                'shouldContinue': result.get('shouldContinue', False),
                'reason': result.get('reason', ''),
                'suggestions': result.get('suggestions', [])
            }
            
        except Exception as error:
            self.log('error', f'Fehler bei Entscheidungsfindung: {error}')
            
            # Fallback: Entscheide basierend auf Score
            overall_score = performance_analysis.get('overallScore', 0.0)
            should_continue = overall_score < 7.0 and iteration < max_iterations
            
            return {
                'shouldContinue': should_continue,
                'reason': f'Fallback: Score {overall_score:.2f}, Iteration {iteration}/{max_iterations}',
                'suggestions': []
            }

