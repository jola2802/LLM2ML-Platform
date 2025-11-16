"""
Performance-Evaluation mit LLM
"""

import json
from typing import Dict, Any
from core.llm import llm
from shared.utils.data_processing import extract_and_validate_json

def evaluate_performance_with_llm(project: Dict[str, Any]) -> Dict[str, Any]:
    """Performance-Evaluation mit LLM"""
    from core.agents.prompts import PERFORMANCE_EVALUATION_PROMPT, format_prompt
    
    prompt = format_prompt(PERFORMANCE_EVALUATION_PROMPT, {
        'projectName': project.get('name', 'Unknown'),
        'algorithm': project.get('algorithm', 'Unknown'),
        'modelType': project.get('modelType', 'Unknown'),
        'targetVariable': project.get('targetVariable', 'Unknown'),
        'features': ', '.join(project.get('features', [])),
        'dataSourceName': project.get('dataSourceName', 'Unknown'),
        'performanceMetrics': json.dumps(project.get('performanceMetrics', {}), indent=2),
        'recommendations': json.dumps(project.get('llmRecommendations', {}), indent=2)
    })
    
    response = llm.call_llm_api(prompt, None, None, 3)
    result = extract_and_validate_json(response)
    
    return result

