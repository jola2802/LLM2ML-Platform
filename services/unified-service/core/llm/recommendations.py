"""
LLM-Empfehlungen für ML-Pipeline
"""

from typing import Optional, Dict, Any, List
from core.llm import llm
from infrastructure.clients.python_client import python_client
from shared.utils.data_processing import (
    filter_columns, filter_data_overview_for_features,
    extract_and_validate_json, build_data_overview
)

def get_llm_recommendations(
    analysis: Dict[str, Any],
    file_path: Optional[str] = None,
    selected_features: Optional[List[str]] = None,
    excluded_features: Optional[List[str]] = None,
    user_preferences: Optional[str] = None
) -> Dict[str, Any]:
    """LLM-Empfehlungen für Algorithmus und Features"""
    from core.agents.prompts import LLM_RECOMMENDATIONS_PROMPT, format_prompt, get_hyperparameters_info
    from shared.utils.algorithms import ALGORITHMS
    
    data_overview = _get_data_overview(analysis, file_path, selected_features, excluded_features)
    
    prompt = format_prompt(LLM_RECOMMENDATIONS_PROMPT, {
        'dataOverview': data_overview,
        'userPreferences': user_preferences or 'No specific preferences provided.',
        'hyperparametersInfo': get_hyperparameters_info(ALGORITHMS)
    })
    
    # LLM-API-Call
    response = llm.call_llm_api(prompt, None, None, 3)
    recommendations = extract_and_validate_json(response)
    
    # Validiere Features
    _validate_recommendations(recommendations, analysis, selected_features, excluded_features)
    
    return recommendations

def _get_data_overview(
    analysis: Dict[str, Any],
    file_path: Optional[str],
    selected_features: Optional[List[str]],
    excluded_features: Optional[List[str]]
) -> str:
    """Holt oder erstellt Data Overview"""
    data_overview = ''
    
    # Versuche zuerst, Data Overview aus der bereits vorhandenen Analysis zu extrahieren
    if analysis and analysis.get('analysis_summary'):
        # Verwende die bereits vorhandene LLM-Summary aus der Analysis
        data_overview = filter_data_overview_for_features(
            analysis.get('analysis_summary', ''),
            selected_features,
            excluded_features
        )
    
    # Falls keine Summary vorhanden, versuche automatische Datenexploration
    if not data_overview and file_path:
        try:
            print(f'[Recommendations] Führe automatische Datenexploration durch für {file_path}')
            data_analysis = python_client.analyze_data(file_path, False)
            if data_analysis.get('success'):
                data_overview = filter_data_overview_for_features(
                    data_analysis.get('llm_summary', ''),
                    selected_features,
                    excluded_features
                )
        except Exception as error:
            print(f'Fehler bei automatischer Datenexploration: {error}')
    
    # Fallback: Baue Data Overview aus Analysis-Daten
    if not data_overview:
        data_overview = build_data_overview(analysis, selected_features, excluded_features)
    
    return data_overview

def _validate_recommendations(
    recommendations: Dict[str, Any],
    analysis: Dict[str, Any],
    selected_features: Optional[List[str]],
    excluded_features: Optional[List[str]]
):
    """Validiert und bereinigt Recommendations"""
    available_columns = filter_columns(analysis.get('columns', []), selected_features, excluded_features)
    
    # Entferne Target-Variable aus Features
    target = recommendations.get('targetVariable')
    recommendations['features'] = [
        f for f in recommendations.get('features', [])
        if f != target and f in available_columns
    ]
    
    # Sicherstellen, dass generierte Features vorhanden sind
    if 'generatedFeatures' not in recommendations:
        recommendations['generatedFeatures'] = []
    
    # Validierung
    if not recommendations.get('features'):
        raise ValueError('No valid features found')

