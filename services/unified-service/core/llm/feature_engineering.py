"""
Feature Engineering Empfehlungen
"""

from typing import Optional, Dict, Any, List
from core.llm import llm
from infrastructure.clients.python_client import python_client
from shared.utils.data_processing import (
    filter_columns, filter_data_overview_for_features,
    extract_and_validate_json, build_data_overview
)

def get_feature_engineering_recommendations(
    analysis: Optional[Dict[str, Any]] = None,
    file_path: Optional[str] = None,
    selected_features: Optional[List[str]] = None,
    excluded_features: Optional[List[str]] = None,
    user_preferences: Optional[str] = None
) -> Dict[str, Any]:
    """Feature Engineering Empfehlungen"""
    from core.agents.prompts import FEATURE_ENGINEER_PROMPT, format_prompt
    
    data_overview = _get_data_overview(analysis, file_path, selected_features, excluded_features)
    
    if not data_overview:
        raise ValueError('Keine Daten verfügbar für Feature Engineering')
    
    prompt = format_prompt(FEATURE_ENGINEER_PROMPT, {
        'dataOverview': data_overview,
        'userPreferences': user_preferences or 'No specific preferences provided. Focus on feature engineering only.'
    })
    
    response = llm.call_llm_api(prompt, None, None, 3)
    
    try:
        result = extract_and_validate_json(response)
    except ValueError as e:
        print(f'[Feature Engineering] Fehler bei JSON-Extraktion: {e}')
        # Fallback: Leeres Ergebnis zurückgeben
        return {
            'generatedFeatures': [],
            'reasoning': f'Fehler beim Parsen der LLM-Response: {str(e)}'
        }
    
    return {
        'generatedFeatures': result.get('generatedFeatures', []),
        'reasoning': result.get('reasoning', 'Feature Engineering abgeschlossen')
    }

def _get_data_overview(
    analysis: Optional[Dict[str, Any]],
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
            print(f'[Feature Engineering] Führe automatische Datenexploration durch für {file_path}')
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
    if not data_overview and analysis:
        columns = analysis.get('columns', [])
        filtered_columns = filter_columns(columns, selected_features, excluded_features)
        data_overview = f"Available columns: {', '.join(filtered_columns)}"
    
    return data_overview

