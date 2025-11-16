"""
LLM-API-Funktionen für ML-Empfehlungen (Legacy - wird durch separate Module ersetzt)
"""

# Diese Datei wird für Backward-Compatibility beibehalten
# Neue Implementierungen sind in recommendations.py, feature_engineering.py, performance.py

from .recommendations import get_llm_recommendations
from .feature_engineering import get_feature_engineering_recommendations
from .performance import evaluate_performance_with_llm

__all__ = [
    'get_llm_recommendations',
    'get_feature_engineering_recommendations',
    'evaluate_performance_with_llm'
]

