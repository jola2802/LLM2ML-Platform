"""
Feature Engineer Worker-Agent
Generiert neue Features mit AutoFeat
"""

from typing import Dict, Any
from core.agents.base_agent import BaseWorker

class FeatureEngineerWorker(BaseWorker):
    def __init__(self):
        super().__init__('FEATURE_ENGINEER')
    
    async def execute(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Führt Feature Engineering mit AutoFeat aus"""
        self.log('info', 'Starte Feature Engineering mit AutoFeat')
        
        project = pipeline_state.get('project', {})
        user_preferences = project.get('userPreferences', '')
        csv_file_path = project.get('csvFilePath', '')
        
        if not csv_file_path:
            self.log('error', 'Kein CSV-Pfad verfügbar')
            return {
                'generatedFeatures': [],
                'reasoning': 'Fehler: Kein CSV-Pfad verfügbar'
            }
        
        # Generiere Features mit AutoFeat
        generated_features = await self._generate_features(
            user_preferences=user_preferences,
            csv_file_path=csv_file_path
        )
        
        result = {
            'generatedFeatures': generated_features.get('generatedFeatures', []),
            'reasoning': generated_features.get('reasoning', '')
        }
        
        self.log('success', f'Feature Engineering erfolgreich - {len(result["generatedFeatures"])} Features generiert')
        return result
    
    async def _generate_features(
        self,
        user_preferences: str,
        csv_file_path: str = ''
    ) -> Dict[str, Any]:
        """Generiert Features mit AutoFeat"""
        
        try:
            from autofeat import AutoFeatRegressor
            import pandas as pd
            import re

            data = pd.read_csv(csv_file_path)

            # Prüfe auf Zielspalte (Versuche aus user_preferences extrahieren, sonst nehme die letzte Spalte als Fallback)
            possible_target = None
            
            # Versuche aus expliziten Angaben zu nehmen:
            if hasattr(self, "target_column") and self.target_column in data.columns:
                possible_target = self.target_column
            else:
                # Suche nach möglichen Schlüsselworten in user_preferences:
                if user_preferences:
                    m = re.search(r"(target|Ziel(?:variable)?|predict|Vorhersage)[^\w]*[:=]*[\"']?([\w\-]+)[\"']?", user_preferences, re.IGNORECASE)
                    if m and m.group(2) in data.columns:
                        possible_target = m.group(2)
                
                # Fallback: letzte Spalte
                if possible_target is None and data.columns.size > 0:
                    possible_target = data.columns[-1]

            if possible_target is None or possible_target not in data.columns:
                raise ValueError("Keine Zielspalte gefunden für AutoFeat.")

            X = data.drop(columns=[possible_target])
            y = data[possible_target]

            self.log('info', f'Generiere Features mit AutoFeat (Zielspalte: {possible_target})')
            af = AutoFeatRegressor()
            features = af.fit_transform(X, y)

            feature_names = features.columns.tolist()
            self.log('info', f'Generated Features: {len(feature_names)} Features')
            
            return {
                'generatedFeatures': feature_names,
                'reasoning': f'Features generiert mit AutoFeat (Zielspalte: {possible_target})'
            }
            
        except Exception as error:
            self.log('error', f'Fehler bei Feature Engineering: {error}')
            raise

