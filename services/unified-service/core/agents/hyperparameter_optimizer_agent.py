"""
Hyperparameter-Optimizer-Worker-Agent
Analysiert den Projekt-Kontext und generiert optimierte Hyperparameter für den gewählten Algorithmus
"""

from typing import Dict, Any, Optional
from core.agents.base_agent import BaseWorker
from core.agents.prompts import (
    HYPERPARAMETER_OPTIMIZATION_PROMPT,
    format_prompt,
    get_algorithm_hyperparameter_info
)
from shared.utils.algorithms import ALGORITHMS
from shared.utils.data_processing import extract_and_validate_json
from infrastructure.clients.python_client import python_client

class HyperparameterOptimizerWorker(BaseWorker):
    def __init__(self):
        super().__init__('HYPERPARAMETER_OPTIMIZER')
    
    async def execute(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Führt Hyperparameter-Optimierung aus"""
        self.log('info', 'Starte Hyperparameter-Optimierung')
        
        project = pipeline_state.get('project', {})
        recommendations = project.get('llmRecommendations', {})
        
        # Extrahiere relevante Informationen
        algorithm = recommendations.get('algorithm', 'RandomForestClassifier')
        model_type = recommendations.get('modelType', 'Classification')
        target_variable = recommendations.get('targetVariable', '')
        features = recommendations.get('features', [])
        csv_file_path = project.get('csvFilePath', '')
        
        # Hole Daten-Charakteristika
        data_characteristics = self._get_data_characteristics(csv_file_path, features)
        
        # Hole aktuelle Hyperparameter (falls vorhanden)
        current_hyperparameters = recommendations.get('hyperparameters', {})
        
        # Generiere optimierte Hyperparameter mit LLM
        optimized_hyperparameters = await self._optimize_hyperparameters(
            algorithm=algorithm,
            model_type=model_type,
            target_variable=target_variable,
            num_features=len(features),
            data_characteristics=data_characteristics,
            current_hyperparameters=current_hyperparameters
        )
        
        result = {
            'hyperparameters': optimized_hyperparameters.get('hyperparameters', {}),
            'algorithm': algorithm,
            'reasoning': optimized_hyperparameters.get('reasoning', ''),
            'expectedPerformance': optimized_hyperparameters.get('expectedPerformance', ''),
            'tuningStrategy': optimized_hyperparameters.get('tuningStrategy', '')
        }
        
        self.log('success', f'Hyperparameter-Optimierung erfolgreich für {algorithm}')
        return result
    
    def _get_data_characteristics(self, file_path: str, features: list) -> str:
        """Holt Daten-Charakteristika für den Prompt"""
        if not file_path:
            return f"Number of features: {len(features)}"
        
        try:
            # Hole Data Analysis
            data_analysis = python_client.analyze_data(file_path, False)
            
            if data_analysis.get('success'):
                exploration = data_analysis.get('exploration', {})
                row_count = exploration.get('rowCount', 0)
                columns = exploration.get('columns', [])
                
                # Erstelle Charakteristik-String
                characteristics = [
                    f"Dataset size: {row_count} rows",
                    f"Number of features: {len(features)}",
                    f"Total columns: {len(columns)}"
                ]
                
                # Füge Datentyp-Informationen hinzu
                data_types = exploration.get('dataTypes', {})
                numeric_count = sum(1 for dt in data_types.values() if dt == 'numeric')
                categorical_count = sum(1 for dt in data_types.values() if dt == 'categorical')
                
                if numeric_count > 0:
                    characteristics.append(f"Numeric columns: {numeric_count}")
                if categorical_count > 0:
                    characteristics.append(f"Categorical columns: {categorical_count}")
                
                # Füge statistische Informationen hinzu (falls verfügbar)
                numeric_stats = exploration.get('numericStats', {})
                if numeric_stats:
                    characteristics.append("Numeric statistics available for analysis")
                
                return "\n".join(characteristics)
            else:
                return f"Number of features: {len(features)}"
        except Exception as error:
            self.log('warning', f'Fehler beim Abrufen der Daten-Charakteristika: {error}')
            return f"Number of features: {len(features)}"
    
    async def _optimize_hyperparameters(
        self,
        algorithm: str,
        model_type: str,
        target_variable: str,
        num_features: int,
        data_characteristics: str,
        current_hyperparameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimiert Hyperparameter mit LLM"""
        
        # Bestimme Problem-Typ
        problem_type = 'classification' if 'Classification' in model_type or 'Classifier' in algorithm else 'regression'
        
        # Hole Hyperparameter-Info für den Algorithmus
        hyperparameter_info = get_algorithm_hyperparameter_info(algorithm, ALGORITHMS)
        
        # Berechne geschätzte Anzahl Samples (falls nicht verfügbar)
        # Versuche aus data_characteristics zu extrahieren
        num_samples = 1000  # Default
        if 'rows' in data_characteristics.lower():
            try:
                import re
                match = re.search(r'(\d+)\s*rows', data_characteristics, re.IGNORECASE)
                if match:
                    num_samples = int(match.group(1))
            except:
                pass
        
        # Erstelle Prompt
        prompt = format_prompt(HYPERPARAMETER_OPTIMIZATION_PROMPT, {
            'algorithm': algorithm,
            'modelType': model_type,
            'targetVariable': target_variable,
            'numFeatures': num_features,
            'numSamples': num_samples,
            'problemType': problem_type,
            'dataCharacteristics': data_characteristics,
            'hyperparameterInfo': hyperparameter_info,
            'currentHyperparameters': str(current_hyperparameters) if current_hyperparameters else 'None'
        })
        
        try:
            # Rufe LLM auf
            self.log('info', f'Rufe LLM auf für Hyperparameter-Optimierung von {algorithm}')
            response = await self.call_llm(prompt, None, self.config.get('maxTokens', 2048))
            
            # Extrahiere JSON aus Response
            result = extract_and_validate_json(response)
            
            # Validiere und bereinige Hyperparameter
            validated_hyperparameters = self._validate_hyperparameters(
                result.get('hyperparameters', {}),
                algorithm
            )
            
            return {
                'hyperparameters': validated_hyperparameters,
                'reasoning': result.get('reasoning', 'Hyperparameter optimiert basierend auf Daten-Charakteristika'),
                'expectedPerformance': result.get('expectedPerformance', ''),
                'tuningStrategy': result.get('tuningStrategy', '')
            }
            
        except Exception as error:
            self.log('error', f'Fehler bei Hyperparameter-Optimierung: {error}')
            
            # Fallback: Verwende Standard-Hyperparameter aus ALGORITHMS
            fallback_params = ALGORITHMS.get(algorithm, {}).get('hyperparameters', {})
            
            self.log('warning', f'Verwende Fallback-Hyperparameter für {algorithm}')
            return {
                'hyperparameters': fallback_params.copy(),
                'reasoning': f'Fallback: Standard-Hyperparameter für {algorithm} verwendet (LLM-Optimierung fehlgeschlagen)',
                'expectedPerformance': 'Standard performance expected',
                'tuningStrategy': 'Default parameters'
            }
    
    def _validate_hyperparameters(
        self,
        hyperparameters: Dict[str, Any],
        algorithm: str
    ) -> Dict[str, Any]:
        """Validiert und bereinigt Hyperparameter"""
        # Hole erwartete Hyperparameter für den Algorithmus
        expected_params = ALGORITHMS.get(algorithm, {}).get('hyperparameters', {})
        
        validated = {}
        
        # Validiere jeden Hyperparameter
        for param_name, param_value in hyperparameters.items():
            # Prüfe ob Parameter für diesen Algorithmus gültig ist
            if param_name in expected_params:
                # Validiere Typ und Wert
                expected_type = type(expected_params[param_name])
                
                try:
                    # Konvertiere zu erwartetem Typ
                    if expected_type == int:
                        validated[param_name] = int(float(param_value))
                    elif expected_type == float:
                        validated[param_name] = float(param_value)
                    elif expected_type == bool:
                        validated[param_name] = bool(param_value)
                    elif param_value is None:
                        validated[param_name] = None
                    else:
                        validated[param_name] = param_value
                    
                    # Zusätzliche Validierungen für spezifische Parameter
                    validated[param_name] = self._validate_specific_parameter(
                        param_name,
                        validated[param_name],
                        algorithm
                    )
                    
                except (ValueError, TypeError) as e:
                    self.log('warning', f'Ungültiger Wert für {param_name}: {param_value}, verwende Default')
                    validated[param_name] = expected_params[param_name]
            else:
                # Unbekannter Parameter - ignoriere oder logge Warnung
                self.log('warning', f'Unbekannter Hyperparameter {param_name} für {algorithm}, ignoriert')
        
        # Füge fehlende Standard-Parameter hinzu
        for param_name, default_value in expected_params.items():
            if param_name not in validated:
                validated[param_name] = default_value
        
        return validated
    
    def _validate_specific_parameter(
        self,
        param_name: str,
        value: Any,
        algorithm: str
    ) -> Any:
        """Validiert spezifische Parameter mit domänenspezifischen Regeln"""
        
        # n_estimators: sollte positiv sein
        if param_name == 'n_estimators':
            if isinstance(value, (int, float)):
                return max(1, int(value))
        
        # max_depth: sollte positiv sein oder None
        if param_name == 'max_depth':
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return max(1, int(value))
        
        # learning_rate: sollte zwischen 0 und 1 sein
        if param_name == 'learning_rate':
            if isinstance(value, (int, float)):
                return max(0.001, min(1.0, float(value)))
        
        # C (Regularization): sollte positiv sein
        if param_name == 'C':
            if isinstance(value, (int, float)):
                return max(0.001, float(value))
        
        # min_samples_split: sollte mindestens 2 sein
        if param_name == 'min_samples_split':
            if isinstance(value, (int, float)):
                return max(2, int(value))
        
        # max_iter: sollte positiv sein
        if param_name == 'max_iter':
            if isinstance(value, (int, float)):
                return max(1, int(value))
        
        return value
