"""
Code-Generator-Worker-Agent
"""

from typing import Dict, Any, Optional
from core.agents.base_agent import BaseWorker
from shared.templates.training_template import CODE_TEMPLATE

class CodeGeneratorWorker(BaseWorker):
    def __init__(self):
        super().__init__('CODE_GENERATOR')
    
    async def execute(self, pipeline_state: Dict[str, Any]) -> str:
        """Generiert Python-Code"""
        self.log('info', 'Starte Code-Generierung')
        
        project = pipeline_state.get('project', {})
        results = pipeline_state.get('results', {})
        
        # Hole Hyperparameter-Vorschläge aus vorherigen Schritten
        hyperparameter_suggestions = results.get('HYPERPARAMETER_OPTIMIZER', {})
        
        # Generiere Python-Code mit Template
        code = self.generate_python_code(project, hyperparameter_suggestions, pipeline_state)
        
        self.log('success', 'Code-Generierung erfolgreich')
        return code
    
    def generate_python_code(self, project: Dict[str, Any], hyperparameter_suggestions: Dict[str, Any], pipeline_state: Dict[str, Any]) -> str:
        """Generiert Python-Code aus Template"""
        # Hole Template
        code = CODE_TEMPLATE
        
        # Adaptiere Header
        adapted_code = self.adapt_header(code, project, hyperparameter_suggestions, pipeline_state)
        
        return adapted_code
    
    def adapt_header(self, code: str, project: Dict[str, Any], hyperparameter_suggestions: Dict[str, Any], pipeline_state: Dict[str, Any]) -> str:
        """Passt Template-Header an Projekt an"""
        llm_recommendations = project.get('llmRecommendations', {})
        
        # Basis-Ersetzungen
        adapted_code = code.replace('PROJECT_NAME', f"'{project.get('name', 'project')}'")
        adapted_code = adapted_code.replace('FILE_PATH', f"r'{project.get('csvFilePath', '')}'")
        adapted_code = adapted_code.replace('TARGET_COLUMN', f"'{llm_recommendations.get('targetVariable', 'target')}'")
        adapted_code = adapted_code.replace('PROBLEM_TYPE', f"'{llm_recommendations.get('modelType', 'classification')}'")
        adapted_code = adapted_code.replace('MODEL_TYPE', f"'{llm_recommendations.get('algorithm', 'RandomForest')}'")
        
        # Hyperparameter
        hyperparameters = llm_recommendations.get('hyperparameters', {})
        if isinstance(hyperparameters, dict):
            library = hyperparameters.get('library', 'sklearn')
            params = hyperparameters.get('params', {})
            
            # Falls params verschachtelt ist (z.B. {'RandomForest': {'n_estimators': 100}})
            if params and isinstance(params, dict) and len(params) == 1:
                first_key = list(params.keys())[0]
                if isinstance(params[first_key], dict):
                    params = params[first_key]
        else:
            library = 'sklearn'
            params = {}
        
        # Falls Hyperparameter-Vorschläge vorhanden, verwende diese
        if hyperparameter_suggestions and isinstance(hyperparameter_suggestions, dict):
            suggested_params = hyperparameter_suggestions.get('hyperparameters', {})
            if suggested_params:
                params = suggested_params
        
        adapted_code = adapted_code.replace('MODEL_LIB', f"'{library}'")
        adapted_code = adapted_code.replace('MODEL_PARAMS', self.convert_to_python_syntax(params))
        adapted_code = adapted_code.replace('MODEL_SAVE_PATH', f"model_{project.get('id', 'unknown')}.pkl")
        
        # Generierte Features (aus Feature Engineer oder LLM Recommendations)
        results = pipeline_state.get('results', {})
        feature_engineer_result = results.get('FEATURE_ENGINEER', {})
        generated_features = feature_engineer_result.get('generatedFeatures', [])
        
        # Fallback: Falls keine Features vom Feature Engineer, verwende aus LLM Recommendations
        if not generated_features:
            generated_features = llm_recommendations.get('generatedFeatures', [])
        
        # Konvertiere Feature-Format für Template
        # AutoFeat gibt Feature-Namen (Strings) zurück, die bereits im DataFrame vorhanden sind
        # Das Template erwartet eine Liste von Dicts mit 'name' und 'formula' Keys für NEUE Features
        # Da AutoFeat-Features bereits generiert wurden, geben wir eine leere Liste zurück
        formatted_features = []
        if generated_features and isinstance(generated_features, list) and len(generated_features) > 0:
            # Prüfe ob erste Feature ein String ist (AutoFeat-Format)
            if isinstance(generated_features[0], str):
                # AutoFeat-Features sind bereits im DataFrame - keine Neugenerierung nötig
                formatted_features = []
            elif isinstance(generated_features[0], dict):
                # Bereits im richtigen Format (von LLM)
                formatted_features = generated_features
        
        adapted_code = adapted_code.replace('GENERATED_FEATURES', self.convert_to_python_syntax(formatted_features))
        
        return adapted_code
    
    def convert_to_python_syntax(self, obj: Any, indent: int = 0) -> str:
        """Konvertiert Python-Werte zu Python-Syntax-String"""
        indent_str = ' ' * indent
        next_indent = indent + 4
        
        if obj is None:
            return 'None'
        
        if isinstance(obj, bool):
            return 'True' if obj else 'False'
        
        if isinstance(obj, str):
            # Prüfe ob String mehrzeilig ist oder komplexe Code-Snippets enthält
            has_newlines = '\n' in obj or '\r' in obj
            has_quotes = "'" in obj or '"' in obj
            has_backslashes = '\\' in obj
            is_code_snippet = 'import ' in obj or 'def ' in obj or 'return ' in obj or '=' in obj and 'pd.' in obj
            
            # Für mehrzeilige Strings oder Code-Snippets: verwende Triple-Quotes
            if has_newlines or (is_code_snippet and has_quotes):
                # Escape Triple-Quotes falls vorhanden (selten, aber möglich)
                escaped = obj.replace('"""', '\\"\\"\\"')
                return f'"""{escaped}"""'
            
            # Für normale Strings: verwende einfache Anführungszeichen mit Escaping
            # WICHTIG: Reihenfolge ist kritisch - Backslashes zuerst!
            escaped = obj
            escaped = escaped.replace('\\', '\\\\')  # Backslashes zuerst (wichtig!)
            escaped = escaped.replace("'", "\\'")  # Dann einfache Anführungszeichen
            escaped = escaped.replace('\n', '\\n')  # Newlines
            escaped = escaped.replace('\r', '\\r')  # Carriage returns
            escaped = escaped.replace('\t', '\\t')  # Tabs
            
            return f"'{escaped}'"
        
        if isinstance(obj, (int, float)):
            return str(obj)
        
        if isinstance(obj, list):
            if len(obj) == 0:
                return '[]'
            items = [self.convert_to_python_syntax(item, next_indent) for item in obj]
            return '[\n' + ',\n'.join(' ' * next_indent + item for item in items) + '\n' + indent_str + ']'
        
        if isinstance(obj, dict):
            if len(obj) == 0:
                return '{}'
            items = []
            for key, value in obj.items():
                key_str = self.convert_to_python_syntax(key, next_indent)
                value_str = self.convert_to_python_syntax(value, next_indent)
                items.append(' ' * next_indent + f"{key_str}: {value_str}")
            return '{\n' + ',\n'.join(items) + '\n' + indent_str + '}'
        
        # Fallback: JSON
        import json
        return json.dumps(obj, ensure_ascii=False)

