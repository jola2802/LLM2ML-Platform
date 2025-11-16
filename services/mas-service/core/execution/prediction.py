"""
Prediction-Script-Generierung und -Ausführung
"""

import os
import json
import hashlib
import re
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
from .code_executor import execute_python_script

def generate_project_hash(project: Dict[str, Any]) -> str:
    """Generiert Hash für Projekt-Änderungserkennung"""
    relevant_data = {
        'id': project.get('id'),
        'modelType': project.get('modelType'),
        'algorithm': project.get('algorithm'),
        'hyperparameters': project.get('hyperparameters'),
        'pythonCode': project.get('pythonCode'),
        'originalPythonCode': project.get('originalPythonCode'),
        'modelPath': project.get('modelPath'),
        'features': project.get('features'),
        'targetVariable': project.get('targetVariable')
    }
    
    data_string = json.dumps(relevant_data, sort_keys=True)
    return hashlib.sha256(data_string.encode()).hexdigest()

def convert_to_python_syntax(obj: Any, indent: int = 0) -> str:
    """Konvertiert Objekt zu Python-Syntax-String"""
    if isinstance(obj, dict):
        items = []
        for key, value in obj.items():
            key_str = f"'{key}'" if isinstance(key, str) else str(key)
            value_str = convert_to_python_syntax(value, indent + 1)
            items.append(f"{'  ' * (indent + 1)}{key_str}: {value_str}")
        return '{\n' + ',\n'.join(items) + '\n' + '  ' * indent + '}'
    elif isinstance(obj, list):
        items = [convert_to_python_syntax(item, indent + 1) for item in obj]
        return '[' + ', '.join(items) + ']'
    elif isinstance(obj, str):
        return f"'{obj}'"
    else:
        return str(obj)

def convert_input_features(input_features: Dict[str, Any]) -> Dict[str, Any]:
    """Konvertiert Input-Features für Prediction"""
    # Vereinfachte Implementierung
    return input_features

def generate_prediction_script(
    project: Dict[str, Any],
    input_features: Dict[str, Any],
    script_dir: str,
    models_dir: str
) -> str:
    """Generiert Prediction-Script"""
    # Lade Template (vereinfacht - Template sollte aus separater Datei kommen)
    template = _get_prediction_template()
    
    model_file_name = os.path.basename(project.get('modelPath', f"model_{project.get('id')}.pkl"))
    model_path = os.path.join(models_dir, model_file_name)
    
    encoder_path = os.path.join(models_dir, f"model_{project.get('id')}_encoder.pkl")
    scaler_path = os.path.join(models_dir, f"model_{project.get('id')}_scaler.pkl")
    
    script = template.replace('PROJECT_ID', str(project.get('id')))
    script = script.replace('MODEL_PATH', model_path)
    script = script.replace('INPUT_FEATURES', convert_to_python_syntax(input_features))
    script = script.replace('PROBLEM_TYPE', project.get('modelType', 'classification').lower())
    script = script.replace('ENCODER_PATH', encoder_path)
    script = script.replace('SCALER_PATH', scaler_path)
    
    return script

def _get_prediction_template() -> str:
    """Gibt Prediction-Template zurück"""
    return """import joblib
import sys
import json

project_id = 'PROJECT_ID'
model_path = r'MODEL_PATH'
input_data = INPUT_FEATURES
problem_type = 'PROBLEM_TYPE'

# Load model
model = joblib.load(model_path)

# Make prediction
prediction = model.predict([list(input_data.values())])[0]

# Output
print(json.dumps({'prediction': float(prediction) if problem_type == 'regression' else int(prediction)}))
"""

def predict_with_model(
    project: Dict[str, Any],
    input_features: Dict[str, Any],
    script_dir: str,
    venv_dir: str,
    models_dir: str
) -> Dict[str, Any]:
    """Führt Prediction mit Modell aus"""
    script_path = os.path.join(script_dir, f"predict_{project.get('id')}.py")
    metadata_path = os.path.join(script_dir, f"predict_{project.get('id')}_metadata.json")
    
    should_regenerate = True
    
    # Prüfe ob Script existiert und aktuell ist
    if os.path.exists(script_path) and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            current_hash = generate_project_hash(project)
            if metadata.get('projectHash') == current_hash:
                should_regenerate = False
        except Exception:
            pass
    
    # Generiere Script falls nötig
    if should_regenerate:
        prediction_script = generate_prediction_script(project, convert_input_features(input_features), script_dir, models_dir)
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(prediction_script)
        
        metadata = {
            'projectId': project.get('id'),
            'projectHash': generate_project_hash(project),
            'version': '1.0',
            'createdAt': datetime.now().isoformat(),
            'lastUsed': datetime.now().isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    else:
        # Aktualisiere lastUsed Timestamp
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata['lastUsed'] = datetime.now().isoformat()
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception:
            pass
    
    # Führe Script aus
    result = execute_python_script(script_path, script_dir, venv_dir)
    
    # Parse Prediction aus Output
    output = result.get('stdout', '')
    try:
        prediction_data = json.loads(output)
        return prediction_data.get('prediction')
    except json.JSONDecodeError:
        # Fallback: Versuche Prediction aus Output zu extrahieren
        match = re.search(r'prediction["\']?\s*:\s*([\d.]+)', output)
        if match:
            return float(match.group(1))
        raise ValueError(f'Konnte Prediction nicht aus Output extrahieren: {output}')

