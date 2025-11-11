"""
Code-Validierung und -Korrektur
"""

import re
from typing import Tuple

def validate_python_code(python_code: str) -> str:
    """Validiert und korrigiert Python-Code"""
    if not python_code or not isinstance(python_code, str):
        return ''
    
    # Sicherstellen, dass wichtige Imports vorhanden sind
    if 'import pandas' not in python_code:
        python_code = 'import pandas as pd\n' + python_code
    
    if 'import joblib' not in python_code:
        python_code = python_code.replace('import pandas as pd', 'import pandas as pd\nimport joblib')
    
    # Entferne Markdown-Code-Blocks
    python_code = re.sub(r'```python', '', python_code)
    python_code = re.sub(r'```json', '', python_code)
    python_code = re.sub(r'```', '', python_code)
    python_code = python_code.replace('`', '')
    
    return python_code.strip()

def apply_deterministic_fixes(python_code: str, error_text: str) -> str:
    """Wendet deterministische Korrekturen für bekannte Fehler an"""
    try:
        fixed = python_code
        
        # Bekannte Fehler-Korrekturen
        if 'NameError' in error_text or 'is not defined' in error_text:
            # Versuche fehlende Imports hinzuzufügen
            if 'pd' in fixed and 'import pandas' not in fixed:
                fixed = 'import pandas as pd\n' + fixed
            if 'np' in fixed and 'import numpy' not in fixed:
                fixed = 'import numpy as np\n' + fixed
        
        return fixed
    except Exception:
        return python_code

