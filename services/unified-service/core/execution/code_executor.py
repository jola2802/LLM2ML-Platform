"""
Python-Code-Ausführung
"""

import os
import subprocess
from typing import Dict, Any, Optional
from .code_validator import validate_python_code, apply_deterministic_fixes

def execute_python_script(
    script_path: str,
    script_dir: str,
    venv_dir: str,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Führt ein Python-Script aus mit Retry-Logik
    """
    with open(script_path, 'r', encoding='utf-8') as f:
        current_code = f.read()
    
    attempt = 0
    
    while attempt < max_retries:
        try:
            attempt += 1
            print(f'Python Script Ausführung - Versuch {attempt}/{max_retries}')
            
            # Code validieren und korrigieren
            try:
                validated_code = validate_python_code(current_code)
                if validated_code != current_code:
                    with open(script_path, 'w', encoding='utf-8') as f:
                        f.write(validated_code)
                    current_code = validated_code
                    print(f'Code wurde automatisch korrigiert (Versuch {attempt})')
            except Exception as validation_error:
                print(f'Code-Validierungsfehler: {validation_error}')
            
            # Bestimme Python-Executable
            if os.name == 'nt':  # Windows
                venv_path = os.path.join(venv_dir, 'Scripts', 'python.exe')
            else:  # Unix/Linux/Mac
                venv_path = os.path.join(venv_dir, 'bin', 'python')
            
            # Führe Script aus
            result = subprocess.run(
                [venv_path, script_path],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10 Minuten Timeout
            )
            
            stdout = result.stdout or ''
            stderr = result.stderr or ''
            full_output = stdout + stderr
            
            # Prüfe auf echte Fehler
            has_real_error = stderr.strip() and any(
                error_indicator in stderr
                for error_indicator in [
                    'Error:', 'Exception:', 'Traceback', 'SyntaxError',
                    'ImportError', 'ModuleNotFoundError', 'FileNotFoundError',
                    'PermissionError', 'exit(1)', 'sys.exit(1)'
                ]
            )
            
            if has_real_error:
                print(f'Echter Fehler bei Ausführung (Versuch {attempt}): {stderr}')
                
                # Versuche deterministische Korrekturen
                deterministically_fixed = apply_deterministic_fixes(current_code, stderr)
                if deterministically_fixed != current_code:
                    with open(script_path, 'w', encoding='utf-8') as f:
                        f.write(deterministically_fixed)
                    current_code = deterministically_fixed
                    print(f'Deterministische Korrektur angewendet (Versuch {attempt})')
                    continue  # Erneut versuchen
                
                if attempt >= max_retries:
                    return {'stdout': full_output, 'stderr': stderr}
            
            # Erfolgreiche Ausführung
            return {'stdout': full_output, 'stderr': ''}
            
        except subprocess.TimeoutExpired:
            print(f'Timeout bei Ausführung (Versuch {attempt})')
            if attempt >= max_retries:
                raise TimeoutError(f'Python execution timeout after {max_retries} attempts')
        except Exception as error:
            print(f'Fehler bei Ausführung (Versuch {attempt}): {error}')
            
            # Versuche deterministische Korrekturen
            deterministically_fixed = apply_deterministic_fixes(current_code, str(error))
            if deterministically_fixed != current_code and attempt < max_retries:
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(deterministically_fixed)
                current_code = deterministically_fixed
                print(f'Deterministische Korrektur (Exception) angewendet (Versuch {attempt})')
                continue
            
            if attempt >= max_retries:
                raise RuntimeError(f'Python execution failed after {max_retries} attempts: {error}')
    
    raise RuntimeError(f'Python execution failed after {max_retries} attempts')

