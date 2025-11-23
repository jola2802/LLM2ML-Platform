"""
Code Executor Worker-Agent
Führt generierten Python-Code aus (ohne LLM)
"""

import os
import tempfile
from typing import Dict, Any
from core.agents.base_agent import BaseWorker
from core.execution.code_executor import execute_python_script

class CodeExecutorWorker(BaseWorker):
    def __init__(self):
        super().__init__('CODE_EXECUTOR')
    
    async def execute(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Führt generierten Code aus"""
        self.log('info', 'Starte Code-Ausführung')
        
        results = pipeline_state.get('results', {})
        project = pipeline_state.get('project', {})
        
        # Hole generierten Code
        code_gen_result = results.get('CODE_GENERATOR', '')
        
        # Handle both string and dict results
        if isinstance(code_gen_result, str):
            code = code_gen_result
        elif isinstance(code_gen_result, dict):
            code = code_gen_result.get('code', '')
        else:
            code = ''
        
        if not code:
            raise ValueError('Kein Code zum Ausführen verfügbar')
        
        # Prüfe ob Code-Review verfügbar ist
        code_review = results.get('CODE_REVIEWER', {})
        if code_review and isinstance(code_review, dict) and code_review.get('reviewPerformed'):
            is_safe = code_review.get('isSafe', False)
            improved_code = code_review.get('improvedCode', '')
            
            if is_safe and improved_code:
                self.log('info', 'Verwende verbesserten Code vom Code-Reviewer')
                code = improved_code
            elif not is_safe:
                self.log('warning', f'Code-Review fand kritische Probleme - verwende Original-Code mit Vorsicht')
                # Optional: Könnte hier auch abbrechen
                # raise ValueError('Code ist nicht sicher zur Ausführung')
        
        
        # Erstelle temporäre Datei für Code
        script_dir = tempfile.mkdtemp()
        script_path = os.path.join(script_dir, 'training_script.py')
        
        try:
            # Schreibe Code in Datei
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            self.log('info', f'Code in temporäre Datei geschrieben: {script_path}')
            
            # Bestimme venv-Verzeichnis (falls vorhanden)
            # Standard: verwende System-Python
            venv_dir = os.environ.get('VENV_DIR', '')
            if not venv_dir:
                # Versuche venv im Projekt-Verzeichnis zu finden
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                venv_dir = os.path.join(project_root, 'venv')
                if not os.path.exists(venv_dir):
                    venv_dir = ''  # Verwende System-Python
            
            # Führe Code aus
            try:
                execution_result = execute_python_script(
                    script_path=script_path,
                    script_dir=script_dir,
                    venv_dir=venv_dir,
                    max_retries=3
                )
            except Exception as exec_error:
                # Wenn execute_python_script eine Exception wirft, fange sie ab
                error_msg = str(exec_error)
                self.log('error', f'Code-Ausführung fehlgeschlagen: {error_msg}')
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': error_msg,
                    'scriptPath': script_path,
                    'scriptDir': script_dir,
                    'returncode': 1
                }
            
            stdout = execution_result.get('stdout', '')
            stderr = execution_result.get('stderr', '')
            success = execution_result.get('success', True)
            returncode = execution_result.get('returncode', 0)
            
            # Prüfe auf echte Fehler
            has_error = not success or (returncode != 0) or bool(stderr and stderr.strip())
            
            if has_error:
                error_msg = stderr[:1000] if stderr else stdout[:1000] if stdout else "Unbekannter Fehler"
                self.log('error', f'Code-Ausführung mit Fehlern: {error_msg}')
            else:
                self.log('success', 'Code erfolgreich ausgeführt')
            
            result = {
                'success': not has_error,
                'stdout': stdout,
                'stderr': stderr,
                'scriptPath': script_path,
                'scriptDir': script_dir,
                'returncode': returncode
            }
            
            return result
            
        except Exception as error:
            error_msg = str(error)
            self.log('error', f'Fehler bei Code-Ausführung: {error_msg}')
            # Re-raise mit besserer Fehlermeldung
            raise Exception(f'Code-Ausführung fehlgeschlagen: {error_msg}') from error
        finally:
            # Optional: Aufräumen (kann auch für Debugging behalten werden)
            # os.remove(script_path)
            # os.rmdir(script_dir)
            pass

