"""
Python Worker Pool für parallele Code-Ausführung
"""

import os
import threading
import time
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, Future
from infrastructure.queue.job_queue import job_queue, JOB_TYPES, JOB_STATUS
from core.execution.code_executor import execute_python_script
from core.execution.metrics_extractor import extract_metrics_from_output

class PythonWorkerPool:
    def __init__(self, script_dir: str, venv_dir: str, max_workers: int = 5):
        self.script_dir = script_dir
        self.venv_dir = venv_dir
        self.max_workers = max_workers
        self.min_workers = 1
        
        # Thread Pool Executor
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_jobs = {}  # job_id -> Future
        
        # Statistiken
        self.stats = {
            'totalJobsProcessed': 0,
            'totalErrors': 0,
            'averageExecutionTime': 0,
            'queueLength': 0,
            'activeWorkers': 0
        }
        
        # Event-Listener
        self.listeners = {}
        
        # Setup Job Queue Listeners
        self._setup_queue_listeners()
    
    def on(self, event: str, callback):
        """Event-Listener registrieren"""
        if event not in self.listeners:
            self.listeners[event] = []
        self.listeners[event].append(callback)
    
    def emit(self, event: str, *args, **kwargs):
        """Event emittieren"""
        if event in self.listeners:
            for callback in self.listeners[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as error:
                    print(f'Fehler in Event-Listener: {error}')
    
    def _setup_queue_listeners(self):
        """Richte Job Queue Event-Listener ein"""
        print('Registriere executeJob Event-Listener im Worker Pool')
        job_queue.on('executeJob', self._execute_job)
        print(f'Event-Listener registriert. Anzahl Listener: {len(job_queue.listeners.get("executeJob", []))}')
    
    def _execute_job(self, job: Dict[str, Any]):
        """Führt einen Job aus"""
        job_id = job['id']
        job_type = job['type']
        
        print(f'[Worker Pool] Führe Job {job_id} ({job_type}) aus')
        print(f'[Worker Pool] Job-Daten: {list(job.get("data", {}).keys())}')
        
        # Submit Job an Thread Pool
        future = self.executor.submit(self._run_job, job)
        self.active_jobs[job_id] = future
        print(f'[Worker Pool] Job {job_id} an Thread Pool übergeben')
    
    def _run_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Führt einen Job in einem Worker-Thread aus"""
        job_id = job['id']
        job_type = job['type']
        job_data = job.get('data', {})
        
        print(f'[Worker Thread] Starte Ausführung von Job {job_id} ({job_type})')
        print(f'[Worker Thread] Job-Daten: projectId={job_data.get("projectId")}, hasPythonCode={bool(job_data.get("pythonCode"))}')
        
        start_time = time.time()
        
        try:
            if job_type == JOB_TYPES['TRAINING'] or job_type == JOB_TYPES['RETRAINING']:
                print(f'[Worker Thread] Führe Training-Job aus für Projekt {job_data.get("projectId")}')
                result = self._execute_training_job(job_data)
                print(f'[Worker Thread] Training-Job abgeschlossen für Projekt {job_data.get("projectId")}')
            else:
                result = {'error': f'Unbekannter Job-Typ: {job_type}'}
            
            execution_time = (time.time() - start_time) * 1000  # in ms
            
            # Statistiken aktualisieren
            self.stats['totalJobsProcessed'] += 1
            self._update_average_execution_time(execution_time)
            
            print(f'[Worker Thread] Job {job_id} erfolgreich abgeschlossen ({execution_time:.0f}ms)')
            
            # Job als abgeschlossen markieren
            # result enthält bereits output, stderr, metrics
            job_queue.complete_job(job_id, result)
            
            return result
            
        except Exception as error:
            execution_time = (time.time() - start_time) * 1000
            self.stats['totalErrors'] += 1
            
            print(f'[Worker Thread] Job {job_id} fehlgeschlagen: {error}')
            import traceback
            print(f'[Worker Thread] Traceback: {traceback.format_exc()}')
            
            job_queue.complete_job(job_id, {'error': str(error)})
            
            return {'error': str(error)}
        finally:
            # Job aus aktiven Jobs entfernen
            self.active_jobs.pop(job_id, None)
    
    def _execute_training_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Führt Training-Job aus"""
        project_id = job_data.get('projectId')
        python_code = job_data.get('pythonCode')
        
        print(f'[Training Job] ===== STARTE TRAINING-JOB =====')
        print(f'[Training Job] Projekt-ID: {project_id}')
        print(f'[Training Job] Python-Code Länge: {len(python_code) if python_code else 0} Zeichen')
        print(f'[Training Job] Script-Dir: {self.script_dir}')
        print(f'[Training Job] Script-Dir existiert: {os.path.exists(self.script_dir)}')
        
        if not project_id:
            raise ValueError('projectId ist erforderlich')
        
        if not python_code:
            raise ValueError('pythonCode ist erforderlich')
        
        # Stelle sicher, dass Script-Verzeichnis existiert
        if not os.path.exists(self.script_dir):
            print(f'[Training Job] Erstelle Script-Verzeichnis: {self.script_dir}')
            try:
                os.makedirs(self.script_dir, exist_ok=True)
                print(f'[Training Job] Script-Verzeichnis erfolgreich erstellt')
            except Exception as e:
                print(f'[Training Job] FEHLER beim Erstellen des Verzeichnisses: {e}')
                raise
        
        # Erstelle Script-Pfad
        script_path = os.path.join(self.script_dir, f'train_{project_id}.py')
        print(f'[Training Job] Script-Pfad: {script_path}')
        print(f'[Training Job] Absoluter Script-Pfad: {os.path.abspath(script_path)}')
        
        # Schreibe Code in Datei
        print(f'[Training Job] Schreibe Python-Code in Datei...')
        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(python_code)
            
            # Verifiziere, dass Datei geschrieben wurde
            if not os.path.exists(script_path):
                raise Exception(f'Script-Datei wurde nicht erstellt: {script_path}')
            
            file_size = os.path.getsize(script_path)
            print(f'[Training Job] ✅ Script erfolgreich erstellt: {script_path}')
            print(f'[Training Job] Script-Größe: {file_size} Bytes')
            
            # Prüfe ersten Teil des geschriebenen Codes
            with open(script_path, 'r', encoding='utf-8') as f:
                first_lines = ''.join(f.readlines()[:5])
                print(f'[Training Job] Erste Zeilen des Scripts: {first_lines[:200]}...')
                
        except Exception as e:
            print(f'[Training Job] ❌ FEHLER beim Schreiben des Scripts: {e}')
            import traceback
            print(f'[Training Job] Traceback: {traceback.format_exc()}')
            raise
        
        try:
            # Führe Script aus
            print(f'[Training Job] Starte Python-Script-Ausführung...')
            result = execute_python_script(script_path, self.script_dir, self.venv_dir)
            print(f'[Training Job] Script-Ausführung abgeschlossen')
            print(f'[Training Job] Output-Länge: {len(result.get("stdout", ""))} Zeichen')
            print(f'[Training Job] Stderr-Länge: {len(result.get("stderr", ""))} Zeichen')
            
            # Extrahiere Metriken
            output = result.get('stdout', '')
            metrics = extract_metrics_from_output(output, 'classification')  # TODO: model_type aus job_data
            print(f'[Training Job] Extrahierte Metriken: {list(metrics.keys())}')
            
            return {
                'output': output,
                'stderr': result.get('stderr', ''),
                'metrics': metrics
            }
        except Exception as error:
            print(f'[Training Job] FEHLER bei Training-Job: {error}')
            import traceback
            print(f'[Training Job] Traceback: {traceback.format_exc()}')
            raise
        finally:
            # Optional: Script löschen (oder für Debugging behalten)
            pass
    
    def _update_average_execution_time(self, execution_time: float):
        """Aktualisiert durchschnittliche Ausführungszeit"""
        total = self.stats['totalJobsProcessed']
        current_avg = self.stats['averageExecutionTime']
        
        if total == 1:
            self.stats['averageExecutionTime'] = execution_time
        else:
            # Gleitender Durchschnitt
            self.stats['averageExecutionTime'] = (current_avg * (total - 1) + execution_time) / total
    
    def add_training_job(self, project_id: str, python_code: str, priority: int = 0) -> str:
        """Fügt Training-Job zur Queue hinzu"""
        return job_queue.add_job({
            'type': JOB_TYPES['TRAINING'],
            'data': {
                'projectId': project_id,
                'pythonCode': python_code
            },
            'priority': priority
        })
    
    def add_retraining_job(self, project_id: str, python_code: str, priority: int = 0) -> str:
        """Fügt Retraining-Job zur Queue hinzu"""
        return job_queue.add_job({
            'type': JOB_TYPES['RETRAINING'],
            'data': {
                'projectId': project_id,
                'pythonCode': python_code
            },
            'priority': priority
        })
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Gibt Pool-Status zurück"""
        queue_status = job_queue.get_queue_status()
        
        return {
            'totalWorkers': self.max_workers,
            'availableWorkers': self.max_workers - len(self.active_jobs),
            'busyWorkers': len(self.active_jobs),
            'queueLength': queue_status['queueLength'],
            'stats': self.stats.copy()
        }
    
    def shutdown(self):
        """Fährt Worker Pool herunter"""
        print('Fahre Worker Pool herunter...')
        self.executor.shutdown(wait=True)
        print('Worker Pool heruntergefahren')

