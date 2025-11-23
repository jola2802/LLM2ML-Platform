"""
LLM Queue mit asyncio für parallele Verarbeitung
"""

import asyncio
import os
from typing import Optional, Dict, Any
from datetime import datetime
from core.llm.llm import call_llm_api_async

# Konfiguration
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
    from config.worker_scaling_config import get_worker_config
    config = get_worker_config().get('llm', {})
    MAX_CONCURRENCY = config.get('maxConcurrency', 3)
    MAX_WORKERS = config.get('maxWorkers', 3)
    MIN_WORKERS = config.get('minWorkers', 1)
except Exception:
    MAX_CONCURRENCY = 3
    MAX_WORKERS = 3
    MIN_WORKERS = 1

class LLMQueue:
    def __init__(self, max_concurrency: int = MAX_CONCURRENCY, max_workers: int = MAX_WORKERS):
        self.max_concurrency = max_concurrency
        self.max_workers = max_workers
        self.min_workers = MIN_WORKERS
        self.queue = asyncio.Queue()
        self.processing = {}  # request_id -> task
        self.results = {}  # request_id -> result
        self.request_counter = 0
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.workers = []
        self.running = True
        self.workers_started = False
    
    def _ensure_workers_started(self):
        """Stelle sicher, dass Worker-Tasks gestartet sind"""
        if not self.workers_started:
            try:
                # Prüfe ob Event-Loop läuft
                loop = asyncio.get_running_loop()
                # Starte Worker-Tasks
                for i in range(self.min_workers):
                    task = asyncio.create_task(self._worker_task(f'worker-{i}'))
                    self.workers.append(task)
                self.workers_started = True
            except RuntimeError as e:
                # Keine laufende Event-Loop - wird beim nächsten Versuch erneut versucht
                print(f'[LLM Queue] Warnung: Worker konnten nicht gestartet werden (keine Event-Loop): {e}')
                # Setze workers_started NICHT auf True, damit es beim nächsten Mal erneut versucht wird
    
    async def _worker_task(self, worker_id: str):
        """Worker-Task für Verarbeitung von Queue-Items"""
        print(f'[LLM Queue] Worker {worker_id} gestartet')
        while self.running:
            try:
                # Warte auf Item aus Queue (mit Timeout)
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    # Worker wurde abgebrochen
                    print(f'[LLM Queue] Worker {worker_id} wurde abgebrochen')
                    break
                
                request_id, prompt, file_path, custom_model, max_retries, future = item

                # Prüfe ob Future bereits abgebrochen wurde
                if future.cancelled():
                    print(f'[LLM Queue] Request {request_id} wurde bereits abgebrochen, überspringe')
                    self.queue.task_done()
                    continue
                
                # Verarbeite Request
                async with self.semaphore:
                    self.processing[request_id] = worker_id
                    try:
                        print(f'[LLM Queue] Request {request_id}: Rufe Ollama API auf...')
                        result = await call_llm_api_async(prompt, file_path, custom_model, max_retries)
                        self.results[request_id] = result
                        if not future.done() and not future.cancelled():
                            future.set_result(result)
                        else:
                            print(f'[LLM Queue] Request {request_id}: Future bereits erledigt/abgebrochen')
                    except asyncio.CancelledError:
                        # Request wurde während Verarbeitung abgebrochen
                        print(f'[LLM Queue] Request {request_id} wurde während Verarbeitung abgebrochen')
                        if not future.done():
                            future.cancel()
                    except Exception as error:
                        print(f'[LLM Queue] Request {request_id} Fehler: {error}')
                        if not future.done() and not future.cancelled():
                            future.set_exception(error)
                        else:
                            print(f'[LLM Queue] Request {request_id}: Future bereits erledigt, kann Exception nicht setzen')
                    finally:
                        self.processing.pop(request_id, None)
                        self.queue.task_done()
                        
            except asyncio.CancelledError:
                # Worker-Task wurde abgebrochen
                print(f'[LLM Queue] Worker {worker_id} Task wurde abgebrochen')
                break
            except Exception as error:
                print(f'[LLM Queue] Worker {worker_id} unerwarteter Fehler: {error}')
                import traceback
                traceback.print_exc()
                # Bei unerwarteten Fehlern, warte kurz bevor weiter gemacht wird
                await asyncio.sleep(0.1)
    
    async def add_request(
        self,
        prompt: str,
        file_path: Optional[str] = None,
        custom_model: Optional[str] = None,
        max_retries: int = 3,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Request zur Queue hinzufügen"""
        # Stelle sicher, dass Worker gestartet sind
        self._ensure_workers_started()
        
        # Prüfe ob Worker tatsächlich gestartet wurden
        if not self.workers_started or len(self.workers) == 0:
            raise Exception('LLM Queue Worker konnten nicht gestartet werden - keine Event-Loop verfügbar')
        
        # Standard-Timeout: 1 Minuten (60 Sekunden)
        if timeout is None:
            timeout = 60.0
        
        self.request_counter += 1
        request_id = self.request_counter
        
        print(f'[LLM Queue] Request {request_id} zur Queue hinzugefügt (Model: {custom_model or "default"})')
        
        # Erstelle Future für asynchrones Ergebnis
        future = asyncio.Future()
        
        # Füge Request zur Queue hinzu
        try:
            # queue.put() ist eine coroutine und sollte mit await aufgerufen werden
            await self.queue.put((request_id, prompt, file_path, custom_model, max_retries, future))
            print(f'[LLM Queue] Request {request_id} erfolgreich in Queue eingefügt (Queue-Größe: {self.queue.qsize()})')
        except Exception as error:
            # Wenn Queue-Fehler auftritt, bereinige Future
            print(f'[LLM Queue] Fehler beim Hinzufügen von Request {request_id} zur Queue: {error}')
            if not future.done():
                future.cancel()
            raise Exception(f'Fehler beim Hinzufügen zur Queue: {error}')
        
        # Warte auf Ergebnis mit Timeout
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            # Timeout: Bereinige Future und Request
            print(f'[LLM Queue] Request {request_id} Timeout nach {timeout} Sekunden')
            if not future.done():
                future.cancel()
            self.processing.pop(request_id, None)
            raise Exception(f'LLM Request Timeout nach {timeout} Sekunden')
        except asyncio.CancelledError:
            # Task wurde abgebrochen: Bereinige Future und Request
            print(f'[LLM Queue] Request {request_id} wurde abgebrochen')
            if not future.done():
                future.cancel()
            self.processing.pop(request_id, None)
            raise Exception('LLM Request wurde abgebrochen')
        except Exception as error:
            # Andere Fehler: Bereinige Future
            print(f'[LLM Queue] Request {request_id} Fehler beim Warten auf Ergebnis: {error}')
            if not future.done():
                future.cancel()
            self.processing.pop(request_id, None)
            raise Exception(f'LLM Request fehlgeschlagen: {error}')
    
    def cancel_request(self, request_id: int, reason: str = 'User cancelled') -> bool:
        """Request abbrechen"""
        if request_id in self.processing:
            # Request ist bereits in Verarbeitung - kann nicht abgebrochen werden
            return False
        # Request ist noch in Queue - könnte theoretisch entfernt werden
        # Für jetzt: nur bereits verarbeitete Requests können nicht abgebrochen werden
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Queue Status abrufen"""
        return {
            'queueSize': self.queue.qsize(),
            'processing': len(self.processing),
            'workers': len(self.workers),
            'maxWorkers': self.max_workers,
            'maxConcurrency': self.max_concurrency
        }
    
    async def shutdown(self):
        """Queue herunterfahren"""
        self.running = False
        
        # Bereinige alle wartenden Futures
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                if item and len(item) > 5:
                    future = item[5]
                    if not future.done():
                        future.cancel()
                self.queue.task_done()
            except Exception:
                pass
        
        # Warte auf alle Worker (mit Timeout)
        for worker in self.workers:
            if not worker.done():
                worker.cancel()
                try:
                    await asyncio.wait_for(worker, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
        
        # Bereinige alle verarbeitenden Requests
        for request_id in list(self.processing.keys()):
            self.processing.pop(request_id, None)

# Globale Queue-Instanz (wird bei Bedarf initialisiert)
llm_queue = None

def get_queue():
    """Hole oder erstelle Queue-Instanz"""
    global llm_queue
    if llm_queue is None:
        try:
            llm_queue = LLMQueue()
            print('[LLM Queue] Queue-Instanz erstellt')
        except Exception as e:
            print(f'[LLM Queue] Fehler beim Erstellen der Queue-Instanz: {e}')
            return None
    return llm_queue

