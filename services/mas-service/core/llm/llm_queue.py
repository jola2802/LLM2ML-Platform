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
        
        # Starte Worker-Tasks
        self._start_workers()
    
    def _start_workers(self):
        """Starte Worker-Tasks"""
        for i in range(self.min_workers):
            task = asyncio.create_task(self._worker_task(f'worker-{i}'))
            self.workers.append(task)
    
    async def _worker_task(self, worker_id: str):
        """Worker-Task für Verarbeitung von Queue-Items"""
        while self.running:
            try:
                # Warte auf Item aus Queue (mit Timeout)
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                request_id, prompt, file_path, custom_model, max_retries, future = item
                
                # Verarbeite Request
                async with self.semaphore:
                    self.processing[request_id] = worker_id
                    try:
                        result = await call_llm_api_async(prompt, file_path, custom_model, max_retries)
                        self.results[request_id] = result
                        if not future.done():
                            future.set_result(result)
                    except Exception as error:
                        if not future.done():
                            future.set_exception(error)
                    finally:
                        self.processing.pop(request_id, None)
                        self.queue.task_done()
                        
            except Exception as error:
                print(f'Worker {worker_id} Fehler: {error}')
    
    async def add_request(
        self,
        prompt: str,
        file_path: Optional[str] = None,
        custom_model: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Request zur Queue hinzufügen"""
        self.request_counter += 1
        request_id = self.request_counter
        
        # Erstelle Future für asynchrones Ergebnis
        future = asyncio.Future()
        
        # Füge Request zur Queue hinzu
        await self.queue.put((request_id, prompt, file_path, custom_model, max_retries, future))
        
        # Warte auf Ergebnis
        try:
            result = await future
            return result
        except Exception as error:
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
    
    def shutdown(self):
        """Queue herunterfahren"""
        self.running = False
        # Warte auf alle Worker
        for worker in self.workers:
            if not worker.done():
                worker.cancel()

# Globale Queue-Instanz (wird bei Bedarf initialisiert)
llm_queue = None

def get_queue():
    """Hole oder erstelle Queue-Instanz"""
    global llm_queue
    if llm_queue is None:
        llm_queue = LLMQueue()
    return llm_queue

