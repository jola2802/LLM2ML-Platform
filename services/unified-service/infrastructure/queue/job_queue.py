"""
Job Queue für asynchrone Job-Verwaltung
"""

import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from threading import Lock
import threading

# Job-Status Konstanten
JOB_STATUS = {
    'PENDING': 'pending',
    'RUNNING': 'running',
    'COMPLETED': 'completed',
    'FAILED': 'failed',
    'CANCELLED': 'cancelled'
}

# Job-Typen
JOB_TYPES = {
    'TRAINING': 'training',
    'RETRAINING': 'retraining',
    'PREDICTION': 'prediction'
}

class JobQueue:
    def __init__(self):
        self.jobs = {}  # job_id -> job
        self.queue = []  # Liste von job_ids
        self.running_jobs = set()  # Set von aktuell laufenden job_ids
        self.max_concurrent_jobs = 5
        self.lock = Lock()
        self.listeners = {}  # event -> [callbacks]
        
        self.stats = {
            'totalJobs': 0,
            'completedJobs': 0,
            'failedJobs': 0,
            'currentlyRunning': 0
        }
        
        # Starte Cleanup-Thread
        self._start_cleanup_thread()
    
    def on(self, event: str, callback):
        """Event-Listener registrieren"""
        if event not in self.listeners:
            self.listeners[event] = []
        self.listeners[event].append(callback)
    
    def emit(self, event: str, *args, **kwargs):
        """Event emittieren"""
        print(f'[Job Queue] Emittiere Event: {event}')
        print(f'[Job Queue] Registrierte Listener für {event}: {len(self.listeners.get(event, []))}')
        if event in self.listeners:
            for i, callback in enumerate(self.listeners[event]):
                try:
                    print(f'[Job Queue] Rufe Listener {i+1}/{len(self.listeners[event])} für Event {event} auf')
                    callback(*args, **kwargs)
                    print(f'[Job Queue] Listener {i+1} erfolgreich aufgerufen')
                except Exception as error:
                    print(f'[Job Queue] Fehler in Event-Listener {i+1} für {event}: {error}')
                    import traceback
                    print(f'[Job Queue] Traceback: {traceback.format_exc()}')
        else:
            print(f'[Job Queue] WARNUNG: Keine Listener für Event {event} registriert')
    
    def add_job(self, job_data: Dict[str, Any]) -> str:
        """Job zur Queue hinzufügen"""
        job_id = str(uuid.uuid4())
        job = {
            'id': job_id,
            'type': job_data['type'],
            'status': JOB_STATUS['PENDING'],
            'data': job_data.get('data', {}),
            'priority': job_data.get('priority', 0),
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'error': None,
            'result': None
        }
        
        with self.lock:
            self.jobs[job_id] = job
            self.queue.append(job_id)
            self.stats['totalJobs'] += 1
        
        # Queue nach Priorität sortieren
        self.queue.sort(key=lambda jid: self.jobs[jid]['priority'], reverse=True)
        
        print(f'Job {job_id} ({job["type"]}) zur Queue hinzugefügt. Queue-Länge: {len(self.queue)}')
        
        self.emit('jobAdded', job)
        self.process_queue()
        
        return job_id
    
    def process_queue(self):
        """Queue verarbeiten"""
        with self.lock:
            if len(self.running_jobs) >= self.max_concurrent_jobs:
                print(f'Queue-Verarbeitung übersprungen: {len(self.running_jobs)} Jobs laufen bereits (Max: {self.max_concurrent_jobs})')
                return
            
            if not self.queue:
                print('Queue-Verarbeitung: Keine Jobs in der Queue')
                return
            
            job_id = self.queue.pop(0)
            job = self.jobs.get(job_id)
            
            if not job:
                print(f'Queue-Verarbeitung: Job {job_id} nicht gefunden')
                self.process_queue()
                return
            
            if job['status'] != JOB_STATUS['PENDING']:
                print(f'Queue-Verarbeitung: Job {job_id} hat Status {job["status"]}, nicht PENDING')
                self.process_queue()
                return
            
            print(f'[Job Queue] ===== QUEUE-VERARBEITUNG =====')
            print(f'[Job Queue] Starte Job {job_id} ({job["type"]})')
            print(f'[Job Queue] Job-Daten: {list(job.get("data", {}).keys())}')
        
        # Außerhalb des Locks, um Deadlocks zu vermeiden
        self.start_job(job_id)
        
        # Rekursiv weitere Jobs starten
        with self.lock:
            if len(self.running_jobs) < self.max_concurrent_jobs and self.queue:
                self.process_queue()
    
    def start_job(self, job_id: str):
        """Job starten"""
        job = self.jobs.get(job_id)
        if not job:
            print(f'[Job Queue] ❌ FEHLER: Job {job_id} nicht gefunden beim Starten')
            return
        
        with self.lock:
            job['status'] = JOB_STATUS['RUNNING']
            job['started_at'] = datetime.now().isoformat()
            self.running_jobs.add(job_id)
            self.stats['currentlyRunning'] += 1
        
        print(f'[Job Queue] ✅ Job {job_id} ({job["type"]}) gestartet. Laufende Jobs: {len(self.running_jobs)}')
        print(f'[Job Queue] Job-Daten: projectId={job.get("data", {}).get("projectId")}, hasPythonCode={bool(job.get("data", {}).get("pythonCode"))}')
        print(f'[Job Queue] Emittiere Events: jobStarted und executeJob für Job {job_id}')
        print(f'[Job Queue] Registrierte Listener für executeJob: {len(self.listeners.get("executeJob", []))}')
        
        if 'executeJob' not in self.listeners or len(self.listeners.get('executeJob', [])) == 0:
            print(f'[Job Queue] ⚠️ WARNUNG: Keine Listener für executeJob registriert!')
            print(f'[Job Queue] Verfügbare Events: {list(self.listeners.keys())}')
        
        self.emit('jobStarted', job)
        self.emit('executeJob', job)
        
        print(f'[Job Queue] Events emittiert für Job {job_id}')
    
    def complete_job(self, job_id: str, result: Optional[Dict[str, Any]] = None):
        """Job abschließen"""
        if result is None:
            result = {}
        
        job = self.jobs.get(job_id)
        if not job:
            return
        
        with self.lock:
            job['completed_at'] = datetime.now().isoformat()
            # result kann entweder {'result': ...} oder direkt {'output': ..., 'metrics': ...} sein
            if 'result' in result:
                job['result'] = result.get('result')
            else:
                # Direktes Result-Dict (output, stderr, metrics)
                job['result'] = result
            job['error'] = result.get('error')
            job['status'] = JOB_STATUS['FAILED'] if result.get('error') else JOB_STATUS['COMPLETED']
            
            self.running_jobs.discard(job_id)
            self.stats['currentlyRunning'] -= 1
            
            if result.get('error'):
                self.stats['failedJobs'] += 1
                print(f'Job {job_id} ({job["type"]}) fehlgeschlagen: {result.get("error")}')
            else:
                self.stats['completedJobs'] += 1
                print(f'Job {job_id} ({job["type"]}) erfolgreich abgeschlossen')
        
        self.emit('jobCompleted', job)
        self.process_queue()
    
    def cancel_job(self, job_id: str) -> bool:
        """Job abbrechen"""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        with self.lock:
            if job['status'] == JOB_STATUS['PENDING']:
                if job_id in self.queue:
                    self.queue.remove(job_id)
                job['status'] = JOB_STATUS['CANCELLED']
                job['completed_at'] = datetime.now().isoformat()
                print(f'Job {job_id} ({job["type"]}) abgebrochen (war in Queue)')
                self.emit('jobCancelled', job)
                return True
            elif job['status'] == JOB_STATUS['RUNNING']:
                job['status'] = JOB_STATUS['CANCELLED']
                print(f'Job {job_id} ({job["type"]}) zum Abbruch markiert (läuft gerade)')
                self.emit('jobCancelled', job)
                return True
        
        return False
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Job-Status abrufen"""
        return self.jobs.get(job_id)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Queue-Status abrufen"""
        with self.lock:
            return {
                'queueLength': len(self.queue),
                'runningJobs': len(self.running_jobs),
                'maxConcurrentJobs': self.max_concurrent_jobs,
                'stats': self.stats.copy()
            }
    
    def cleanup_old_jobs(self):
        """Alte Jobs aufräumen"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        cleaned = 0
        
        with self.lock:
            to_remove = []
            for job_id, job in self.jobs.items():
                if job['status'] in [JOB_STATUS['COMPLETED'], JOB_STATUS['FAILED']]:
                    created_at = datetime.fromisoformat(job['created_at'])
                    if created_at < cutoff_time:
                        to_remove.append(job_id)
            
            for job_id in to_remove:
                del self.jobs[job_id]
                cleaned += 1
        
        if cleaned > 0:
            print(f'{cleaned} alte Jobs aufgeräumt')
        
        return cleaned
    
    def _start_cleanup_thread(self):
        """Starte Cleanup-Thread"""
        def cleanup_loop():
            while True:
                time.sleep(6 * 60 * 60)  # 6 Stunden
                self.cleanup_old_jobs()
        
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()

# Singleton-Instanz
job_queue = JobQueue()

