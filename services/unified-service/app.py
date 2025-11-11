"""
Unified ML Service - Kombiniert MAS-Service und Python-Service
Flask-basierter Service für LLM-API, Agent-Pipeline, Datenanalyse und Code-Ausführung
"""

import os
import signal
import sys
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime

# Importiere Routen
from api.routes import llm_routes, agent_routes, queue_routes, data_routes, execution_routes
from core.llm.llm import initialize_ollama_models
from infrastructure.workers.worker_pool import PythonWorkerPool
from infrastructure.queue.job_queue import job_queue
from infrastructure.clients.webhook_client import send_job_completion_webhook

app = Flask(__name__)
PORT = int(os.getenv('PORT', 3002))

# CORS konfigurieren
CORS(app, origins=[os.getenv('API_GATEWAY_URL', 'http://localhost:3001')], supports_credentials=True)

# JSON-Limit erhöhen
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Verzeichnisse
SCRIPT_DIR = os.getenv('SCRIPT_DIR', '/app/scripts')
VENV_DIR = os.getenv('VENV_DIR', '/opt/venv')
UPLOADS_DIR = os.getenv('UPLOADS_DIR', '/app/uploads')
MODELS_DIR = os.getenv('MODELS_DIR', '/app/models')

# Python Worker Pool initialisieren
print(f'Initialisiere Python Worker Pool...')
print(f'Script-Dir: {SCRIPT_DIR}')
print(f'Venv-Dir: {VENV_DIR}')
python_worker_pool = PythonWorkerPool(SCRIPT_DIR, VENV_DIR, 5)
print(f'Python Worker Pool initialisiert')

# Job-Completion-Handler - Sendet Webhook an API Gateway
def on_job_completed(job):
    """Handler für abgeschlossene Jobs"""
    print(f'Job {job["id"]} abgeschlossen: {job["status"]}')
    
    project_id = job.get('data', {}).get('projectId')
    if project_id:
        try:
            send_job_completion_webhook(
                job['id'],
                job['type'],
                project_id,
                job.get('result'),
                job['status']
            )
        except Exception as error:
            print(f'Fehler beim Senden des Webhooks für Job {job["id"]}: {error}')

job_queue.on('jobCompleted', on_job_completed)

# Health-Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    pool_status = python_worker_pool.get_pool_status()
    return jsonify({
        'status': 'healthy',
        'service': 'unified-service',
        'port': PORT,
        'pool': {
            'totalWorkers': pool_status['totalWorkers'],
            'availableWorkers': pool_status['availableWorkers'],
            'busyWorkers': pool_status['busyWorkers']
        },
        'timestamp': datetime.now().isoformat()
    })

# API-Routen registrieren
llm_routes.setup_llm_routes(app)
agent_routes.setup_agent_routes(app, python_worker_pool)
queue_routes.setup_queue_routes(app)
data_routes.setup_data_routes(app, VENV_DIR, UPLOADS_DIR)
execution_routes.setup_execution_routes(app, python_worker_pool, SCRIPT_DIR, VENV_DIR, MODELS_DIR)

# Fehlerbehandlung
@app.errorhandler(Exception)
def handle_error(error):
    print(f'Fehler: {error}', file=sys.stderr)
    return jsonify({
        'error': str(error) if hasattr(error, '__str__') else 'Interner Serverfehler',
        'service': 'unified-service'
    }), 500

# Graceful Shutdown Handler
def signal_handler(sig, frame):
    print('\nSIGTERM/SIGINT empfangen, fahre Service herunter...')
    python_worker_pool.shutdown()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    # Initialisiere Ollama-Modelle beim Start
    initialize_ollama_models()
    
    print(f'Unified ML Service läuft auf Port {PORT}')
    print(f'OLLAMA_URL: {os.getenv("OLLAMA_URL", "http://192.168.0.206:11434")}')
    print(f'Script-Dir: {SCRIPT_DIR}')
    print(f'Venv-Dir: {VENV_DIR}')
    print(f'Uploads-Dir: {UPLOADS_DIR}')
    
    app.run(host='0.0.0.0', port=PORT, debug=False)

