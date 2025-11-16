"""
Webhook-Client für API Gateway
"""

import os
import requests
import time
from typing import Dict, Any, Optional

API_GATEWAY_URL = os.getenv('API_GATEWAY_URL', 'http://localhost:3001')

def send_job_completion_webhook(
    job_id: str,
    job_type: str,
    project_id: str,
    result: Optional[Dict[str, Any]],
    status: str
) -> Optional[Dict[str, Any]]:
    """Sendet Webhook an API Gateway wenn Job abgeschlossen ist"""
    try:
        webhook_data = {
            'jobId': job_id,
            'jobType': job_type,
            'projectId': project_id,
            'result': result,
            'status': status
        }
        
        print(f'Sende Webhook für Job {job_id} ({job_type}) an API Gateway...')
        
        response = requests.post(
            f'{API_GATEWAY_URL}/api/webhooks/job-completed',
            json=webhook_data,
            timeout=10
        )
        response.raise_for_status()
        
        print(f'Webhook für Job {job_id} erfolgreich gesendet')
        return response.json()
    except Exception as error:
        print(f'Fehler beim Senden des Webhooks für Job {job_id}: {error}')
        print(f'Webhook-URL: {API_GATEWAY_URL}/api/webhooks/job-completed')
        
        # Versuche erneut bei Verbindungsfehlern
        if hasattr(error, 'errno') and error.errno in ['ECONNREFUSED', 'ETIMEDOUT']:
            print('API Gateway nicht erreichbar, versuche Webhook erneut in 5 Sekunden...')
            time.sleep(5)
            try:
                response = requests.post(
                    f'{API_GATEWAY_URL}/api/webhooks/job-completed',
                    json=webhook_data,
                    timeout=10
                )
                response.raise_for_status()
                print(f'Webhook für Job {job_id} erfolgreich nach Wiederholung gesendet')
                return response.json()
            except Exception as retry_error:
                print(f'Webhook-Wiederholung für Job {job_id} fehlgeschlagen: {retry_error}')
        
        return None

