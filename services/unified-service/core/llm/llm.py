"""
LLM-API Implementation für Ollama
"""

import os
import json
import asyncio
from typing import Optional, Dict, Any, List
import ollama
from datetime import datetime

# LLM Provider Enum
LLM_PROVIDERS = {
    'OLLAMA': 'ollama'
}

# LLM Konfiguration
llm_config = {
    'activeProvider': LLM_PROVIDERS['OLLAMA'],
    'ollama': {
        'host': os.getenv('OLLAMA_URL', 'http://192.168.0.206:11434'),
        'defaultModel': 'llama3.2:latest',
        'availableModels': []
    }
}

# File-Cache für bereits hochgeladene Dateien
file_cache = {}

# ===== KONFIGURATION FUNKTIONEN =====

def get_llm_config() -> Dict[str, Any]:
    """Aktuelle LLM-Konfiguration abrufen"""
    return llm_config.copy()

def update_ollama_config(config: Dict[str, Any]):
    """Ollama-Konfiguration aktualisieren"""
    llm_config['ollama'].update(config)
    print(f'Ollama configuration updated: {llm_config["ollama"]}')

# ===== OLLAMA FUNKTIONEN =====

def get_available_ollama_models() -> Dict[str, Any]:
    """Verfügbare Ollama-Modelle abrufen"""
    try:
        client = ollama.Client(host=llm_config['ollama']['host'])
        response = client.list()
        
        if response and 'models' in response and isinstance(response['models'], list):
            models = []
            for model in response['models']:
                models.append({
                    'name': model.get('name', ''),
                    'size': model.get('size', 0),
                    'modified_at': model.get('modified_at', datetime.now().isoformat()),
                    'digest': model.get('digest', '')
                })
            
            # Update lokale Konfiguration
            llm_config['ollama']['availableModels'] = [m['name'] for m in models]
            
            return {
                'success': True,
                'models': models,
                'defaultModel': llm_config['ollama']['defaultModel'],
                'availableModels': llm_config['ollama']['availableModels']
            }
        else:
            print('Keine Modelle in der Ollama-Antwort gefunden')
            return {
                'success': False,
                'error': 'Keine Modelle gefunden',
                'models': [],
                'availableModels': []
            }
    except Exception as error:
        print(f'Fehler beim Abrufen der Ollama-Modelle: {error}')
        return {
            'success': False,
            'error': str(error),
            'models': [],
            'availableModels': []
        }

def test_ollama_connection() -> Dict[str, Any]:
    """Ollama-Verbindung testen"""
    try:
        client = ollama.Client(host=llm_config['ollama']['host'])
        response = client.chat(
            model=llm_config['ollama']['defaultModel'],
            messages=[{'role': 'user', 'content': 'Antworte nur mit "OK" wenn du diese Nachricht erhältst.'}]
        )
        
        content = response.get('message', {}).get('content', '') or response.get('content', '')
        is_connected = 'ok' in content.lower()
        
        return {
            'success': True,
            'connected': is_connected,
            'model': llm_config['ollama']['defaultModel'],
            'response': content
        }
    except Exception as error:
        print(f'Ollama-Verbindungstest fehlgeschlagen: {error}')
        return {
            'success': False,
            'connected': False,
            'error': str(error)
        }

# ===== EINHEITLICHE LLM API =====

async def call_llm_api_async(
    prompt: str,
    file_path: Optional[str] = None,
    custom_model: Optional[str] = None,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Asynchrone LLM-API-Call-Funktion
    """
    model = custom_model or llm_config['ollama']['defaultModel']
    provider = LLM_PROVIDERS['OLLAMA']
    
    attempt = 0
    
    while attempt < max_retries:
        try:
            attempt += 1
            print(f'LLM API Call - Versuch {attempt}/{max_retries} mit {provider}:{model}')
            
            # Datei-Inhalte vorbereiten (falls vorhanden)
            if file_path and file_path in file_cache:
                print(f'Verwende gecachte Datei: {file_path}')
            
            # Ollama API-Call
            client = ollama.Client(host=llm_config['ollama']['host'])
            response = client.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            result_text = response.get('message', {}).get('content', '') or response.get('content', '')
            
            # Validiere Response
            if not result_text:
                raise ValueError('Leere Response vom LLM erhalten')
            
            return {
                'result': result_text,
                'file_uploaded': bool(file_path),
                'provider': provider,
                'model': model
            }
            
        except Exception as error:
            print(f'LLM API Fehler (Versuch {attempt}): {error}')
            
            # Bei letzten Versuch, Fehler werfen
            if attempt >= max_retries:
                raise Exception(f'LLM API fehlgeschlagen nach {max_retries} Versuchen: {error}')
            
            # Kurze Pause vor nächstem Versuch
            await asyncio.sleep(1 * attempt)

def call_llm_api(
    prompt: str,
    file_path: Optional[str] = None,
    custom_model: Optional[str] = None,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Synchrone Wrapper-Funktion für LLM-API-Calls
    """
    try:
        # Versuche Queue zu verwenden
        try:
            from core.llm.llm_queue import get_queue
            queue = get_queue()
            return asyncio.run(queue.add_request(prompt, file_path, custom_model, max_retries))
        except (ImportError, AttributeError, Exception) as e:
            # Fallback auf direkte API-Calls wenn Queue nicht verfügbar
            print(f'Queue nicht verfügbar, verwende direkte API: {e}')
            return asyncio.run(call_llm_api_async(prompt, file_path, custom_model, max_retries))
    except Exception as error:
        print(f'LLM Queue Fehler: {error}')
        # Fallback auf direkte API-Calls
        print('Fallback auf direkte LLM API...')
        return asyncio.run(call_llm_api_async(prompt, file_path, custom_model, max_retries))

# ===== QUEUE MANAGEMENT =====

def get_llm_queue_status() -> Dict[str, Any]:
    """Queue Status abrufen"""
    try:
        from llm.api.llm_queue import get_queue
        queue = get_queue()
        return queue.get_status()
    except Exception:
        return {
            'queueSize': 0,
            'processing': 0,
            'workers': 0,
            'maxWorkers': 3
        }

def cancel_llm_request(request_id: int, reason: str = 'User cancelled') -> bool:
    """Queue Request abbrechen"""
    try:
        from llm.api.llm_queue import get_queue
        queue = get_queue()
        return queue.cancel_request(request_id, reason)
    except Exception:
        return False

# ===== INITIALISIERUNG =====

def initialize_ollama_models():
    """Initialisiere Ollama-Modelle beim Start"""
    try:
        result = get_available_ollama_models()
        if result.get('success'):
            print(f"Ollama-Modelle geladen: {len(result.get('models', []))} Modelle verfügbar")
        else:
            print('Keine Ollama-Modelle gefunden oder Ollama nicht verfügbar')
    except Exception as error:
        print(f'Fehler beim Initialisieren der Ollama-Modelle: {error}')

