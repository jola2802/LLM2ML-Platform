"""
Basis-Worker-Klasse für alle spezialisierten Worker-Agents
"""

from typing import Any, Dict, Optional
from core.llm import llm
from core.agents.config_agent_network import get_agent_config, get_agent_model, log_agent_call

class BaseWorker:
    def __init__(self, agent_key: str):
        self.agent_key = agent_key
        self.config = get_agent_config(agent_key)
        self.model = get_agent_model(agent_key)
    
    async def execute(self, pipeline_state: Dict[str, Any]) -> Any:
        """Führt den Worker-Agent aus - muss überschrieben werden"""
        raise NotImplementedError(f'execute() Methode muss in {self.agent_key} implementiert werden')
    
    async def test(self) -> bool:
        """Testet die Verbindung des Worker-Agents"""
        from llm.agents.prompts import BASE_WORKER_TEST_PROMPT, format_prompt
        
        test_prompt = format_prompt(BASE_WORKER_TEST_PROMPT, {
            'agentName': self.config['name']
        })
        
        try:
            response = llm.call_llm_api(test_prompt, None, self.model, 1)
            result = response.get('result', '') if isinstance(response, dict) else str(response)
            return 'ok' in result.lower()
        except Exception as error:
            print(f'Test für {self.agent_key} fehlgeschlagen: {error}')
            return False
    
    async def call_llm(self, prompt: str, context: Optional[str] = None, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Ruft die LLM-API auf mit Agent-spezifischen Einstellungen"""
        log_agent_call(self.agent_key, self.model, 'LLM-Aufruf')
        
        tokens = max_tokens or self.config.get('maxTokens', 2048)
        
        try:
            response = llm.call_llm_api(prompt, context, self.model, tokens)
            return response
        except Exception as error:
            print(f'LLM-Aufruf für {self.agent_key} fehlgeschlagen: {error}')
            raise
    
    def validate_result(self, result: Any, expected_type: str = 'string') -> bool:
        """Validiert das Ergebnis eines Worker-Agents"""
        if not result:
            raise ValueError(f'{self.agent_key}: Kein Ergebnis erhalten')
        
        if expected_type == 'string' and not isinstance(result, str):
            raise TypeError(f'{self.agent_key}: Erwarteter String, erhalten: {type(result)}')
        
        if expected_type == 'object' and not isinstance(result, dict):
            raise TypeError(f'{self.agent_key}: Erwartetes Objekt, erhalten: {type(result)}')
        
        return True
    
    def clean_code(self, code: str) -> str:
        """Bereinigt generierten Code"""
        if not code or not isinstance(code, str):
            return ''
        
        import re
        cleaned = re.sub(r'^\s*```[\w]*\s*', '', code, flags=re.MULTILINE)  # Markdown code blocks
        cleaned = re.sub(r'\s*```\s*$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^\s*# Output:\s*$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^\s*# Ausgabe:\s*$', '', cleaned, flags=re.MULTILINE)
        
        return cleaned.strip()
    
    def extract_json(self, text: str) -> Dict[str, Any]:
        """Extrahiert JSON aus einer LLM-Antwort"""
        import json
        import re
        
        try:
            # Versuche direktes JSON-Parsing
            return json.loads(text)
        except json.JSONDecodeError:
            # Versuche JSON-Extraktion mit Regex
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    print(f'{self.agent_key}: JSON-Extraktion fehlgeschlagen')
                    return {}
            
            print(f'{self.agent_key}: Kein JSON in der Antwort gefunden')
            return {}
    
    def extract_code(self, text: str) -> str:
        """Extrahiert Code aus einer LLM-Antwort"""
        import re
        block = re.search(r"'start code'\n([\s\S]*?)\n'end code'", text, re.IGNORECASE)
        return block.group(1).strip() if block else text.strip()
    
    def log(self, level: str, message: str, data: Optional[Any] = None):
        """Loggt Worker-Aktivitäten"""
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        log_message = f'[{timestamp}] {self.agent_key}: {message}'
        
        icons = {
            'info': 'ℹ️',
            'warn': '⚠️',
            'error': '❌',
            'success': '✅'
        }
        
        icon = icons.get(level, '')
        print(f'{icon} {log_message}')
        
        if data:
            print(f'   Daten: {data}')

