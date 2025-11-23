"""
Basis-Worker-Klasse für alle spezialisierten Worker-Agents
"""

import asyncio
from typing import Any, Dict, Optional
from core.llm import llm
from core.agents.config_agent_network import get_agent_config, get_agent_model, log_agent_call
from core.tools.tool_executor import process_llm_response_with_tools, execute_tool_call
from core.tools.tool_registry import get_tool_registry

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
        from core.agents.prompts import BASE_WORKER_TEST_PROMPT, format_prompt
        
        test_prompt = format_prompt(BASE_WORKER_TEST_PROMPT, {
            'agentName': self.config['name']
        })
        
        try:
            # Verwende async-Version
            response = await self.call_llm(test_prompt, None, 1)
            result = response.get('result', '') if isinstance(response, dict) else str(response)
            return 'ok' in result.lower()
        except Exception as error:
            print(f'Test für {self.agent_key} fehlgeschlagen: {error}')
            return False
    
    async def call_llm(
        self, 
        prompt: str, 
        context: Optional[str] = None, 
        max_tokens: Optional[int] = None,
        use_tools: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Ruft die LLM-API auf mit Agent-spezifischen Einstellungen
        
        Args:
            prompt: Prompt-Text
            context: Optionaler Kontext (wird als file_path interpretiert)
            max_tokens: Maximale Token-Anzahl
            use_tools: Wenn True, werden verfügbare Tools dem Prompt hinzugefügt.
                      Wenn None, wird automatisch basierend auf Agent-Key entschieden.
                      Nur FEATURE_ENGINEER hat standardmäßig Tools aktiviert.
        
        Returns:
            LLM-Response mit optionalen Tool-Ergebnissen
        """
        log_agent_call(self.agent_key, self.model, 'LLM-Aufruf')
        
        tokens = max_tokens or self.config.get('maxTokens', 2048)
        
        # Bestimme ob Tools verwendet werden sollen
        # Standardmäßig nur für FEATURE_ENGINEER aktiviert
        if use_tools is None:
            use_tools = (self.agent_key == 'FEATURE_ENGINEER')
        
        # Füge Tool-Informationen zum Prompt hinzu (falls aktiviert)
        if use_tools:
            registry = get_tool_registry()
            tools_info = registry.get_tools_for_prompt()
            if tools_info:
                prompt = prompt + "\n\n" + tools_info
        
        try:
            # Verwende die async-Version direkt, da wir bereits in einer async-Funktion sind
            from core.llm.llm import call_llm_api_async
            
            # Versuche Queue zu verwenden (falls verfügbar)
            try:
                from core.llm.llm_queue import get_queue
                queue = get_queue()
                if queue:
                    # context wird als file_path interpretiert
                    try:
                        response = await queue.add_request(prompt, context, self.model, 3)
                        
                        # Verarbeite Tool-Calls (falls vorhanden)
                        if use_tools and isinstance(response, dict):
                            response_text = response.get('result', '')
                            if response_text:
                                tool_processing = process_llm_response_with_tools(response_text)
                                if tool_processing.get('has_tools'):
                                    response['tool_results'] = tool_processing.get('tool_results', [])
                                    response['tool_calls'] = tool_processing.get('tool_calls', [])
                        
                        return response
                    except asyncio.CancelledError:
                        # Request wurde abgebrochen
                        raise
                    except Exception as queue_error:
                        # Queue-Fehler: Fallback zu direkter API
                        self.log('warning', f'Queue-Fehler, verwende direkte API: {queue_error}')
                        pass
            except (ImportError, AttributeError) as e:
                # Queue nicht verfügbar, verwende direkte API
                pass
            
            # Fallback: Direkte async API-Call
            # context wird als file_path interpretiert
            response = await call_llm_api_async(prompt, context, self.model, 3)
            
            # Verarbeite Tool-Calls (falls vorhanden)
            if use_tools and isinstance(response, dict):
                response_text = response.get('result', '')
                if response_text:
                    tool_processing = process_llm_response_with_tools(response_text)
                    if tool_processing.get('has_tools'):
                        response['tool_results'] = tool_processing.get('tool_results', [])
                        response['tool_calls'] = tool_processing.get('tool_calls', [])
            
            return response
            
        except Exception as error:
            print(f'LLM-Aufruf für {self.agent_key} fehlgeschlagen: {error}')
            raise
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Führt ein Tool direkt aus (ohne LLM)
        
        Args:
            tool_name: Name des Tools
            arguments: Tool-Argumente
        
        Returns:
            Tool-Ergebnis
        """
        try:
            result = execute_tool_call({'tool': tool_name, 'arguments': arguments})
            return result
        except Exception as error:
            return {
                'success': False,
                'error': str(error)
            }
    
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

