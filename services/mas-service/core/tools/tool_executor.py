"""
Tool Executor für MCP Tools
Führt Tools aus und integriert sie in LLM-Calls
"""

from typing import Dict, Any, Optional
import json
import re
from .tool_registry import get_tool_registry


def extract_tool_calls_from_response(response_text: str) -> list:
    """
    Extrahiert Tool-Calls aus einer LLM-Antwort
    
    Erwartetes Format:
    ```json
    {
      "tool": "tool_name",
      "arguments": {...}
    }
    ```
    
    Oder mehrere Tools:
    ```json
    [
      {"tool": "tool1", "arguments": {...}},
      {"tool": "tool2", "arguments": {...}}
    ]
    ```
    """
    tool_calls = []
    
    # Suche nach JSON-Code-Blöcken
    json_pattern = r'```json\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```'
    matches = re.findall(json_pattern, response_text, re.IGNORECASE)
    
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            
            # Einzelnes Tool
            if isinstance(parsed, dict) and 'tool' in parsed:
                tool_calls.append(parsed)
            
            # Liste von Tools
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and 'tool' in item:
                        tool_calls.append(item)
        
        except json.JSONDecodeError:
            continue
    
    # Falls kein Code-Block gefunden, versuche direktes JSON
    if not tool_calls:
        try:
            parsed = json.loads(response_text.strip())
            if isinstance(parsed, dict) and 'tool' in parsed:
                tool_calls.append(parsed)
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and 'tool' in item:
                        tool_calls.append(item)
        except json.JSONDecodeError:
            pass
    
    return tool_calls


def execute_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Führt einen Tool-Call aus
    
    Args:
        tool_call: Dictionary mit 'tool' und 'arguments'
    
    Returns:
        Ergebnis des Tool-Calls
    """
    registry = get_tool_registry()
    tool_name = tool_call.get('tool')
    arguments = tool_call.get('arguments', {})
    
    if not tool_name:
        return {
            'success': False,
            'error': 'Tool-Name fehlt im Tool-Call'
        }
    
    try:
        result = registry.execute_tool(tool_name, arguments)
        return {
            'success': True,
            'tool': tool_name,
            'result': result
        }
    except Exception as e:
        return {
            'success': False,
            'tool': tool_name,
            'error': str(e)
        }


def process_llm_response_with_tools(response_text: str) -> Dict[str, Any]:
    """
    Verarbeitet eine LLM-Antwort und führt enthaltene Tool-Calls aus
    
    Args:
        response_text: LLM-Antwort-Text
    
    Returns:
        Dictionary mit verarbeiteter Antwort und Tool-Ergebnissen
    """
    tool_calls = extract_tool_calls_from_response(response_text)
    
    if not tool_calls:
        return {
            'has_tools': False,
            'response': response_text,
            'tool_results': []
        }
    
    tool_results = []
    for tool_call in tool_calls:
        result = execute_tool_call(tool_call)
        tool_results.append(result)
    
    return {
        'has_tools': True,
        'response': response_text,
        'tool_calls': tool_calls,
        'tool_results': tool_results
    }

