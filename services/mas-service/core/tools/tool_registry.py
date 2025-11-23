"""
MCP Tool Registry
Verwaltet alle verfügbaren Tools für Agents
"""

from typing import Dict, List, Any, Callable, Optional
import json


class ToolRegistry:
    """
    Registry für MCP Tools
    Verwaltet Tool-Definitionen und Implementierungen
    """
    
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._implementations: Dict[str, Callable] = {}
    
    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        implementation: Callable
    ):
        """
        Registriert ein neues Tool
        
        Args:
            name: Tool-Name (z.B. 'csv_drop_columns')
            description: Beschreibung des Tools
            parameters: JSON Schema für Parameter
            implementation: Funktion, die das Tool ausführt
        """
        self._tools[name] = {
            'name': name,
            'description': description,
            'parameters': parameters
        }
        self._implementations[name] = implementation
    
    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Gibt das Schema eines Tools zurück"""
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Gibt alle registrierten Tools zurück"""
        return list(self._tools.values())
    
    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Führt ein Tool aus
        
        Args:
            name: Tool-Name
            arguments: Tool-Argumente
        
        Returns:
            Tool-Ergebnis
        """
        if name not in self._implementations:
            raise ValueError(f"Tool '{name}' nicht gefunden")
        
        implementation = self._implementations[name]
        return implementation(**arguments)
    
    def get_tools_for_prompt(self) -> str:
        """
        Gibt eine formatierte Liste aller Tools für LLM-Prompts zurück
        """
        if not self._tools:
            return ""
        
        tools_text = "\n## Verfügbare Tools:\n\n"
        
        for tool_name, tool_def in self._tools.items():
            tools_text += f"### {tool_def['name']}\n"
            tools_text += f"{tool_def['description']}\n\n"
            
            if 'parameters' in tool_def and 'properties' in tool_def['parameters']:
                tools_text += "Parameter:\n"
                for param_name, param_def in tool_def['parameters']['properties'].items():
                    param_type = param_def.get('type', 'unknown')
                    param_desc = param_def.get('description', '')
                    required = param_name in tool_def['parameters'].get('required', [])
                    req_marker = " (erforderlich)" if required else " (optional)"
                    tools_text += f"  - {param_name} ({param_type}){req_marker}: {param_desc}\n"
            
            tools_text += "\n"
        
        tools_text += "\nUm ein Tool zu verwenden, antworte im folgenden Format:\n"
        tools_text += "```json\n"
        tools_text += "{\n"
        tools_text += '  "tool": "tool_name",\n'
        tools_text += '  "arguments": {\n'
        tools_text += '    "param1": "value1",\n'
        tools_text += '    "param2": "value2"\n'
        tools_text += "  }\n"
        tools_text += "}\n"
        tools_text += "```\n"
        
        return tools_text


# Globale Registry-Instanz
_registry = None


def get_tool_registry() -> ToolRegistry:
    """Gibt die globale Tool-Registry zurück"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry

