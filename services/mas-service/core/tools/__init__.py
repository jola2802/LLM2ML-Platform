"""
MCP Tools für Agents
"""

from .csv_tools import CSV_TOOLS
from .tool_registry import ToolRegistry, get_tool_registry

# Registriere alle CSV-Tools bei Initialisierung
def _register_csv_tools():
    """Registriert alle CSV-Tools in der globalen Registry"""
    registry = get_tool_registry()
    for tool_def in CSV_TOOLS:
        registry.register_tool(
            name=tool_def['name'],
            description=tool_def['description'],
            parameters=tool_def['parameters'],
            implementation=tool_def['implementation']
        )

# Automatische Registrierung beim Import
_register_csv_tools()

__all__ = ['CSV_TOOLS', 'ToolRegistry', 'get_tool_registry']
