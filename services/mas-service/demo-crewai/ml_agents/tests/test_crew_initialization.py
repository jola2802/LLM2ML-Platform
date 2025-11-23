"""
Tests für die Crew-Initialisierung
"""
import pytest
import os
import sys
from pathlib import Path

# Füge das src Verzeichnis zum Python-Pfad hinzu
test_dir = Path(__file__).parent
src_dir = test_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from ml_agents.crew import MlAgents


class TestCrewInitialization:
    """Test-Klasse für die Crew-Initialisierung"""
    
    @pytest.fixture
    def ml_agents(self):
        """Erstellt eine Instanz der MlAgents Crew"""
        return MlAgents()
    
    def test_crew_can_be_created(self, ml_agents):
        """Testet, ob die Crew erstellt werden kann"""
        assert ml_agents is not None
    
    def test_crew_has_agents(self, ml_agents):
        """Testet, ob die Crew Agenten hat"""
        crew_instance = ml_agents.crew()
        assert crew_instance is not None
        assert hasattr(crew_instance, 'agents')
        assert len(crew_instance.agents) > 0
    
    def test_crew_has_tasks(self, ml_agents):
        """Testet, ob die Crew Tasks hat"""
        crew_instance = ml_agents.crew()
        assert crew_instance is not None
        assert hasattr(crew_instance, 'tasks')
        assert len(crew_instance.tasks) > 0
    
    def test_data_analyst_agent_exists(self, ml_agents):
        """Testet, ob der Data Analyst Agent existiert"""
        crew_instance = ml_agents.crew()
        # Prüfe, ob ein Agent mit "Data Analyst" im Namen existiert
        assert any("Data Analyst" in agent.role or "data_analyst" in str(agent.role).lower() for agent in crew_instance.agents)
    
    def test_data_analyst_has_csv_tool(self, ml_agents):
        """Testet, ob der Data Analyst Agent das CSV-Reader Tool hat"""
        data_analyst = ml_agents.data_analyst()
        assert data_analyst is not None
        assert hasattr(data_analyst, 'tools')
        assert len(data_analyst.tools) > 0
        # Prüfe, ob csv_reader Tool vorhanden ist
        tool_names = [tool.name if hasattr(tool, 'name') else str(tool) for tool in data_analyst.tools]
        assert any("csv_reader" in str(name).lower() for name in tool_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
