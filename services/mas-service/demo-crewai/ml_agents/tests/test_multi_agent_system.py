"""
Integrationstests für das Multi-Agent-System
"""
import pytest
import os
import sys
from pathlib import Path
from datetime import datetime

# Füge das src Verzeichnis zum Python-Pfad hinzu
test_dir = Path(__file__).parent
src_dir = test_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from ml_agents.crew import MlAgents


class TestMultiAgentSystem:
    """Test-Klasse für das vollständige Multi-Agent-System"""
    
    @pytest.fixture
    def ml_agents(self):
        """Erstellt eine Instanz der MlAgents Crew"""
        return MlAgents()
    
    @pytest.fixture
    def test_csv_path(self):
        """Gibt den Pfad zur Test-CSV-Datei zurück"""
        csv_path = test_dir / "test_dataset.csv"
        return str(csv_path.absolute())
    
    @pytest.fixture
    def test_inputs(self, test_csv_path):
        """Erstellt Test-Inputs für die Crew"""
        return {
            'topic': 'Machine Learning Modell für Kreditwürdigkeit',
            'current_year': str(datetime.now().year),
            'csv_path': test_csv_path
        }
    
    def test_crew_can_be_initialized(self, ml_agents):
        """Testet, ob die Crew initialisiert werden kann"""
        crew = ml_agents.crew()
        assert crew is not None
        assert len(crew.agents) > 0
        assert len(crew.tasks) > 0
    
    @pytest.mark.slow
    def test_data_analysis_task_can_run(self, ml_agents, test_inputs):
        """
        Testet, ob die Data Analysis Task ausgeführt werden kann.
        Markiert als 'slow', da es LLM-Aufrufe erfordert.
        """
        crew = ml_agents.crew()
        
        # Führe nur die erste Task aus (Data Analysis)
        # Dies ist ein schnellerer Test als der vollständige Workflow
        try:
            # Prüfe, ob Ollama verfügbar ist
            import requests
            ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
            try:
                response = requests.get(f"{ollama_url}/api/tags", timeout=2)
                if response.status_code != 200:
                    pytest.skip("Ollama ist nicht verfügbar")
            except:
                pytest.skip("Ollama ist nicht verfügbar")
            
            # Führe die Crew aus (kann lange dauern)
            result = crew.kickoff(inputs=test_inputs)
            
            assert result is not None
            # Prüfe, ob ein Ergebnis zurückgegeben wurde
            assert len(str(result)) > 0
            
        except Exception as e:
            # Wenn es ein Netzwerk- oder LLM-Problem ist, skip den Test
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                pytest.skip(f"LLM nicht verfügbar: {e}")
            else:
                raise
    
    def test_all_agents_have_required_attributes(self, ml_agents):
        """Testet, ob alle Agenten die erforderlichen Attribute haben"""
        crew = ml_agents.crew()
        
        for agent in crew.agents:
            assert hasattr(agent, 'role') or hasattr(agent, 'goal')
            assert agent.llm is not None
    
    def test_all_tasks_have_required_attributes(self, ml_agents):
        """Testet, ob alle Tasks die erforderlichen Attribute haben"""
        crew = ml_agents.crew()
        
        for task in crew.tasks:
            assert hasattr(task, 'description') or hasattr(task, 'expected_output')
            assert task.agent is not None
    
    def test_csv_path_is_valid(self, test_csv_path):
        """Testet, ob der CSV-Pfad gültig ist"""
        assert os.path.exists(test_csv_path), f"CSV-Datei nicht gefunden: {test_csv_path}"
        assert test_csv_path.endswith('.csv'), "Datei ist keine CSV-Datei"


class TestAgentTools:
    """Test-Klasse für Agent-Tools"""
    
    @pytest.fixture
    def ml_agents(self):
        """Erstellt eine Instanz der MlAgents Crew"""
        return MlAgents()
    
    def test_data_analyst_has_csv_reader_tool(self, ml_agents):
        """Testet, ob der Data Analyst das CSV-Reader Tool hat"""
        agent = ml_agents.data_analyst()
        assert agent.tools is not None
        assert len(agent.tools) > 0
        
        # Prüfe, ob csv_reader Tool vorhanden ist
        tool_found = False
        for tool in agent.tools:
            if hasattr(tool, 'name') and tool.name == "csv_reader":
                tool_found = True
                break
        
        assert tool_found, "csv_reader Tool nicht gefunden"
    
    def test_performance_analyzer_has_train_model_tool(self, ml_agents):
        """Testet, ob der Performance Analyzer das Train Model Tool hat"""
        agent = ml_agents.performance_analyzer()
        assert agent.tools is not None
        assert len(agent.tools) > 0
        
        # Prüfe, ob train_model_tool vorhanden ist
        tool_found = False
        for tool in agent.tools:
            if hasattr(tool, 'name') and "train" in tool.name.lower():
                tool_found = True
                break
        
        assert tool_found, "train_model_tool nicht gefunden"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
