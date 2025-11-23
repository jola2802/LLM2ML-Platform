"""
Tests für das CSV-Reader Tool
"""
import pytest
import os
import sys
from pathlib import Path

# Füge das src Verzeichnis zum Python-Pfad hinzu
test_dir = Path(__file__).parent
src_dir = test_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from ml_agents.tools.csv_reader_tool import CsvReaderTool


class TestCsvReaderTool:
    """Test-Klasse für das CSV-Reader Tool"""
    
    @pytest.fixture
    def csv_tool(self):
        """Erstellt eine Instanz des CSV-Reader Tools"""
        return CsvReaderTool()
    
    @pytest.fixture
    def test_csv_path(self):
        """Gibt den Pfad zur Test-CSV-Datei zurück"""
        csv_path = test_dir / "test_dataset.csv"
        return str(csv_path.absolute())
    
    def test_tool_initialization(self, csv_tool):
        """Testet, ob das Tool korrekt initialisiert wird"""
        assert csv_tool.name == "csv_reader"
        assert csv_tool.description is not None
        assert len(csv_tool.description) > 0
    
    def test_tool_reads_existing_csv(self, csv_tool, test_csv_path):
        """Testet, ob das Tool eine existierende CSV-Datei lesen kann"""
        result = csv_tool._run(test_csv_path)
        
        assert result is not None
        assert isinstance(result, str)
        assert "CSV-DATEI ERFOLGREICH GELADEN" in result or "CSV-Datei erfolgreich geladen" in result
        assert "age" in result or "Anzahl Zeilen" in result
    
    def test_tool_handles_missing_file(self, csv_tool):
        """Testet, ob das Tool korrekt mit fehlenden Dateien umgeht"""
        non_existent_path = "non_existent_file.csv"
        result = csv_tool._run(non_existent_path)
        
        assert result is not None
        assert "FEHLER" in result or "Fehler" in result
        assert "nicht gefunden" in result.lower()
    
    def test_tool_returns_data_summary(self, csv_tool, test_csv_path):
        """Testet, ob das Tool eine vollständige Datenzusammenfassung zurückgibt"""
        result = csv_tool._run(test_csv_path)
        
        # Prüfe, ob wichtige Informationen enthalten sind
        assert "Anzahl Zeilen" in result or "Zeilen" in result
        assert "Anzahl Spalten" in result or "Spalten" in result
        assert "Spaltennamen" in result or "Spalten" in result
    
    def test_tool_handles_empty_csv(self, csv_tool, tmp_path):
        """Testet, ob das Tool mit leeren CSV-Dateien umgeht"""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")
        
        result = csv_tool._run(str(empty_csv))
        assert result is not None
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
