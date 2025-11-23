"""
Pytest Konfiguration und gemeinsame Fixtures
"""
import pytest
import os
import sys
from pathlib import Path

# Füge das src Verzeichnis zum Python-Pfad hinzu
test_dir = Path(__file__).parent
src_dir = test_dir.parent / "src"
sys.path.insert(0, str(src_dir))


def pytest_configure(config):
    """Pytest Konfiguration"""
    # Registriere Marker
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session")
def test_data_dir():
    """Gibt das Test-Datenverzeichnis zurück"""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def test_csv_path(test_data_dir):
    """Gibt den Pfad zur Test-CSV-Datei zurück"""
    csv_path = test_data_dir / "test_dataset.csv"
    if not csv_path.exists():
        pytest.skip(f"Test-CSV-Datei nicht gefunden: {csv_path}")
    return str(csv_path.absolute())
