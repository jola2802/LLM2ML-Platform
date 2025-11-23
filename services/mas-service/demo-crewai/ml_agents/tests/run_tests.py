#!/usr/bin/env python
"""
Einfaches Skript zum Ausführen der Tests
"""
import sys
import subprocess
from pathlib import Path

def main():
    """Führt die Tests aus"""
    test_dir = Path(__file__).parent
    
    print("=" * 80)
    print("ML-AGENTS MULTI-AGENT-SYSTEM TESTS")
    print("=" * 80)
    print()
    
    # Prüfe, ob pytest installiert ist
    try:
        import pytest
    except ImportError:
        print("❌ pytest ist nicht installiert!")
        print("Bitte installiere pytest mit: pip install pytest pytest-mock requests")
        return 1
    
    # Führe Tests aus
    print("Führe Tests aus...")
    print()
    
    # Optionen für pytest
    pytest_args = [
        str(test_dir),
        "-v",  # Verbose output
        "--tb=short",  # Kürzere Tracebacks
    ]
    
    # Prüfe, ob --slow übergeben wurde
    if "--slow" not in sys.argv:
        pytest_args.extend(["-m", "not slow"])
        print("ℹ️  Führe nur schnelle Tests aus (ohne LLM-Aufrufe)")
        print("   Verwende '--slow' um auch langsame Tests auszuführen")
        print()
    else:
        print("ℹ️  Führe alle Tests aus (inkl. langsamer Tests mit LLM-Aufrufen)")
        print()
    
    # Führe pytest aus
    exit_code = pytest.main(pytest_args)
    
    print()
    print("=" * 80)
    if exit_code == 0:
        print("✅ Alle Tests erfolgreich!")
    else:
        print(f"❌ Einige Tests sind fehlgeschlagen (Exit Code: {exit_code})")
    print("=" * 80)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
