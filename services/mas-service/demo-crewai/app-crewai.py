#!/usr/bin/env python
"""
CrewAI Anwendung mit Logging
Führt die ML-Agents Crew aus und loggt alle Ausgaben
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Lade .env Datei falls vorhanden
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ .env Datei geladen: {env_path}")
except ImportError:
    # python-dotenv ist optional
    pass

# Füge das ml_agents Verzeichnis zum Python-Pfad hinzu
project_root = Path(__file__).parent
ml_agents_path = project_root / "src" / "ml_agents"
sys.path.insert(0, str(ml_agents_path))

from ml_agents.src.ml_agents.crew import MlAgents

# Globale Variablen für Logging
log_file: Optional[object] = None
log_file_path: Optional[str] = None


class TeeOutput:
    """Klasse die Ausgaben sowohl in Konsole als auch in Datei schreibt"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


def init_logging():
    """Initialisiert das Logging-System"""
    global log_file, log_file_path
    
    # Erstelle logs Verzeichnis falls es nicht existiert
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Erstelle Log-Datei mit Timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = logs_dir / f"crewai_run_{timestamp}.log"
    
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    # Schreibe Header in Log-Datei
    log_file.write("=" * 80 + "\n")
    log_file.write("CREWAI ML-AGENTS EXECUTION LOG\n")
    log_file.write(f"Startzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 80 + "\n\n")
    log_file.flush()
    
    # Konfiguriere Python Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Leite stdout und stderr um
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeOutput(original_stdout, log_file)
    sys.stderr = TeeOutput(original_stderr, log_file)
    
    return log_file_path, original_stdout, original_stderr


def close_logging(original_stdout, original_stderr):
    """Schließt die Log-Datei und stellt stdout/stderr wieder her"""
    global log_file
    
    if log_file:
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(f"Endzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 80 + "\n")
        log_file.close()
        log_file = None
    
    # Stelle stdout/stderr wieder her
    if isinstance(sys.stdout, TeeOutput):
        sys.stdout = original_stdout
    if isinstance(sys.stderr, TeeOutput):
        sys.stderr = original_stderr


def find_csv_file() -> Path:
    """Findet die test_dataset.csv Datei"""
    # Versuche verschiedene Pfade
    possible_paths = [
        project_root.parent / "test_dataset.csv",  # Im mas-service Verzeichnis
        project_root / "test_dataset.csv",  # Im demo-crewai Verzeichnis
        Path("test_dataset.csv"),  # Im aktuellen Verzeichnis
    ]
    
    for path in possible_paths:
        if path.exists():
            return path.absolute()
    
    raise FileNotFoundError(
        "test_dataset.csv wurde nicht gefunden. "
        "Bitte stelle sicher, dass die Datei im mas-service Verzeichnis liegt."
    )


def run_crew():
    """Führt die Crew aus"""
    logger = logging.getLogger(__name__)
    
    try:
        # Prüfe Ollama-Konfiguration
        ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.2:latest')
        
        logger.info("=" * 80)
        logger.info("OLLAMA KONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Ollama URL: {ollama_url}")
        logger.info(f"Ollama Modell: {ollama_model}")
        logger.info("=" * 80 + "\n")
        
        # Finde CSV-Datei
        logger.info("Suche nach test_dataset.csv...")
        csv_path = find_csv_file()
        logger.info(f"✅ CSV-Datei gefunden: {csv_path}")
        
        # Erstelle Inputs für die Crew
        inputs = {
            'topic': 'Machine Learning Modell für Kreditwürdigkeit',
            'current_year': str(datetime.now().year),
            'csv_path': str(csv_path)
        }
        
        logger.info("=" * 80)
        logger.info("STARTE CREWAI ML-AGENTS")
        logger.info("=" * 80)
        logger.info(f"Thema: {inputs['topic']}")
        logger.info(f"CSV-Datei: {inputs['csv_path']}")
        logger.info(f"LLM Provider: Ollama ({ollama_model})")
        logger.info("=" * 80 + "\n")
        
        # Erstelle und starte die Crew
        crew = MlAgents().crew()
        logger.info("Crew wurde initialisiert. Starte Ausführung...\n")
        
        # Führe die Crew aus
        result = crew.kickoff(inputs=inputs)
        
        logger.info("\n" + "=" * 80)
        logger.info("CREW AUSFÜHRUNG ABGESCHLOSSEN")
        logger.info("=" * 80)
        logger.info(f"Ergebnis: {result}")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Fehler bei der Crew-Ausführung: {e}", exc_info=True)
        raise


def main():
    """Hauptfunktion"""
    original_stdout = None
    original_stderr = None
    
    try:
        # Initialisiere Logging
        log_path, original_stdout, original_stderr = init_logging()
        
        print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                    CREWAI ML-AGENTS AUSFÜHRUNG                           ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
        print(f"📝 Alle Ausgaben werden in Log-Datei gespeichert: {log_path}\n")
        
        # Führe Crew aus
        result = run_crew()
        
        print(f"\n✅ Crew erfolgreich ausgeführt!")
        print(f"📝 Log-Datei: {log_path}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Ausführung wurde vom Benutzer abgebrochen.")
        if log_file_path:
            print(f"📝 Ausgaben wurden in Log-Datei gespeichert: {log_file_path}")
        return 1
        
    except Exception as error:
        print(f"\n\n❌ Unerwarteter Fehler: {error}")
        import traceback
        traceback.print_exc()
        if log_file_path:
            print(f"📝 Fehlerausgaben wurden in Log-Datei gespeichert: {log_file_path}")
        return 1
        
    finally:
        # Stelle sicher, dass Logging geschlossen wird
        if original_stdout and original_stderr:
            close_logging(original_stdout, original_stderr)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

