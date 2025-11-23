"""
Test-Skript für die ML-Pipeline
Testet die komplette Pipeline mit einem Beispiel-Datensatz
"""

import asyncio
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from io import StringIO

# Füge Projekt-Root zum Python-Pfad hinzu
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.agents.pipline import run_simple_pipeline

# Import für detailliertes Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Globale Log-Datei
log_file = None
log_file_path = None

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

def init_log_file():
    """Initialisiert die Log-Datei mit Timestamp"""
    global log_file, log_file_path
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(project_root, f"pipeline_test_log_{timestamp}.log")
    
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    # Schreibe Header in Log-Datei
    log_file.write("=" * 80 + "\n")
    log_file.write(f"ML-PIPELINE TEST LOG\n")
    log_file.write(f"Startzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 80 + "\n\n")
    log_file.flush()
    
    # Leite stdout und stderr um
    sys.stdout = TeeOutput(sys.stdout, log_file)
    sys.stderr = TeeOutput(sys.stderr, log_file)
    
    return log_file_path

def close_log_file():
    """Schließt die Log-Datei und stellt stdout/stderr wieder her"""
    global log_file
    
    if log_file:
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(f"Endzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 80 + "\n")
        log_file.close()
        log_file = None
    
    # Stelle stdout/stderr wieder her (falls nötig)
    if isinstance(sys.stdout, TeeOutput):
        sys.stdout = sys.stdout.files[0]
    if isinstance(sys.stderr, TeeOutput):
        sys.stderr = sys.stderr.files[0]

def create_test_dataset(output_path: str = "test_dataset.csv") -> str:
    """
    Erstellt einen Test-Datensatz für Klassifikation
    """

    # Check if output_path exists
    if os.path.exists(output_path):
        print(f"📊 Test-Datensatz bereits vorhanden: {output_path}")
        return output_path
    
    print(f"\n📊 Erstelle Test-Datensatz: {output_path}")
    
    # Erstelle synthetische Daten für Klassifikation
    np.random.seed(42)
    n_samples = 500
    
    # Features
    age = np.random.randint(18, 80, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    credit_score = np.random.normal(650, 100, n_samples)
    years_employed = np.random.randint(0, 40, n_samples)
    
    # Target Variable (Kreditwürdigkeit basierend auf Features)
    # Einfache Regel: Kreditwürdig wenn income > 45000 und credit_score > 600
    creditworthy = ((income > 45000) & (credit_score > 600)).astype(int)
    # Füge etwas Rauschen hinzu
    noise = np.random.random(n_samples) < 0.15
    creditworthy = (creditworthy ^ noise).astype(int)
    
    # Erstelle DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'years_employed': years_employed,
        'creditworthy': creditworthy
    })
    
    # Speichere CSV
    full_path = os.path.join(project_root, output_path)
    df.to_csv(full_path, index=False)
    
    print(f"✅ Test-Datensatz erstellt: {full_path}")
    
    return full_path

def print_section(title: str, char: str = "="):
    """Druckt einen Abschnitts-Titel"""
    print(f"\n{char * 80}")
    print(f"{title}")
    print(f"{char * 80}")

def print_dict(data: dict, indent: int = 0, max_depth: int = 3, current_depth: int = 0):
    """Druckt ein Dictionary formatiert"""
    if current_depth >= max_depth:
        print(" " * indent + "...")
        return
    
    for key, value in data.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_dict(value, indent + 2, max_depth, current_depth + 1)
        elif isinstance(value, list):
            print(" " * indent + f"{key}: [List mit {len(value)} Einträgen]")
            if len(value) > 0 and isinstance(value[0], dict):
                for i, item in enumerate(value[:3]):  # Zeige nur erste 3
                    print(" " * (indent + 2) + f"[{i}]:")
                    print_dict(item, indent + 4, max_depth, current_depth + 1)
                if len(value) > 3:
                    print(" " * (indent + 2) + f"... und {len(value) - 3} weitere")
        elif isinstance(value, str) and len(value) > 200:
            print(" " * indent + f"{key}: {value[:200]}...")
        else:
            print(" " * indent + f"{key}: {value}")

async def test_pipeline():
    """Testet die komplette Pipeline"""
    
    print_section("🚀 ML-PIPELINE TEST", "=")
    print(f"Startzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Erstelle Test-Datensatz
    print_section("📊 SCHRITT 1: Test-Datensatz erstellen", "-")
    test_dataset_path = create_test_dataset("test_dataset.csv")
    
    # 2. Erstelle Projekt-Dictionary
    print_section("📝 SCHRITT 2: Projekt-Dictionary erstellen", "-")
    project = {
        'id': 'test_project_001',
        'name': 'Test Projekt - Kreditwürdigkeit',
        'csvFilePath': test_dataset_path,
        'userPreferences': 'Klassifikationsproblem: Vorhersage der Kreditwürdigkeit',
        'llmRecommendations': {
            'targetVariable': 'creditworthy',
            'modelType': 'Classification',
            'algorithm': 'RandomForestClassifier',
            'features': ['age', 'income', 'credit_score', 'years_employed']
        }
    }
    
    print("Projekt-Konfiguration:")
    print_dict(project)
    
    # 3. Führe Pipeline aus
    print_section("🔄 SCHRITT 3: Pipeline-Ausführung", "-")
    print("Starte Pipeline mit max_iterations=2 (für schnelleren Test)")
    
    
    try:
        # Erstelle eine Kopie für besseres Debugging
        import copy
        project_copy = copy.deepcopy(project)
        
        # Prüfe ob CSV-Datei existiert
        if not os.path.exists(test_dataset_path):
            raise FileNotFoundError(f"Test-Datensatz nicht gefunden: {test_dataset_path}")
        
        # print(f"📁 Verwendete CSV-Datei: {test_dataset_path}")
        # print(f"📊 Dateigröße: {os.path.getsize(test_dataset_path)} Bytes")
        
        # Starte Pipeline
        print("\n" + "=" * 80)
        print("🚀 STARTE PIPELINE...")
        print("=" * 80 + "\n")
        
        result = await run_simple_pipeline(project_copy, max_iterations=2)
        
        print_section("✅ PIPELINE ERFOLGREICH ABGESCHLOSSEN", "=")
        
        # Validiere Ergebnis
        if not result:
            print("⚠️  WARNUNG: Pipeline hat kein Ergebnis zurückgegeben")
            return None
        
        if not isinstance(result, str):
            print(f"⚠️  WARNUNG: Pipeline hat unerwarteten Typ zurückgegeben: {type(result)}")
            print(f"   Ergebnis: {result}")
            return result
        
        print(f"📏 Anzahl Zeilen: {len(result.split(chr(10)))}")
        print("-" * 80)
        
        return result
        
    except FileNotFoundError as error:
        print_section("❌ DATEI NICHT GEFUNDEN", "=")
        print(f"Fehler: {error}")
        print("\n💡 TIPP: Stelle sicher, dass der Test-Datensatz erstellt wurde.")
        raise
    except Exception as error:
        print_section("❌ PIPELINE FEHLGESCHLAGEN", "=")
        print(f"Fehlertyp: {type(error).__name__}")
        print(f"Fehlermeldung: {error}")
        import traceback
        print("\n" + "=" * 80)
        print("VOLLSTÄNDIGER TRACEBACK:")
        print("=" * 80)
        traceback.print_exc()
        print("\n💡 TIPP: Prüfe die Fehlermeldung oben und stelle sicher, dass:")
        print("   - Alle benötigten Module installiert sind")
        print("   - Die LLM-API erreichbar ist")
        print("   - Die CSV-Datei korrekt formatiert ist")
        raise

def test_individual_agents():
    """Testet einzelne Agents (optional)"""
    print_section("🔍 OPTIONAL: Einzelne Agents testen", "-")
    print("Diese Funktion kann erweitert werden, um einzelne Agents zu testen")
    print("Aktuell wird die komplette Pipeline getestet")

async def main():
    """Hauptfunktion"""
    start_time = datetime.now()
    
    try:
        # Teste komplette Pipeline
        result = await test_pipeline()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_section("📊 TEST ZUSAMMENFASSUNG", "=")
        
        if result and isinstance(result, str) and len(result.strip()) > 0:
            print("✅ Pipeline-Test erfolgreich abgeschlossen")
            print(f"📝 Generierter Code: {len(result)} Zeichen")
            print(f"📏 Anzahl Zeilen: {len(result.split(chr(10)))}")
        else:
            print("⚠️  Pipeline abgeschlossen, aber kein Code generiert")
            print(f"   Ergebnis-Typ: {type(result)}")
            print(f"   Ergebnis-Länge: {len(result) if result else 0}")
        
        print(f"📁 Test-Datensatz: test_dataset.csv")
        print(f"🕐 Startzeit: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕐 Endzeit: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Dauer: {duration:.2f} Sekunden ({duration/60:.2f} Minuten)")
        
        # Speichere generierten Code in Datei
        output_file = "test_generated_code.py"
        if result and isinstance(result, str) and len(result.strip()) > 0:
            output_path = os.path.join(project_root, output_file)
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result)
                print(f"💾 Generierter Code gespeichert in: {output_path}")
            except Exception as save_error:
                print(f"⚠️  Fehler beim Speichern des Codes: {save_error}")
        else:
            print("⚠️  Kein Code zum Speichern verfügbar")
        
        print("\n" + "=" * 80)
        if result and isinstance(result, str) and len(result.strip()) > 0:
            print("🎉 TEST ERFOLGREICH ABGESCHLOSSEN!")
        else:
            print("⚠️  TEST ABGESCHLOSSEN (mit Warnungen)")
        print("=" * 80)
        
        if log_file_path:
            print(f"\n📝 Alle Ausgaben wurden in Log-Datei gespeichert: {log_file_path}")
        
        # Schließe Log-Datei
        close_log_file()
        
        return 0
        
    except KeyboardInterrupt:
        print_section("⚠️  TEST ABGEBROCHEN", "=")
        print("Der Test wurde vom Benutzer abgebrochen (Ctrl+C)")
        if log_file_path:
            print(f"\n📝 Ausgaben bis zum Abbruch wurden in Log-Datei gespeichert: {log_file_path}")
        close_log_file()
        return 1
    except Exception as error:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_section("❌ TEST FEHLGESCHLAGEN", "=")
        print(f"Fehlertyp: {type(error).__name__}")
        print(f"Fehlermeldung: {error}")
        print(f"🕐 Startzeit: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕐 Endzeit: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Dauer bis Fehler: {duration:.2f} Sekunden")
        
        import traceback
        print("\n" + "=" * 80)
        print("VOLLSTÄNDIGER TRACEBACK:")
        print("=" * 80)
        traceback.print_exc()
        
        print("\n💡 DEBUGGING-TIPPS:")
        print("   1. Prüfe ob alle Python-Module installiert sind")
        print("   2. Stelle sicher, dass die LLM-API (Ollama) läuft")
        print("   3. Prüfe ob die CSV-Datei korrekt formatiert ist")
        print("   4. Schaue in die Logs oben nach spezifischen Fehlermeldungen")
        
        if log_file_path:
            print(f"\n📝 Alle Ausgaben wurden in Log-Datei gespeichert: {log_file_path}")
        
        close_log_file()
        
        return 1

if __name__ == "__main__":
    try:
        # Initialisiere Log-Datei früh, um auch Start-Meldungen zu erfassen
        log_path = init_log_file()
        
        print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                    ML-PIPELINE TEST-SKRIPT                               ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
        print(f"📝 Alle Ausgaben werden in Log-Datei gespeichert: {log_path}\n")
        
        exit_code = asyncio.run(main())
        
        # Stelle sicher, dass Log-Datei geschlossen wird
        close_log_file()
        
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test wurde vom Benutzer abgebrochen.")
        if log_file_path:
            print(f"📝 Ausgaben wurden in Log-Datei gespeichert: {log_file_path}")
        close_log_file()
        sys.exit(1)
    except Exception as error:
        print(f"\n\n❌ Unerwarteter Fehler beim Starten des Tests: {error}")
        import traceback
        traceback.print_exc()
        if log_file_path:
            print(f"📝 Fehlerausgaben wurden in Log-Datei gespeichert: {log_file_path}")
        close_log_file()
        sys.exit(1)

