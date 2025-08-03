# Backend Services

## Automatische Datenexploration & Token-Optimierung

### Problem
Die ursprüngliche Implementierung hatte zwei Hauptprobleme:
1. **Mehrfache Datei-Uploads**: Dateien wurden mehrmals an das LLM hochgeladen
2. **Hohe Token-Kosten**: Komplette Datensätze wurden an das LLM gesendet
3. **JSON-Serialisierungsfehler**: numpy/pandas Objekte konnten nicht serialisiert werden
4. **Eingeschränkte Dateiformat-Unterstützung**: Nur wenige Dateitypen wurden unterstützt

### Lösung
Ein robustes, zweistufiges Optimierungssystem wurde implementiert:

#### 1. File-Cache-System
- **Dateien werden nur einmal hochgeladen** und im Speicher zwischengespeichert
- **Wiederverwendung** bei nachfolgenden Anfragen
- **Automatische Erkennung** bereits hochgeladener Dateien

#### 2. Robuste Automatische Datenexploration
- **Python-basierte Analyse** mit `ydata-profiling` und `sweetviz`
- **Strukturierte Datenübersicht** statt komplette Datensätze
- **LLM-optimierte Zusammenfassung** mit allen wichtigen Metriken
- **Sichere JSON-Serialisierung** für alle Datentypen
- **Umfassende Dateiformat-Unterstützung**

### Implementierung

#### Robuste Datenexploration (`data_exploration.py`)
```python
# Sichere JSON-Serialisierung für numpy/pandas Objekte
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        # ... weitere Konvertierungen

def load_data_safely(file_path):
    """Lädt Daten sicher mit verschiedenen Methoden"""
    # Unterstützt CSV, Excel, JSON, Parquet, Feather, Pickle, Text
    # Automatische Encoding-Erkennung
    # Fallback-Mechanismen für problematische Dateien
```

#### JavaScript-Service (`data_exploration.js`)
```javascript
export async function performDataExploration(filePath, venvDir) {
  // Plattform-unabhängige Python-Ausführung
  // Automatische venv-Erkennung
  // Robuste Fehlerbehandlung
}
```

#### Cache-System (`llm.js`)
```javascript
// File-Cache für bereits hochgeladene Dateien
const fileCache = new Map();

async function uploadFileOrGetFromCache(filePath, genAI) {
  if (fileCache.has(filePath)) {
    return fileCache.get(filePath);
  }
  
  // Datei hochladen und in Cache speichern
  const file = await genAI.files.upload({...});
  fileCache.set(filePath, file);
  return file;
}
```

### Unterstützte Dateiformate

#### Vollständig unterstützt
- **CSV**: Automatische Encoding-Erkennung (UTF-8, Latin-1, CP1252, ISO-8859-1)
- **Excel**: XLSX/XLS mit openpyxl Fallback
- **JSON**: Standard und JSONL (JSON Lines)
- **Parquet**: Effiziente binäre Speicherung
- **Feather**: Schnelle pandas-kompatible Speicherung
- **Pickle**: Python-spezifische Serialisierung
- **Text**: Automatische CSV-Erkennung oder Text-Behandlung

#### Automatische Fallbacks
- **Unbekannte Formate**: Versucht verschiedene Parsing-Methoden
- **Encoding-Probleme**: Testet mehrere Encodings
- **Beschädigte Dateien**: Graceful Error Handling

### Robuste Fehlerbehandlung

#### JSON-Serialisierung
- **Sichere Konvertierung** aller numpy/pandas Datentypen
- **NaN/None Behandlung** für fehlende Werte
- **Timestamp-Konvertierung** für Datumsfelder
- **Array/Series Konvertierung** für Listen

#### Datenanalyse
- **Try-Catch Blöcke** für jede Spalte
- **Fallback-Werte** bei Berechnungsfehlern
- **Detaillierte Fehlerprotokollierung**
- **Graceful Degradation** bei Problemen

### Neue API-Endpoints

#### Cache-Management
- `GET /api/cache/status` - File-Cache-Status
- `POST /api/cache/clear` - File-Cache leeren
- `GET /api/analysis-cache/status` - Datenanalyse-Cache-Status
- `POST /api/analysis-cache/clear` - Datenanalyse-Cache leeren

#### Datenexploration
- `POST /api/explore-data` - Automatische Datenexploration für eine Datei
- `GET /api/projects/:id/stats` - Datenstatistiken für ein Projekt

### Vorteile der Optimierung

#### Token-Einsparung
- **90%+ Reduktion** der Token-Kosten
- **Keine kompletten Datensätze** mehr an LLM gesendet
- **Intelligente Zusammenfassung** mit allen wichtigen Metriken

#### Performance-Verbesserung
- **Schnellere Verarbeitung** durch lokale Analyse
- **Weniger API-Aufrufe** an Gemini
- **Bessere Skalierbarkeit** für große Datensätze

#### Robustheit
- **Keine JSON-Serialisierungsfehler** mehr
- **Unterstützung aller gängigen Dateiformate**
- **Automatische Encoding-Erkennung**
- **Graceful Error Handling**

#### Qualitätsverbesserung
- **Professionelle Datenanalyse** mit etablierten Tools
- **Konsistente Metriken** (Korrelationen, Ausreißer, etc.)
- **Detaillierte Insights** für bessere ML-Entscheidungen

### Monitoring und Verwaltung

#### Cache-Status überprüfen
```bash
# File-Cache
curl http://localhost:3000/api/cache/status

# Datenanalyse-Cache
curl http://localhost:3000/api/analysis-cache/status
```

#### Cache leeren
```bash
# File-Cache
curl -X POST http://localhost:3000/api/cache/clear

# Datenanalyse-Cache
curl -X POST http://localhost:3000/api/analysis-cache/clear
```

#### Datenexploration testen
```bash
curl -X POST http://localhost:3000/api/explore-data \
  -H "Content-Type: application/json" \
  -d '{"filePath": "/path/to/your/data.csv"}'
```

### Technische Details

#### Automatische Erkennung
- **Datentypen**: Numerisch, kategorisch, datetime, object
- **Korrelationen**: Starke Beziehungen zwischen Variablen
- **Ausreißer**: IQR-basierte Erkennung mit Null-Check
- **Fehlende Werte**: Detaillierte Analyse und Prozentsätze

#### LLM-Optimierung
- **Strukturierte Zusammenfassung** mit Markdown-Format
- **Wichtige Metriken** für ML-Entscheidungen
- **Beispieldaten** für Kontext
- **Automatische Alerts** von ydata-profiling

#### Plattform-Unterstützung
- **Windows**: Automatische venv-Scripts-Erkennung
- **Unix/Linux/Mac**: Standard venv/bin/python Pfad
- **Fallback**: System-Python wenn venv nicht verfügbar

## Weitere Services

### LLM Integration
- `llm.js`: Hauptservice für LLM-API-Aufrufe mit File-Cache
- `llm_api.js`: LLM-Empfehlungen und Performance-Evaluation
- `file_analysis.js`: Datei-Analyse mit automatischer Datenexploration
- `data_exploration.js`: Automatische Datenexploration und LLM-Zusammenfassung

### Python Code Generation
- `python_generator.js`: LLM-basierte Python-Script-Generierung
- `code_exec.js`: Python-Script-Ausführung und Vorhersagen

### Datenbank
- `db.js`: Projekt-Management und Datenbank-Operationen

### Logging
- `log.js`: LLM-Kommunikations-Logging 