# ML Platform Backend

Backend für die No-Code ML Platform mit SQLite-Datenbank und echter Python-Ausführung.

## Voraussetzungen

- Node.js (Version 16 oder höher)
- Python 3.x mit folgenden Paketen:
  - pandas
  - scikit-learn
  - joblib
  - numpy

## Installation

```bash
# Dependencies installieren
npm install

# Python-Pakete installieren (falls nicht vorhanden)  
pip install pandas scikit-learn joblib numpy xgboost

# Environment-Variablen konfigurieren
cp example.env .env
# Dann .env bearbeiten und API_KEY setzen
```

## Starten

```bash
# Entwicklungsserver starten
npm run dev

# Oder normaler Start
npm start
```

Der Server läuft auf Port 3001.

## API Endpoints

- `GET /api/projects` - Alle Projekte abrufen
- `POST /api/projects` - Neues Projekt erstellen
- `GET /api/projects/:id` - Einzelnes Projekt abrufen
- `DELETE /api/projects/:id` - Projekt löschen
- `GET /api/projects/:id/download` - Trainiertes Modell herunterladen

## Funktionen

✅ **LLM-basierte Python-Script-Generierung** - Intelligente, maßgeschneiderte Scripts
✅ **SQLite-Datenbank** für persistente Speicherung
✅ **Echte Python-Ausführung** für ML-Training  
✅ **8 ML-Algorithmen** (RandomForest, XGBoost, SVM, etc.)
✅ **Model-Export** als .pkl Dateien
✅ **Intelligente Datenanalyse** und Preprocessing
✅ **Performance-Metriken Extraktion**
✅ **Asynchrones Training** mit Status-Updates
✅ **REST-API** für Predictions
✅ **Hyperparameter-Optimierung**

## Verzeichnisstruktur

- `/models/` - Gespeicherte trainierte Modelle
- `/scripts/` - Generierte Python-Scripts
- `/uploads/` - Hochgeladene Dateien
- `projects.db` - SQLite-Datenbank 