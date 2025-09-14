# 🤖 No-Code ML Platform

Eine vollständige Machine Learning Platform, die es ermöglicht ohne Programmierkenntnisse ML-Modelle zu trainieren und bereitzustellen.

## ✨ Features

### 🔧 **Flexible Algorithmus-Auswahl**
- **Klassifikation**: Random Forest, Logistic Regression, SVM, XGBoost
- **Regression**: Random Forest, Linear Regression, SVR, XGBoost
- Automatische Algorithmus-Empfehlungen basierend auf Datentypen
- Komplexitäts-Bewertung für jeden Algorithmus

### 📊 **Intelligente Datenverarbeitung**
- **CSV-Upload & Analyse**: Automatische Datentyp-Erkennung
- **Erweiterte Preprocessing-Pipeline**: Skalierung, One-Hot-Encoding, Label-Encoding
- **Smart Data Cleaning**: Behandlung fehlender Werte
- **Feature-Engineering**: Automatische Preprocessing je nach Datentyp

### 🎯 **Erweiterte Metriken**
- **Klassifikation**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Classification Report
- **Regression**: MAE, MSE, RMSE, R², MAPE
- **Feature Importance**: Visualisierung der wichtigsten Features

### 🧠 **Dynamische Multi-Agent-Orchestrierung**
- **Autonome Agent-Entscheidungen**: Agents entscheiden eigenständig über nächste Schritte
- **Event-basiertes Tracking**: Real-time Verfolgung aller Agent-Aktivitäten
- **Intelligente Übergaben**: Agents können an jeden anderen Agent übergeben
- **Adaptive Pipeline**: Keine starre Reihenfolge - dynamische Workflows
- **Zentrale Agent-Konfiguration**: Jeder Agent kann eigenes LLM-Modell verwenden
- **Zero-Template-Ansatz**: Vollständig generierte Scripts statt starrer Templates
- **Lokale LLM-Provider**: Vollständig lokale Implementierung mit Ollama

### 🚀 **REST-API**
- **Training-API**: Automatisches Model-Training mit echten Python-Scripts (Worker-Pool)
- **Prediction-API**: Vorhersagen mit trainierten Modellen zur externen Nutzung und zu Testzwecken
- **Model-Export**: Download der trainierten .pkl-Modelle
- **Persistente Speicherung**: SQLite-Datenbank für alle Projekte
- **Monitoring & Scaling**: Endpoints für Queue-, Worker- und Scaling-Status
- **File & Cache Management**: Endpoints für Dateien, Analyse-Cache und Predict-Cache

### 🎨 **Moderne Benutzeroberfläche**
- Intuitive 3-Schritte Wizard für Projekt-Erstellung
- Echzeit-Datei-Analyse mit Spalten- und Zeilenzahl
- Live-Training-Status mit Polling
- Performance-Visualisierung mit Charts
- Responsive Design

## 🏗️ Architektur

```
ML-Platform/
├── backend/                      # Node.js + Express + SQLite
│   ├── server.js                 # Haupt-Server mit REST-API
│   ├── services/
│   │   ├── api/                  # Endpoints (projects, upload, analyze, predict, files, agents, ...)
│   │   ├── execution/            # Python-Exec, Code-Gen, Worker, Predict-Cache
│   │   ├── llm/                  # Dynamische Agents, Agent-Config, Queue, Tuning
│   │   ├── monitoring/           # Job-Queue, Scaling-Monitor, Logs
│   │   └── config/               # Worker-Scaling-Konfiguration
│   ├── models/                   # Gespeicherte .pkl-Modelle
│   ├── scripts/                  # Generierte Python-Scripte (Train/Predict)
│   ├── uploads/                  # Hochgeladene CSV-Dateien
│   └── services/python/venv/     # Virtuelle Umgebung für Python
├── frontend/                     # React + TypeScript
│   ├── components/               # UI-Komponenten
│   ├── services/                 # API-Services
│   └── types.ts                  # TypeScript-Definitionen
└── README.md
```

## 🚀 Installation & Start

### Voraussetzungen
- Node.js (Version 16+)
- Python 3.x

Richte vor dem Start ein virtuelles Python-Environment unter `backend/services/python/venv` ein und installiere die Requirements aus `backend/requirements.txt`:

```bash
# Windows (PowerShell)
python -m venv backend/services/python/venv
backend/services/python/venv/Scripts/pip install -r backend/requirements.txt

# macOS/Linux
python3 -m venv backend/services/python/venv
backend/services/python/venv/bin/pip install -r backend/requirements.txt
```

### Backend starten
```bash
cd backend
npm install
npm run dev          # Entwicklungsserver auf Port 3001
```

### Frontend starten  
```bash
cd frontend
npm install
npm run dev          # Entwicklungsserver auf Port 5173
```

## 📡 API Endpoints

### 🔄 Projekt-Management
```http
GET    /api/projects                          # Alle Projekte
GET    /api/projects/:id                      # Projekt-Details
POST   /api/projects                          # Projekt erstellen (klassisch)
POST   /api/projects/smart-create             # Projekt mit LLM-Empfehlungen
PUT    /api/projects/:id/code                 # Python-Code (+ optional Hyperparameter) aktualisieren
POST   /api/projects/:id/retrain              # Re-Training mit aktuellem Code
POST   /api/projects/:id/evaluate-performance # LLM-gestützte Performance-Insights
GET    /api/projects/:id/data-statistics      # Erweiterte Datenstatistiken zum Projekt
GET    /api/projects/:id/stats                # Basis-Stats zur Quelldatei
GET    /api/projects/:id/download             # Modell (.pkl) herunterladen
DELETE /api/projects/:id                      # Projekt löschen
```

### 📤 Datei-Upload & Analyse
```http
POST   /api/upload                 # Datei hochladen & Basisanalyse
POST   /api/analyze-data           # LLM-Empfehlungen auf manipulierte Spalten anwenden
POST   /api/explore-data           # Automatische Datenexploration (Cache-gestützt)
```

### 🎯 Prediction
```http
POST   /api/predict/:id            # Vorhersage mit trainiertem Modell
```

### 🤖 LLM-Management
```http
GET    /api/llm/config             # Aktuelle LLM-Konfiguration
GET    /api/llm/status             # Ollama-Status
GET    /api/llm/ollama/models      # Verfügbare Ollama-Modelle
POST   /api/llm/ollama/test        # Ollama-Verbindung testen
POST   /api/llm/ollama/config      # Ollama-Host/Default-Model anpassen
```

### 🤖 Agent-Management
```http
GET    /api/agents                 # Alle verfügbaren Agents abrufen
GET    /api/agents/:agentKey       # Spezifische Agent-Konfiguration
PUT    /api/agents/:agentKey/model # Agent-Modell aktualisieren
PUT    /api/agents/models/bulk     # Mehrere Agent-Modelle auf einmal ändern
PUT    /api/agents/models/all      # Alle Agents auf dasselbe Modell setzen
GET    /api/agents/stats           # Agent-Statistiken
POST   /api/agents/:agentKey/test  # Agent-Konfiguration testen

GET    /api/agents/activities      # Alle Agent-Activities
GET    /api/agents/activities/:projectId  # Agent-Activities für Projekt
GET    /api/agents/active          # Aktuell aktive Agents
DELETE /api/agents/activities/:projectId # Agent-Activities löschen
GET    /api/agents/activities/:projectId/stream # Real-time Agent-Updates
```

### 📈 Monitoring
```http
POST   /api/projects/:id/monitoring/init      # Baseline initialisieren
POST   /api/projects/:id/monitoring/event     # Prediction-Event loggen (optional mit truth)
GET    /api/projects/:id/monitoring/status    # Monitoring-Status abrufen
POST   /api/projects/:id/monitoring/reset     # Monitoring zurücksetzen
```

### 🗂️ Datei-Management
```http
GET    /api/files/:type                 # Dateien auflisten (scripts|models|uploads)
GET    /api/files/:type/:filename       # Datei-Info abrufen
DELETE /api/files/:type                 # Datei löschen (Body: { filePath })
GET    /api/files/storage/stats         # Aggregierte Speicher-Statistiken
```

### 🧠 Analyse-/File-Cache
```http
POST   /api/cache/clear                 # (Legacy) File-Cache Nachricht
GET    /api/cache/status                # (Legacy) File-Cache Status

POST   /api/analysis-cache/clear        # Datenanalyse-Cache leeren
GET    /api/analysis-cache/status       # Datenanalyse-Cache Status
```

### ⚡ Predict-Cache
```http
POST   /api/predict-cache/cleanup       # Alte Predict-Skripte bereinigen
GET    /api/predict-cache/status        # Überblick über gecachte Predict-Skripte
DELETE /api/predict-cache/project/:projectId   # Cache für Projekt löschen
DELETE /api/predict-cache/all           # Gesamten Predict-Cache leeren
```

### 📊 Scaling & Queue/Worker
```http
GET    /api/scaling/metrics             # Live-Skalierungsmetriken
GET    /api/scaling/report              # Detaillierter Report
GET    /api/scaling/history/:type       # Verlauf (type: python|llm)
GET    /api/scaling/utilization/:type   # Auslastungsanalyse
POST   /api/scaling/config              # Skalierungs-Konfiguration ändern
GET    /api/scaling/status              # Zusammengefasster Status für Dashboard

GET    /api/llm/queue/status            # LLM-Queue Status
POST   /api/llm/queue/cancel/:requestId # LLM-Request abbrechen

GET    /api/worker/queue-status         # Job-Queue + Workerpool Status
GET    /api/worker/jobs                 # Jobs (limit optional)
GET    /api/worker/jobs/:type           # Jobs nach Typ
GET    /api/worker/job/:jobId           # Einzelner Job
POST   /api/worker/job/:jobId/cancel    # Job abbrechen
GET    /api/worker/stats                # Worker/Queue Kennzahlen
```

## 🔬 Verwendung

### 1️⃣ **Projekt erstellen**
- CSV-Datei hochladen
- Automatische Datenanalyse 
- Algorithmus auswählen
- Zielvarible und Features festlegen

### 2️⃣ **Training starten**
- **Dynamische Multi-Agent-Pipeline** mit autonomen Entscheidungen
- **Agent-zu-Agent-Kommunikation** für optimale Code-Generierung
- **Real-time Agent-Tracking** im Frontend
- **Adaptive Preprocessing-Pipeline** je nach Datentypen
- **Echte scikit-learn/XGBoost Ausführung** mit optimierten Parametern
- **Live-Status-Updates** während des Trainings
- **Performance-Metriken** werden automatisch extrahiert

### 3️⃣ **Modell nutzen**
- **Predictions**: Über UI oder API-Endpoint
- **Export**: .pkl-Datei für lokale Nutzung
- **Analysis**: Performance-Charts und Feature Importance

## 🎯 API-Beispiele

### Prediction-Request
```bash
curl -X POST 'http://localhost:3001/api/predict/{PROJECT_ID}' \
  -H 'Content-Type: application/json' \
  -d '{
    "features": {
      "age": 35,
      "income": 50000,
      "experience": 10
    }
  }'
```

### Response
```json
{
  "prediction": "1"
}
```

## 🔧 Technologie-Stack

### Backend
- **Node.js** + **Express.js** - REST-API Server
- **SQLite** - Persistente Datenspeicherung  
- **Python Integration** - Echte ML-Pipeline mit scikit-learn
- **Worker Threads** - Python Worker Pool mit Auto-Scaling
- **Multer** - File-Upload-Handling
- **Logging** - REST & LLM Kommunikation

### Frontend  
- **React** + **TypeScript** - Moderne UI
- **Vite** - Build-Tool
- **Recharts** - Performance-Visualisierung
- **Tailwind CSS** - Styling

### Machine Learning
- **scikit-learn** - Standard ML-Algorithmen
- **XGBoost** - Gradient Boosting
- **pandas** - Datenmanipulation
- **joblib** - Model-Serialisierung

### LLM
- **Ollama** (Lokal) mit z. B. `mistral:latest`
- **LangGraph/LangChain** für Multi-Agent Orchestrierung (Code-Gen/Review)

## ⚙️ Umgebungsvariablen

- `PORT` (optional, Default: `3001`)
- `OLLAMA_URL` (Default: `http://127.0.0.1:11434`)

## 🎨 Screenshots

### Performance-Dashboard  
![Dashboard](docs/performance.png)

### Demo  
![Demo](docs/Demo-LLM2ML-09092025.mp4)
