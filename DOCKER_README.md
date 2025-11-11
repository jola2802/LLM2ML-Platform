# ML Platform - Docker Setup

Diese Anwendung ist jetzt in separate Docker-Container aufgeteilt:

## Services

### 1. Frontend (Port 3000)
- React-basierte Benutzeroberfläche
- Served durch Nginx für optimale Performance
- Proxy für API-Anfragen zum Backend

### 2. Backend (Port 3001)
- Node.js/Express API-Server
- Python-Integration für ML-Modelle
- Datenbank-Management
- File-Upload-Handling

### 3. Ollama (Port 11434)
- Lokaler LLM-Service
- Vorinstalliertes kleines Modell: `llama3.2:1b`
- Für lokale Text-Generierung und Code-Analyse

## Installation und Start
### Start der Anwendung

```bash
# Alle Services starten
docker-compose up -d

# Logs anzeigen
docker-compose logs -f

# Einzelne Services starten
docker-compose up frontend -d
docker-compose up backend -d
docker-compose up ollama -d
```

### Stoppen der Anwendung

```bash
# Alle Services stoppen
docker-compose down

# Mit Volume-Löschung (Achtung: Daten gehen verloren!)
docker-compose down -v
```

## Ports

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:3001
- **Ollama API**: http://localhost:11434

## Volumes

- `./uploads`: Hochgeladene Dateien
- `./models`: Trainierte ML-Modelle
- `./logs`: Anwendungs-Logs
- `ollama_data`: Ollama-Modelle und Daten

## Health Checks

Alle Services haben Health Checks konfiguriert:

```bash
# Health Status prüfen
docker-compose ps

# Einzelne Service-Logs
docker-compose logs frontend
docker-compose logs backend
docker-compose logs ollama
```

## Entwicklung

### Frontend-Entwicklung
```bash
cd frontend
npm install
npm run dev
```

### Backend-Entwicklung
```bash
cd backend
npm install
npm run dev
```

## Troubleshooting

### Ollama-Modell nicht verfügbar
```bash
# Modell manuell herunterladen
docker-compose exec ollama ollama pull llama3.2:1b
```

### Port-Konflikte
Falls Ports bereits belegt sind, ändere die Port-Mappings in `docker-compose.yml`:

```yaml
ports:
  - "3001:3000"  # Frontend auf anderem Port
  - "3002:3001"  # Backend auf anderem Port
```

### Speicherplatz-Probleme
Ollama-Modelle können viel Speicherplatz benötigen. Prüfe verfügbaren Speicher:

```bash
docker system df
docker volume ls
```

## Performance-Optimierungen

- Frontend verwendet Nginx für bessere Performance
- Backend hat optimierte Python-Environment
- Ollama läuft mit kleinem Modell für bessere Geschwindigkeit 