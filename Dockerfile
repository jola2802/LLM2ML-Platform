# Multi-stage build für optimierte Image-Größe
FROM node:18-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./

# Install dependencies with legacy peer deps to avoid conflicts
RUN npm ci --legacy-peer-deps

COPY frontend/ ./

# Fix Rollup Alpine Linux issue by setting environment variables
ENV ROLLUP_SKIP_NATIVE=true
ENV NODE_ENV=production

# Build frontend
RUN npm run build

# Backend stage - Python wird NICHT mehr benötigt (läuft im python-service)
FROM node:18-slim AS backend

# Nur curl für Health-Check installieren (kein Python mehr!)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Backend dependencies installieren
COPY backend/package*.json ./backend/
WORKDIR /app/backend
RUN npm ci --only=production

# Kopiere Backend-Code (direkt nach /app/backend, nicht /app/backend/backend)
WORKDIR /app/backend
COPY backend/ ./

# Kopiere Config-Dateien nach /app/config/
WORKDIR /app
COPY config/ ./config/

# Frontend Build kopieren
RUN mkdir -p /app/frontend
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Erstelle benötigte Verzeichnisse
RUN mkdir -p uploads models scripts logs

# Umgebungsvariablen setzen
ENV NODE_ENV=production
ENV PORT=3000

# Port freigeben
EXPOSE $PORT

# Arbeitsverzeichnis für Start
WORKDIR /app/backend

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:$PORT/ || exit 1

CMD ["npm", "start"]
