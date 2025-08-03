# Multi-stage build für optimierte Image-Größe
FROM node:18-alpine AS frontend-builder

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

# Backend stage - verwende debian für bessere Python-Kompatibilität
FROM node:18-slim AS backend

# Install system dependencies für Python und Build-Tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Erstelle Python Virtual Environment
# RUN python3 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# Kopiere Python requirements zuerst für besseres Caching
COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# Backend dependencies
COPY backend/package*.json ./backend/
WORKDIR /app/backend
RUN npm ci --only=production

# Kopiere Backend-Code
COPY backend/ ./

# Kopiere Frontend Build
COPY --from=frontend-builder /app/frontend/dist ../frontend/dist

# Erstelle benötigte Verzeichnisse
RUN mkdir -p uploads models scripts logs

# Backend static file serving konfigurieren
WORKDIR /app

# Umgebungsvariablen setzen
ENV NODE_ENV=production
ENV PORT=3001

# Port freigeben (dynamisch über ENV konfigurierbar)
EXPOSE $PORT

# Arbeitsverzeichnis für Start
WORKDIR /app/backend

# Healthcheck hinzufügen
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:$PORT/ || exit 1

CMD ["npm", "start"] 