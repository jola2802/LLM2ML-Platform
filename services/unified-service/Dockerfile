# Unified ML Service Dockerfile (Python)
FROM python:3.11-slim

# Installiere Build-Tools und curl
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python Virtual Environment erstellen
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Python Dependencies installieren
COPY services/unified-service/requirements.txt ./
RUN pip install -r requirements.txt

# Service-Code kopieren
COPY services/unified-service/ ./services/unified-service/

# Config-Dateien kopieren (wird optional durch Volume überschrieben)
COPY config/ ./config/

# Erstelle benötigte Verzeichnisse
RUN mkdir -p scripts models uploads logs

ENV NODE_ENV=production
ENV PORT=3002
ENV PYTHONUNBUFFERED=1
ENV VENV_DIR=/opt/venv
ENV SCRIPT_DIR=/app/scripts
ENV UPLOADS_DIR=/app/uploads
ENV MODELS_DIR=/app/models

EXPOSE 3002

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:3002/health || exit 1

WORKDIR /app/services/unified-service
CMD ["python", "app.py"]

