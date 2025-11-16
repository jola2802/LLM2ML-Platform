# MAS Service Dockerfile (Python)
# Multi-Stage Build für optimierte Image-Größe

# Stage 1: Build-Stage - Installiert Dependencies mit Build-Tools
FROM python:3.11-slim as builder

# Installiere Build-Tools (nur für diese Stage)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python Virtual Environment erstellen
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Python Dependencies installieren (ohne Cache für kleinere Image-Größe)
COPY services/mas-service/requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # PyTorch CPU-only installieren (deutlich kleiner als CUDA-Version, ~500MB vs mehrere GB)
    pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# Stage 2: Runtime-Stage - Nur Runtime-Dependencies
FROM python:3.11-slim

# Installiere nur Runtime-Dependencies (curl für Healthcheck)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Kopiere Virtual Environment von Builder-Stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Service-Code kopieren
COPY services/mas-service/ ./services/mas-service/

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

WORKDIR /app/services/mas-service
CMD ["python", "app.py"]
