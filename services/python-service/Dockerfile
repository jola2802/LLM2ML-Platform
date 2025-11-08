# Python Service Dockerfile
FROM node:18-slim

# Python und Build-Tools installieren
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python Virtual Environment erstellen
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Python Dependencies installieren
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Node.js Dependencies installieren
COPY backend/package*.json ./
RUN npm ci --only=production

# Service-Code kopieren
COPY services/python-service/ ./services/python-service/

# Config-Dateien kopieren
COPY config/ ./config/

# Erstelle benötigte Verzeichnisse
RUN mkdir -p scripts models uploads

ENV NODE_ENV=production
ENV PORT=3003

EXPOSE 3003

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3003/health', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})"

WORKDIR /app/services/python-service
CMD ["node", "server.js"]

