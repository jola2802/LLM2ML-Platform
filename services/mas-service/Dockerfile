# MAS Service Dockerfile (Model Analysis Service - LLM Service)
FROM node:18-slim

# Installiere curl für Health-Checks
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Node.js Dependencies installieren
COPY backend/package*.json ./
RUN npm ci --only=production

# Service-Code kopieren
COPY services/mas-service/ ./services/mas-service/

# Config-Dateien kopieren
COPY config/ ./config/

# Erstelle benötigte Verzeichnisse
RUN mkdir -p logs

ENV NODE_ENV=production
ENV PORT=3002

EXPOSE 3002

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3002/health', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})"

WORKDIR /app/services/mas-service
CMD ["node", "server.js"]

