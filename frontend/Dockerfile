# Frontend Dockerfile - Nginx für statische Dateien
FROM node:18-slim AS builder

WORKDIR /app

# Installiere Dependencies
COPY package*.json ./
RUN npm ci --legacy-peer-deps

# Kopiere Frontend-Code
COPY . .

# Fix Rollup Alpine Linux issue
ENV ROLLUP_SKIP_NATIVE=true
ENV NODE_ENV=production

# Build Frontend
RUN npm run build

# Production Stage - Nginx
FROM nginx:alpine

# Kopiere Build-Ergebnis
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx-Konfiguration für SPA (React Router)
# DNS-Resolver für dynamische Service-Auflösung
RUN echo 'resolver 127.0.0.11 valid=30s; \
\
server { \
    listen 80; \
    server_name _; \
    root /usr/share/nginx/html; \
    index index.html; \
    \
    # Gzip compression \
    gzip on; \
    gzip_vary on; \
    gzip_min_length 1024; \
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss application/json application/javascript; \
    \
    # API-Proxy zum Backend mit dynamischer DNS-Auflösung \
    location /api { \
        set $backend "ml-platform-backend:3000"; \
        proxy_pass http://$backend/api; \
        proxy_http_version 1.1; \
        proxy_set_header Upgrade $http_upgrade; \
        proxy_set_header Connection "upgrade"; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
        proxy_cache_bypass $http_upgrade; \
        \
        # Timeouts für langsame Backend-Requests \
        proxy_connect_timeout 60s; \
        proxy_send_timeout 60s; \
        proxy_read_timeout 60s; \
    } \
    \
    # Statische Dateien \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
    \
    # Cache für statische Assets \
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ { \
        expires 1y; \
        add_header Cache-Control "public, immutable"; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

