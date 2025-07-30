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

# Backend stage
FROM node:18-alpine AS backend

WORKDIR /app
COPY backend/package*.json ./
RUN npm ci --only=production

COPY backend/ ./
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Python dependencies
RUN apk add --no-cache python3 py3-pip
RUN pip3 install pandas scikit-learn joblib numpy xgboost

EXPOSE 3001

CMD ["npm", "start"] 