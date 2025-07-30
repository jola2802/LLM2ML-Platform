#!/bin/bash

echo "🚀 Starting deployment..."

# Frontend builden
echo "📦 Building frontend..."
cd frontend
npm install
npm run build
cd ..

# Backend starten (Production)
echo "🔧 Starting production server..."
cd backend
npm install
NODE_ENV=production npm start 