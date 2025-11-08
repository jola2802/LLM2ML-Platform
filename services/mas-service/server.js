import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { setupLLMRoutes } from './routes/llm_routes.js';
import { setupAgentRoutes } from './routes/agent_routes.js';
import { setupQueueRoutes } from './routes/queue_routes.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3002;

// CORS für Hauptserver
app.use(cors({
    origin: process.env.API_GATEWAY_URL || 'http://localhost:3001',
    credentials: true
}));

app.use(express.json({ limit: '50mb' }));

// Health-Check Endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        service: 'mas-service',
        port: PORT,
        timestamp: new Date().toISOString()
    });
});

// API-Routen
setupLLMRoutes(app);
setupAgentRoutes(app);
setupQueueRoutes(app);

// Fehlerbehandlung
app.use((err, req, res, next) => {
    console.error('Fehler:', err);
    res.status(500).json({
        error: err.message || 'Interner Serverfehler',
        service: 'mas-service'
    });
});

// Server starten
app.listen(PORT, () => {
    console.log(`MAS-Service (Model Analysis Service) läuft auf Port ${PORT}`);
    console.log(`OLLAMA_URL: ${process.env.OLLAMA_URL || 'http://llm-service:11434'}`);
});

// Graceful Shutdown
process.on('SIGTERM', async () => {
    console.log('SIGTERM empfangen, fahre Service herunter...');
    // Cleanup falls nötig
    process.exit(0);
});

process.on('SIGINT', async () => {
    console.log('SIGINT empfangen, fahre Service herunter...');
    // Cleanup falls nötig
    process.exit(0);
});

