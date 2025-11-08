import axios from 'axios';

const MAS_SERVICE_URL = process.env.MAS_SERVICE_URL || 'http://localhost:3002';

const client = axios.create({
    baseURL: MAS_SERVICE_URL,
    timeout: 300000, // 5 Minuten für LLM-Requests
    headers: {
        'Content-Type': 'application/json'
    }
});

export const masClient = {
    // LLM-API-Call
    async callLLM(prompt, filePath = null, customModel = null, maxRetries = 3) {
        try {
            const response = await client.post('/api/llm/call', {
                prompt,
                filePath,
                customModel,
                maxRetries
            });
            return response.data;
        } catch (error) {
            console.error('Fehler bei LLM-API-Call:', error.message);
            throw new Error(`LLM-API-Call fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // LLM-Empfehlungen für ML-Pipeline
    async getLLMRecommendations(analysis, filePath = null, selectedFeatures = null, excludedFeatures = null, userPreferences = null) {
        try {
            const response = await client.post('/api/llm/recommendations', {
                analysis,
                filePath,
                selectedFeatures,
                excludedFeatures,
                userPreferences
            });
            return response.data.recommendations;
        } catch (error) {
            console.error('Fehler bei LLM-Empfehlungen:', error.message);
            throw new Error(`LLM-Empfehlungen fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Performance-Evaluation mit LLM
    async evaluatePerformance(project) {
        try {
            const response = await client.post('/api/llm/evaluate-performance', {
                project
            });
            return response.data.evaluation;
        } catch (error) {
            console.error('Fehler bei Performance-Evaluation:', error.message);
            throw new Error(`Performance-Evaluation fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // LLM-Konfiguration abrufen
    async getLLMConfig() {
        try {
            const response = await client.get('/api/llm/config');
            return response.data.config;
        } catch (error) {
            console.error('Fehler beim Abrufen der LLM-Konfiguration:', error.message);
            throw new Error(`LLM-Konfiguration abrufen fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Ollama-Modelle abrufen
    async getAvailableOllamaModels() {
        try {
            const response = await client.get('/api/llm/ollama/models');
            return response.data;
        } catch (error) {
            console.error('Fehler beim Abrufen der Ollama-Modelle:', error.message);
            throw new Error(`Ollama-Modelle abrufen fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Ollama-Verbindung testen
    async testOllamaConnection() {
        try {
            const response = await client.post('/api/llm/ollama/test');
            return response.data;
        } catch (error) {
            console.error('Fehler beim Testen der Ollama-Verbindung:', error.message);
            throw new Error(`Ollama-Verbindungstest fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // LLM-Status abrufen
    async getLLMStatus() {
        try {
            const response = await client.get('/api/llm/status');
            return response.data;
        } catch (error) {
            console.error('Fehler beim Abrufen des LLM-Status:', error.message);
            throw new Error(`LLM-Status abrufen fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Queue-Status abrufen
    async getQueueStatus() {
        try {
            const response = await client.get('/api/llm/queue/status');
            return response.data.status;
        } catch (error) {
            console.error('Fehler beim Abrufen des Queue-Status:', error.message);
            throw new Error(`Queue-Status abrufen fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Request abbrechen
    async cancelRequest(requestId, reason = 'User cancelled') {
        try {
            const response = await client.post(`/api/llm/queue/cancel/${requestId}`, { reason });
            return response.data.success;
        } catch (error) {
            console.error('Fehler beim Abbrechen des Requests:', error.message);
            throw new Error(`Request abbrechen fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Agent-Pipeline starten
    async runAgentPipeline(project) {
        try {
            const response = await client.post('/api/agents/pipeline/run', { project });
            return response.data.result;
        } catch (error) {
            console.error('Fehler beim Starten der Agent-Pipeline:', error.message);
            throw new Error(`Agent-Pipeline starten fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Pipeline-Status abrufen
    async getPipelineStatus() {
        try {
            const response = await client.get('/api/agents/pipeline/status');
            return response.data;
        } catch (error) {
            console.error('Fehler beim Abrufen des Pipeline-Status:', error.message);
            throw new Error(`Pipeline-Status abrufen fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Alle Agents abrufen
    async getAllAgents() {
        try {
            const response = await client.get('/api/agents');
            return response.data;
        } catch (error) {
            console.error('Fehler beim Abrufen der Agents:', error.message);
            throw new Error(`Agents abrufen fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Agent-Konfiguration abrufen
    async getAgentConfig(agentKey) {
        try {
            const response = await client.get(`/api/agents/${agentKey}`);
            return response.data;
        } catch (error) {
            console.error('Fehler beim Abrufen der Agent-Konfiguration:', error.message);
            throw new Error(`Agent-Konfiguration abrufen fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Agent-Modell abrufen
    async getAgentModel(agentKey) {
        try {
            const config = await this.getAgentConfig(agentKey);
            return config.agent?.model || null;
        } catch (error) {
            console.error('Fehler beim Abrufen des Agent-Modells:', error.message);
            throw error;
        }
    },

    // Worker-Agent testen
    async testWorkerAgent(agentKey) {
        try {
            const response = await client.post(`/api/agents/worker/test/${agentKey}`);
            return response.data;
        } catch (error) {
            console.error('Fehler beim Testen des Worker-Agents:', error.message);
            throw new Error(`Worker-Agent testen fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Auto-Tuning für Modell-Optimierung
    async autoTuneModel(project, maxIterations = 2) {
        try {
            const response = await client.post('/api/agents/auto-tune', {
                project,
                maxIterations
            });
            return response.data.proposal;
        } catch (error) {
            console.error('Fehler beim Auto-Tuning:', error.message);
            throw new Error(`Auto-Tuning fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    }
};

export default masClient;

