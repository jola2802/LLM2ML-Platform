import axios from 'axios';

const UNIFIED_SERVICE_URL = process.env.UNIFIED_SERVICE_URL || 'http://localhost:3002';

const client = axios.create({
    baseURL: UNIFIED_SERVICE_URL,
    timeout: 600000, // 10 Minuten für Training
    headers: {
        'Content-Type': 'application/json'
    }
});

export const pythonClient = {
    // Data Exploration
    async exploreData(filePath) {
        try {
            const response = await client.post('/api/data/explore', { filePath });
            return response.data;
        } catch (error) {
            console.error('Fehler bei Data Exploration:', error.message);
            throw new Error(`Data Exploration fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Data Analysis (mit LLM-Zusammenfassung)
    async analyzeData(filePath, forceRefresh = false) {
        try {
            const response = await client.post('/api/data/analyze', { filePath, forceRefresh });
            return response.data;
        } catch (error) {
            console.error('Fehler bei Data Analysis:', error.message);
            throw new Error(`Data Analysis fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Training starten
    async startTraining(projectId, pythonCode) {
        try {
            const response = await client.post('/api/execution/train', {
                projectId,
                pythonCode
            });
            return response.data;
        } catch (error) {
            console.error('Fehler beim Starten des Trainings:', error.message);
            throw new Error(`Training starten fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Retraining starten
    async startRetraining(projectId, pythonCode) {
        try {
            const response = await client.post('/api/execution/retrain', {
                projectId,
                pythonCode
            });
            return response.data;
        } catch (error) {
            console.error('Fehler beim Starten des Retrainings:', error.message);
            throw new Error(`Retraining starten fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Code ausführen
    async executeCode(code, projectId = null) {
        try {
            const response = await client.post('/api/execution/execute', {
                code,
                projectId
            });
            return response.data;
        } catch (error) {
            console.error('Fehler bei Code-Ausführung:', error.message);
            throw new Error(`Code-Ausführung fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Job-Status abfragen
    async getJobStatus(jobId) {
        try {
            const response = await client.get(`/api/execution/jobs/${jobId}`);
            return response.data;
        } catch (error) {
            console.error('Fehler beim Abfragen des Job-Status:', error.message);
            throw new Error(`Job-Status abfragen fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Worker-Status abfragen
    async getWorkerStatus() {
        try {
            const response = await client.get('/api/execution/status');
            return response.data;
        } catch (error) {
            console.error('Fehler beim Abfragen des Worker-Status:', error.message);
            throw new Error(`Worker-Status abfragen fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    },

    // Prediction ausführen
    async executePrediction(project, inputFeatures) {
        try {
            const response = await client.post('/api/execution/predict', {
                project,
                inputFeatures
            });
            return response.data;
        } catch (error) {
            console.error('Fehler bei Prediction:', error.message);
            throw new Error(`Prediction fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    }
};

export default pythonClient;

