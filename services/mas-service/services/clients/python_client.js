import axios from 'axios';

const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://python-service:3003';

const client = axios.create({
    baseURL: PYTHON_SERVICE_URL,
    timeout: 600000, // 10 Minuten für Training
    headers: {
        'Content-Type': 'application/json'
    }
});

export const pythonClient = {
    // Data Analysis (mit LLM-Zusammenfassung)
    async analyzeData(filePath, forceRefresh = false) {
        try {
            const response = await client.post('/api/data/analyze', { filePath, forceRefresh });
            return response.data;
        } catch (error) {
            console.error('Fehler bei Data Analysis:', error.message);
            throw new Error(`Data Analysis fehlgeschlagen: ${error.response?.data?.error || error.message}`);
        }
    }
};

export default pythonClient;

