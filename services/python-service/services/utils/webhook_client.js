import axios from 'axios';

const API_GATEWAY_URL = process.env.API_GATEWAY_URL || 'http://localhost:3001';

const webhookClient = axios.create({
    baseURL: API_GATEWAY_URL,
    timeout: 10000, // 10 Sekunden Timeout für Webhooks
    headers: {
        'Content-Type': 'application/json'
    }
});

/**
 * Sendet einen Webhook an den API Gateway, wenn ein Job abgeschlossen ist
 * @param {string} jobId - ID des Jobs
 * @param {string} jobType - Typ des Jobs (training, retraining, prediction)
 * @param {string} projectId - ID des Projekts
 * @param {Object} result - Ergebnis des Jobs
 * @param {string} status - Status (completed, failed)
 */
export async function sendJobCompletionWebhook(jobId, jobType, projectId, result, status) {
    try {
        const webhookData = {
            jobId,
            jobType,
            projectId,
            result,
            status
        };

        console.log(`Sende Webhook für Job ${jobId} (${jobType}) an API Gateway...`);

        const response = await webhookClient.post('/api/webhooks/job-completed', webhookData);

        console.log(`Webhook für Job ${jobId} erfolgreich gesendet`);
        return response.data;
    } catch (error) {
        // Webhook-Fehler sollten nicht den Job-Status beeinflussen
        console.error(`Fehler beim Senden des Webhooks für Job ${jobId}:`, error.message);
        console.error(`Webhook-URL: ${API_GATEWAY_URL}/api/webhooks/job-completed`);

        // Versuche es erneut nach kurzer Verzögerung (optional)
        if (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT') {
            console.warn(`API Gateway nicht erreichbar, versuche Webhook erneut in 5 Sekunden...`);
            setTimeout(async () => {
                try {
                    await webhookClient.post('/api/webhooks/job-completed', {
                        jobId,
                        jobType,
                        projectId,
                        result,
                        status
                    });
                    console.log(`Webhook für Job ${jobId} erfolgreich nach Wiederholung gesendet`);
                } catch (retryError) {
                    console.error(`Webhook-Wiederholung für Job ${jobId} fehlgeschlagen:`, retryError.message);
                }
            }, 5000);
        }

        // Wir werfen den Fehler nicht, damit der Job trotzdem als abgeschlossen gilt
        return null;
    }
}

export default {
    sendJobCompletionWebhook
};

