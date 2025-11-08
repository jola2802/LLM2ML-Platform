export function setupQueueRoutes(app) {
    // LLM Queue Status abrufen
    app.get('/api/llm/queue/status', async (req, res) => {
        try {
            const { getLLMQueueStatus } = await import('../llm/api/llm.js');
            const status = getLLMQueueStatus();
            res.json({ success: true, status, timestamp: new Date().toISOString() });
        } catch (error) {
            res.status(500).json({ success: false, error: 'Failed to get queue status: ' + error.message });
        }
    });

    // LLM Request abbrechen
    app.post('/api/llm/queue/cancel/:requestId', async (req, res) => {
        try {
            const { requestId } = req.params;
            const { reason = 'User cancelled' } = req.body;
            const { cancelLLMRequest } = await import('../llm/api/llm.js');
            const cancelled = cancelLLMRequest(parseInt(requestId), reason);
            if (cancelled) {
                res.json({ success: true, message: `Request ${requestId} cancelled`, requestId: parseInt(requestId) });
            } else {
                res.status(404).json({ success: false, error: `Request ${requestId} not found or already completed` });
            }
        } catch (error) {
            res.status(500).json({ success: false, error: 'Failed to cancel request: ' + error.message });
        }
    });
}

