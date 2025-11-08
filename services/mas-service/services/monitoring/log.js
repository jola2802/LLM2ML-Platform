// Einfache Logging-Utility für MAS-Service
// Kann später erweitert werden für File-Logging

export async function logLLMCommunication(type, data) {
    const timestamp = new Date().toISOString();
    const logEntry = {
        timestamp,
        type,
        data: {
            ...data,
            // Entferne große Datenstrukturen für Console-Logging
            prompt: data.prompt ? (data.prompt.length > 200 ? data.prompt.substring(0, 200) + '...' : data.prompt) : undefined,
            response: data.response ? (data.response.length > 200 ? data.response.substring(0, 200) + '...' : data.response) : undefined
        }
    };

    // Console-Logging (kann später durch File-Logging ersetzt werden)
    if (type === 'prompt') {
        console.log(`[LLM] Prompt: ${logEntry.data.prompt || 'N/A'}`);
    } else if (type === 'response') {
        console.log(`[LLM] Response: ${logEntry.data.response || 'N/A'}`);
    } else if (type === 'error') {
        console.error(`[LLM] Error:`, logEntry.data.error || logEntry.data);
    }

    // TODO: Optional File-Logging implementieren
    return logEntry;
}

