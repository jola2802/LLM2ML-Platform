import { parentPort } from 'worker_threads';
import { logLLMCommunication } from '../monitoring/log.js';
import { ChatOllama } from '@langchain/community/chat_models/ollama';

// LLM Provider Enum
const LLM_PROVIDERS = {
  OLLAMA: 'ollama'
};


// Provider basierend auf Modell-Name bestimmen
function determineProvider(modelName) {
  // Für lokale Implementierung immer Ollama verwenden
  return LLM_PROVIDERS.OLLAMA;
}

// Worker State
let isShuttingDown = false;
let currentRequestId = null;

// LLM Konfiguration (aus Environment Variables)
const llmConfig = {
  activeProvider: LLM_PROVIDERS.OLLAMA,
  ollama: {
    host: process.env.OLLAMA_URL || 'http://127.0.0.1:11434',
    defaultModel: 'mistral:latest'
  }
};

// Worker Message Handler
parentPort?.on('message', async (message) => {
  if (isShuttingDown) return;
  
  const { type, requestId } = message;
  
  try {
    switch (type) {
      case 'processRequest':
        await handleProcessRequest(message);
        break;
      case 'cancelRequest':
        handleCancelRequest(requestId);
        break;
      case 'shutdown':
        handleShutdown();
        break;
      default:
        console.warn('Unbekannter Message Type:', type);
    }
  } catch (error) {
    sendError(requestId, error.message);
  }
});

// LLM Request verarbeiten
async function handleProcessRequest({ requestId, prompt, filePath, customModel, maxRetries }) {
  currentRequestId = requestId;
  
  try {
    sendProgress(requestId, { status: 'started', message: 'LLM Request gestartet' });
    
    // Provider und Model bestimmen - nur Ollama
    const model = customModel || llmConfig.ollama.defaultModel;
    const provider = LLM_PROVIDERS.OLLAMA;
    
    let attempt = 0;
    let lastError = null;
    
    while (attempt < maxRetries && !isShuttingDown) {
      attempt++;
      
      try {
        sendProgress(requestId, { 
          status: 'processing', 
          message: `Versuch ${attempt}/${maxRetries} mit ${provider}:${model}`,
          attempt,
          provider,
          model
        });
        
        // Log den Prompt
        await logLLMCommunication('prompt', {
          prompt,
          filePath,
          provider,
          model,
          attempt,
          timestamp: new Date().toISOString(),
          workerId: process.pid
        });

        let response;
        
        // Ollama API-Call
        response = await callOllamaAPI(model, prompt);
        
        // Validiere Response
        if (!response.text && !response.result) {
          throw new Error('Leere Response vom LLM erhalten');
        }
        
        // Log die erfolgreiche Antwort
        await logLLMCommunication('response', {
          response: response.text || response.result,
          provider,
          model,
          attempt,
          success: true,
          workerId: process.pid
        });

        const result = {
          result: response.text || response.result || '',
          file_uploaded: !!filePath,
          provider,
          model,
          attempts: attempt,
          workerId: process.pid
        };
        
        sendResult(requestId, result);
        return;
        
      } catch (error) {
        lastError = error;
        console.log(`LLM API Fehler (Versuch ${attempt}):`, error.message);
        
        sendProgress(requestId, {
          status: 'retry',
          message: `Fehler bei Versuch ${attempt}: ${error.message}`,
          attempt,
          error: error.message
        });
        
        // Log den Fehler
        await logLLMCommunication('error', {
          error: error.message,
          provider,
          model,
          attempt,
          workerId: process.pid
        });
        
        // Bei letzten Versuch, Fehler werfen
        if (attempt >= maxRetries) {
          throw new Error(`LLM API fehlgeschlagen nach ${maxRetries} Versuchen: ${error.message}`);
        }
        
        // Kurze Pause vor nächstem Versuch (nur wenn nicht shutting down)
        if (!isShuttingDown) {
          await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
        }
      }
    }
    
  } catch (error) {
    sendError(requestId, error.message);
  } finally {
    currentRequestId = null;
  }
}

// Ollama API Call via LangChain
async function callOllamaAPI(model, prompt) {
  const chat = new ChatOllama({ baseUrl: llmConfig.ollama.host, model });
  const aiMessage = await chat.invoke(prompt);
  const text = extractAIMessageText(aiMessage);
  return { text, result: text };
}


// Hilfsfunktion: AIMessage zu String extrahieren
function extractAIMessageText(aiMessage) {
  if (!aiMessage) return '';
  const { content } = aiMessage;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    try {
      return content
        .map(part => {
          if (typeof part === 'string') return part;
          if (part && typeof part === 'object') {
            if ('text' in part && typeof part.text === 'string') return part.text;
            if ('value' in part && typeof part.value === 'string') return part.value;
            return JSON.stringify(part);
          }
          return String(part);
        })
        .join(' ');
    } catch {
      return String(content);
    }
  }
  try {
    return String(content);
  } catch {
    return '';
  }
}

// Request abbrechen
function handleCancelRequest(requestId) {
  if (currentRequestId === requestId) {
    console.log(`Cancelling request ${requestId}`);
    // Request als cancelled markieren
    currentRequestId = null;
    sendError(requestId, 'Request cancelled');
  }
}

// Graceful Shutdown
function handleShutdown() {
  console.log('Worker shutdown initiated');
  isShuttingDown = true;
  
  if (currentRequestId) {
    sendError(currentRequestId, 'Worker shutting down');
  }
  
  // Worker beenden
  process.exit(0);
}

// Helper Functions für Parent Communication
function sendResult(requestId, data) {
  parentPort?.postMessage({
    type: 'result',
    requestId,
    data
  });
}

function sendError(requestId, error) {
  parentPort?.postMessage({
    type: 'error',
    requestId,
    error
  });
}

function sendProgress(requestId, data) {
  parentPort?.postMessage({
    type: 'progress',
    requestId,
    data
  });
}

// Unhandled Errors
process.on('uncaughtException', (error) => {
  console.error('Worker uncaught exception:', error);
  if (currentRequestId) {
    sendError(currentRequestId, `Worker error: ${error.message}`);
  }
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Worker unhandled rejection:', reason);
  if (currentRequestId) {
    sendError(currentRequestId, `Worker rejection: ${reason}`);
  }
});

console.log(`LLM Worker ${process.pid} gestartet`);