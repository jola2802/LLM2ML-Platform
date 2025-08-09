import { parentPort } from 'worker_threads';
import { GoogleGenAI } from '@google/genai';
import { Ollama } from 'ollama';
import { logLLMCommunication } from './log.js';

// LLM Provider Enum
const LLM_PROVIDERS = {
  OLLAMA: 'ollama',
  GEMINI: 'gemini'
};

// Gemini-Modelle (zur automatischen Provider-Erkennung)
const GEMINI_MODELS = [
  'gemini-2.0-flash',
  'gemini-2.0-flash-lite', 
  'gemini-2.5-flash',
  'gemini-2.5-flash-lite'
];

// Provider basierend auf Modell-Name bestimmen
function determineProvider(modelName) {
  if (!modelName) {
    return llmConfig.activeProvider;
  }
  
  // Prüfe ob es ein Gemini-Modell ist
  if (GEMINI_MODELS.some(geminiModel => modelName.includes(geminiModel))) {
    return LLM_PROVIDERS.GEMINI;
  }
  
  // Fallback auf konfigurierten Provider
  return llmConfig.activeProvider;
}

// Worker State
let isShuttingDown = false;
let currentRequestId = null;

// LLM Konfiguration (aus Environment Variables)
const llmConfig = {
  activeProvider: process.env.LLM_PROVIDER || LLM_PROVIDERS.OLLAMA,
  ollama: {
    host: process.env.OLLAMA_URL || 'http://127.0.0.1:11434',
    defaultModel: 'mistral:latest'
  },
  gemini: {
    apiKey: process.env.GEMINI_API_KEY || null,
    defaultModel: 'gemini-2.5-flash-lite'
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
    
    // Provider und Model bestimmen - automatische Provider-Erkennung basierend auf Modell
    const model = customModel || (llmConfig.activeProvider === LLM_PROVIDERS.OLLAMA ? llmConfig.ollama.defaultModel : llmConfig.gemini.defaultModel);
    const provider = determineProvider(model);
    
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
        
        // Provider-spezifische API-Calls
        if (provider === LLM_PROVIDERS.OLLAMA) {
          response = await callOllamaAPI(model, prompt);
        } else if (provider === LLM_PROVIDERS.GEMINI) {
          response = await callGeminiAPI(model, prompt);
        } else {
          throw new Error(`Unbekannter Provider: ${provider}`);
        }
        
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

// Ollama API Call
async function callOllamaAPI(model, prompt) {
  const ollama = new Ollama({ host: llmConfig.ollama.host });
  
  const ollamaResponse = await ollama.chat({
    model: model,
    messages: [{ role: 'user', content: prompt }],
  });
  
  return {
    text: ollamaResponse.message?.content || ollamaResponse.content || '',
    result: ollamaResponse.message?.content || ollamaResponse.content || ''
  };
}

// Gemini API Call
async function callGeminiAPI(model, prompt) {
  if (!llmConfig.gemini.apiKey) {
    throw new Error('GEMINI_API_KEY nicht gesetzt');
  }
  
  const genAI = new GoogleGenAI({ apiKey: llmConfig.gemini.apiKey });
  const contents = [{ parts: [{ text: prompt }] }];
  
  const genAIResponse = await genAI.models.generateContent({ 
    model: model,
    contents: contents,
    generationConfig: {
      temperature: 0.1
    }
  });
  
  return {
    text: genAIResponse.response?.text() || genAIResponse.text || '',
    result: genAIResponse.response?.text() || genAIResponse.text || ''
  };
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