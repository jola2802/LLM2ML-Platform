import { GoogleGenAI } from '@google/genai';
import path from 'path';
import { logLLMCommunication } from './log.js';
import { Ollama } from 'ollama';
import llmQueue from './llm_queue.js';

// LLM Provider Enum
export const LLM_PROVIDERS = {
  OLLAMA: 'ollama',
  GEMINI: 'gemini'
};

// Verfügbare Gemini-Modelle
export const GEMINI_MODELS = {
  'gemini-2.0-flash': 'gemini-2.0-flash', 
  'gemini-2.0-flash-lite': 'gemini-2.0-flash-lite',
  'gemini-2.5-flash': 'gemini-2.5-flash',
  'gemini-2.5-flash-lite': 'gemini-2.5-flash-lite'
};

// Gemini-Modell-Namen für automatische Provider-Erkennung
const GEMINI_MODEL_NAMES = [
  'gemini-2.0-flash',
  'gemini-2.0-flash-lite', 
  'gemini-2.5-flash',
  'gemini-2.5-flash-lite'
];

// Provider basierend auf Modell-Name bestimmen
function determineProviderFromModel(modelName) {
  if (!modelName) {
    return llmConfig.activeProvider;
  }
  
  // Prüfe ob es ein Gemini-Modell ist
  if (GEMINI_MODEL_NAMES.some(geminiModel => modelName.includes(geminiModel))) {
    return LLM_PROVIDERS.GEMINI;
  }
  
  // Fallback auf konfigurierten Provider
  return llmConfig.activeProvider;
}

// Standard-Gemini-Modelle (immer verfügbar)
export const GEMINI_DEFAULT_MODELS = [
  'gemini-2.5-flash-lite',
  'gemini-2.5-flash',
  'gemini-2.0-flash-lite',
  'gemini-2.0-flash'
];

// LLM Konfiguration
let llmConfig = {
  activeProvider: LLM_PROVIDERS.OLLAMA, // Standard: Ollama
  ollama: {
    host: process.env.OLLAMA_URL || 'http://127.0.0.1:11434',
    defaultModel: 'mistral:latest', // Standardmodell mit Tag
    availableModels: []
  },
  gemini: {
    apiKey: process.env.GEMINI_API_KEY || null,
    defaultModel: 'gemini-2.5-flash-lite',
    availableModels: GEMINI_DEFAULT_MODELS
  }
};

// Initialisiere Ollama-Modelle beim Start
async function initializeOllamaModels() {
  try {
    console.log('Initialisiere Ollama-Modelle...');
    const result = await getAvailableOllamaModels();
    if (result.success) {
      console.log(`Ollama-Modelle geladen: ${result.models.length} Modelle verfügbar`);
    } else {
      console.log('Keine Ollama-Modelle gefunden oder Ollama nicht verfügbar');
    }
  } catch (error) {
    console.error('Fehler beim Initialisieren der Ollama-Modelle:', error);
  }
}

// Starte Initialisierung
initializeOllamaModels();

// File-Cache für bereits hochgeladene Dateien
const fileCache = new Map();

// ===== KONFIGURATION FUNKTIONEN =====

// Aktuelle LLM-Konfiguration abrufen
export function getLLMConfig() {
  return {
    ...llmConfig,
    gemini: {
      ...llmConfig.gemini,
      apiKey: llmConfig.gemini.apiKey ? `${llmConfig.gemini.apiKey.substring(0, 8)}...` : null
    }
  };
}

// Aktiven Provider setzen
export function setActiveProvider(provider) {
  if (!Object.values(LLM_PROVIDERS).includes(provider)) {
    throw new Error(`Ungültiger Provider: ${provider}`);
  }
  llmConfig.activeProvider = provider;
  console.log(`Aktiver LLM-Provider auf ${provider} gesetzt`);
}

// Ollama-Konfiguration aktualisieren
export function updateOllamaConfig(config) {
  llmConfig.ollama = { ...llmConfig.ollama, ...config };
  console.log('Ollama-Konfiguration aktualisiert:', llmConfig.ollama);
}

// Gemini-Konfiguration aktualisieren
export function updateGeminiConfig(config) {
  llmConfig.gemini = { ...llmConfig.gemini, ...config };
  console.log('Gemini-Konfiguration aktualisiert:', llmConfig.gemini);
}

// ===== OLLAMA FUNKTIONEN =====

// Verfügbare Ollama-Modelle abrufen
export async function getAvailableOllamaModels() {
  try {
    console.log(`Versuche Ollama-Modelle von ${llmConfig.ollama.host} abzurufen...`);
    const ollama = new Ollama({ host: llmConfig.ollama.host });
    const response = await ollama.list();
    
    console.log('Ollama Response:', JSON.stringify(response, null, 2));
    
    if (response && response.models && Array.isArray(response.models)) {
      const models = response.models.map(model => ({
        name: model.name,
        size: model.size || 0,
        modified_at: model.modified_at || new Date().toISOString(),
        digest: model.digest || ''
      }));
      
      // Update lokale Konfiguration
      llmConfig.ollama.availableModels = models.map(m => m.name);
      
      console.log(`Ollama-Modelle erfolgreich geladen: ${models.length} Modelle`);
      
      return {
        success: true,
        models: models,
        defaultModel: llmConfig.ollama.defaultModel,
        availableModels: llmConfig.ollama.availableModels
      };
    } else {
      console.log('Keine Modelle in der Ollama-Antwort gefunden');
      return {
        success: false,
        error: 'Keine Modelle gefunden',
        models: [],
        availableModels: []
      };
    }
  } catch (error) {
    console.error('Fehler beim Abrufen der Ollama-Modelle:', error);
    return {
      success: false,
      error: error.message,
      models: [],
      availableModels: []
    };
  }
}

// Ollama-Verbindung testen
export async function testOllamaConnection() {
  try {
    const ollama = new Ollama({ host: llmConfig.ollama.host });
    const response = await ollama.chat({
      model: llmConfig.ollama.defaultModel,
      messages: [{ role: 'user', content: 'Antworte nur mit "OK" wenn du diese Nachricht erhältst.' }],
    });
    
    const content = response.message?.content || response.content || '';
    const isConnected = content.toLowerCase().includes('ok');
    
    return {
      success: true,
      connected: isConnected,
      model: llmConfig.ollama.defaultModel,
      response: content
    };
  } catch (error) {
    console.error('Ollama-Verbindungstest fehlgeschlagen:', error);
    return {
      success: false,
      connected: false,
      error: error.message
    };
  }
}

// ===== GEMINI FUNKTIONEN =====

// Gemini-Verbindung testen
export async function testGeminiConnection() {
  if (!llmConfig.gemini.apiKey) {
    return {
      success: false,
      connected: false,
      error: 'Kein API-Key konfiguriert'
    };
  }

  try {
    const genAI = new GoogleGenAI({ apiKey: llmConfig.gemini.apiKey });
    const model = 'gemini-2.5-flash-lite'
    
    const result = await genAI.models.generateContent({
      model: model,
      contents: ['Antworte nur mit "OK" wenn du diese Nachricht erhältst.']
    });
    const text = await result.text;
    
    const isConnected = text.toLowerCase().includes('ok');
    
    return {
      success: true,
      connected: isConnected,
      model: llmConfig.gemini.defaultModel,
      response: text
    };
  } catch (error) {
    console.error('Gemini-Verbindungstest fehlgeschlagen:', error);
    return {
      success: false,
      connected: false,
      error: error.message
    };
  }
}

// ===== EINHEITLICHE LLM API =====

// Neue Hauptfunktion für LLM-API-Calls mit Parallelisierung
export async function callLLMAPI(prompt, filePath = null, customModel = null, maxRetries = 3) {
  try {
    // Verwende die Queue für parallele Verarbeitung
    const result = await llmQueue.addRequest(prompt, filePath, customModel, maxRetries);
    return result;
  } catch (error) {
    console.error('LLM Queue Fehler:', error.message);
    
    // Fallback auf direkte API-Calls bei Queue-Fehlern
    console.log('Fallback auf direkte LLM API...');
    return await callLLMAPIDirect(prompt, filePath, customModel, maxRetries);
  }
}

// Direkte LLM API Calls (Fallback für Queue-Fehler)
async function callLLMAPIDirect(prompt, filePath = null, customModel = null, maxRetries = 3) {  
  const model = customModel || (llmConfig.activeProvider === LLM_PROVIDERS.OLLAMA ? llmConfig.ollama.defaultModel : llmConfig.gemini.defaultModel);
  const provider = determineProviderFromModel(model);
  
  let attempt = 0;
  
  while (attempt < maxRetries) {
    try {
      attempt++;
      console.log(`LLM API Call (Direct) - Versuch ${attempt}/${maxRetries} mit ${provider}:${model}`);
      
      // Log den Prompt
      await logLLMCommunication('prompt', {
        prompt,
        filePath,
        provider,
        model,
        attempt,
        timestamp: new Date().toISOString(),
        direct: true
      });

      // Datei-Inhalte vorbereiten (falls vorhanden)
      let contents = [{ parts: [{ text: prompt }] }];
      
      if (filePath) {
        const cachedFile = fileCache.get(filePath);
        if (cachedFile) {
          console.log(`Verwende gecachte Datei: ${filePath}`);
          contents = cachedFile;
        } else {
          console.log(`Lade Datei für LLM: ${filePath}`);
          // Datei-Upload-Logik hier implementieren falls benötigt
        }
      }

      let response;
      
      // Provider-spezifische API-Calls
      if (provider === LLM_PROVIDERS.OLLAMA) {
        const ollama = new Ollama({ host: llmConfig.ollama.host });
        const ollamaResponse = await ollama.chat({
          model: model,
          messages: [{ role: 'user', content: prompt }],
        });
        
        response = {
          text: ollamaResponse.message?.content || ollamaResponse.content || '',
          result: ollamaResponse.message?.content || ollamaResponse.content || ''
        };
      } else if (provider === LLM_PROVIDERS.GEMINI) {
        if (!llmConfig.gemini.apiKey) {
          throw new Error('GEMINI_API_KEY nicht gesetzt');
        }
        
        const genAI = new GoogleGenAI({ apiKey: llmConfig.gemini.apiKey });
        const genAIResponse = await genAI.models.generateContent({ 
          model: model,
          contents: contents,
          generationConfig: {
            temperature: 0.1
          }
        });
        
        response = {
          text: genAIResponse.response?.text() || genAIResponse.text || '',
          result: genAIResponse.response?.text() || genAIResponse.text || ''
        };
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
        temperature: 0.1,
        direct: true
      });

      return {
        result: response.text || response.result || '',
        file_uploaded: !!filePath,
        provider,
        model
      };
      
    } catch (error) {
      console.log(`LLM API Fehler (Direct, Versuch ${attempt}):`, error.message);
      
      // Bei letzten Versuch, Fehler werfen
      if (attempt >= maxRetries) {
        throw new Error(`LLM API fehlgeschlagen nach ${maxRetries} Versuchen: ${error.message}`);
      }
      
      // Kurze Pause vor nächstem Versuch
      await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
    }
  }
}

// ===== QUEUE MANAGEMENT =====

// Queue Status abrufen
export function getLLMQueueStatus() {
  return llmQueue.getStatus();
}

// Request zur Queue hinzufügen (alternative Funktion)
export async function queueLLMRequest(prompt, filePath = null, customModel = null, maxRetries = 3) {
  return await llmQueue.addRequest(prompt, filePath, customModel, maxRetries);
}

// Queue Request abbrechen
export function cancelLLMRequest(requestId, reason = 'User cancelled') {
  return llmQueue.cancelRequest(requestId, reason);
}

// ===== VERALTETE FUNKTIONEN (für Kompatibilität) =====

// Veraltete Funktionen für Rückwärtskompatibilität
export function getAvailableGeminiModels() {
  return llmConfig.gemini.availableModels;
}

export function getCurrentGeminiModel() {
  return llmConfig.gemini.defaultModel;
}

export function setCurrentGeminiModel(model) {
  llmConfig.gemini.defaultModel = model;
}

// Veraltete Test-Funktion
export async function testLLMAPI(prompt, model) {  
  const provider = model === 'ollama' ? LLM_PROVIDERS.OLLAMA : LLM_PROVIDERS.GEMINI;
  const originalProvider = llmConfig.activeProvider;
  
  try {
    setActiveProvider(provider);
    const result = await callLLMAPI(prompt, null, model);
    
    return {
      ollama: provider === LLM_PROVIDERS.OLLAMA ? result.result : '',
      gemini: provider === LLM_PROVIDERS.GEMINI ? result.result : ''
    };
  } finally {
    setActiveProvider(originalProvider);
  }
}