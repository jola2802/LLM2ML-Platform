import path from 'path';
import { logLLMCommunication } from '../monitoring/log.js';
import { Ollama } from 'ollama';
import llmQueue from './llm_queue.js';

// LLM Provider Enum
export const LLM_PROVIDERS = {
  OLLAMA: 'ollama'
};


// LLM Konfiguration
let llmConfig = {
  activeProvider: LLM_PROVIDERS.OLLAMA, // Standard: Ollama
  ollama: {
    host: process.env.OLLAMA_URL || 'http://127.0.0.1:11434',
    defaultModel: 'mistral:latest', // Standardmodell mit Tag
    availableModels: []
  }
};

// Initialisiere Ollama-Modelle beim Start
async function initializeOllamaModels() {
  try {
    // console.log('Initialisiere Ollama-Modelle...');
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
    ...llmConfig
  };
}

// Ollama-Konfiguration aktualisieren
export function updateOllamaConfig(config) {
  llmConfig.ollama = { ...llmConfig.ollama, ...config };
  console.log('Ollama-Konfiguration aktualisiert:', llmConfig.ollama);
}

// ===== OLLAMA FUNKTIONEN =====

// Verfügbare Ollama-Modelle abrufen
export async function getAvailableOllamaModels() {
  try {
    // console.log(`Versuche Ollama-Modelle von ${llmConfig.ollama.host} abzurufen...`);
    const ollama = new Ollama({ host: llmConfig.ollama.host });
    const response = await ollama.list();
    
    // console.log('Ollama Response:', JSON.stringify(response, null, 2));
    
    if (response && response.models && Array.isArray(response.models)) {
      const models = response.models.map(model => ({
        name: model.name,
        size: model.size || 0,
        modified_at: model.modified_at || new Date().toISOString(),
        digest: model.digest || ''
      }));
      
      // Update lokale Konfiguration
      llmConfig.ollama.availableModels = models.map(m => m.name);
      
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
  const model = customModel || llmConfig.ollama.defaultModel;
  const provider = LLM_PROVIDERS.OLLAMA;
  
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
      
      // Ollama API-Call
      const ollama = new Ollama({ host: llmConfig.ollama.host });
      const ollamaResponse = await ollama.chat({
        model: model,
        messages: [{ role: 'user', content: prompt }],
      });
      
      response = {
        text: ollamaResponse.message?.content || ollamaResponse.content || '',
        result: ollamaResponse.message?.content || ollamaResponse.content || ''
      };
      
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

// Alle veralteten Funktionen entfernt - nur dynamische Agent-Orchestrierung wird verwendet