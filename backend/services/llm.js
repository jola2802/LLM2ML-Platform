import { GoogleGenAI } from '@google/genai';
import path from 'path';
import { logLLMCommunication } from './log.js';
import { Ollama } from 'ollama';

// Verfügbare Gemini-Modelle
export const GEMINI_MODELS = {
  'gemini-2.0-flash': 'gemini-2.0-flash', 
  'gemini-2.0-flash-lite': 'gemini-2.0-flash-lite',
  'gemini-2.5-flash': 'gemini-2.5-flash',
  'gemini-2.5-flash-lite': 'gemini-2.5-flash-lite'
};

// Aktuell gewähltes Modell (Standard)
let currentModel = 'ollama'; // 'gemini-2.0-flash-lite'

// File-Cache für bereits hochgeladene Dateien
const fileCache = new Map();

// Modell setzen
export function setCurrentGeminiModel(model) {
  currentModel = model;
}

// Aktuelles Modell abrufen
export function getCurrentGeminiModel() {
  return currentModel;
}

// Alle verfügbaren Modelle abrufen
export function getAvailableGeminiModels() {
  return Object.keys(GEMINI_MODELS);
}

// Datei hochladen oder aus Cache abrufen
async function uploadFileOrGetFromCache(filePath, genAI) {
  // Prüfe ob Datei bereits im Cache ist
  if (fileCache.has(filePath)) {
    console.log(`Datei bereits im Cache: ${filePath}`);
    return fileCache.get(filePath);
  }

  // Parse the file path to get the file type
  const fileType = path.extname(filePath).toLowerCase().substring(1);

  let mimeType = '';
  
  // Bestimme den korrekten MIME-Type basierend auf der Dateiendung
  switch (fileType) {
    case 'csv':
      mimeType = 'text/csv';
      break;
    case 'json':
      mimeType = 'text/plain';
      break;
    case 'txt':
      mimeType = 'text/plain';
      break;
    case 'xlsx':
      mimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
      break;
    case 'xls':
      mimeType = 'application/vnd.ms-excel';
      break;
    case 'xml':
      mimeType = 'text/plain';
      break;
    case 'docx':
      mimeType = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
      break;
    case 'doc':
      mimeType = 'application/msword';
      break;
    case 'pdf':
      mimeType = 'application/pdf';
      break;
    case 'jpg':
    case 'jpeg':
      mimeType = 'image/jpeg';
      break;
    case 'png':
      mimeType = 'image/png';
      break;
    case 'gif':
      mimeType = 'image/gif';
      break;
    case 'bmp':
      mimeType = 'image/bmp';
      break;
    case 'tiff':
      mimeType = 'image/tiff';
      break;
    case 'ico':
      mimeType = 'image/x-icon';
      break;
    case 'webp':
      mimeType = 'image/webp';
      break;
    default:
      // Fallback für unbekannte Dateitypen
      mimeType = 'text/plain';
      console.log(`Unbekannter Dateityp ${fileType}, verwende text/plain als Fallback`);
  }

  // console.log(`Lade Datei hoch mit MIME-Type: ${mimeType}`);

  const file = await genAI.files.upload({
    file: filePath,
    config: { mimeType: mimeType },
  });

  console.log("Datei erfolgreich hochgeladen:", file.uri);
  
  // Speichere in Cache
  fileCache.set(filePath, file);
  
  return file;
}

// Cache leeren (für Tests oder bei Problemen)
export function clearFileCache() {
  fileCache.clear();
  console.log('File-Cache geleert');
}

// Cache-Status abrufen
export function getFileCacheStatus() {
  return {
    cachedFiles: Array.from(fileCache.keys()),
    cacheSize: fileCache.size
  };
}

// LLM API Call für Script-Generierung (filePath optional)
export async function callLLMAPI(prompt, filePath = null, customModel = null, maxRetries = 3) {  
  const API_KEY = process.env.GEMINI_API_KEY;
  
  // Bestimme das zu verwendende Modell
  const modelToUse = customModel ? customModel : 'ollama';
  
  let attempt = 0;
  
  while (attempt < maxRetries) {
    try {
      attempt++;
      console.log(`LLM API Call - Versuch ${attempt}/${maxRetries} mit Modell: ${modelToUse}`);
      
      let ollama;
      let genAI;
      
      // Initialisiere die GenAI-Instanz basierend auf dem Modell
      if (modelToUse === 'ollama') {
        ollama = new Ollama({host: 'http://127.0.0.1:11434'});
      } else if (modelToUse.includes('gemini')) {
        if (!API_KEY) {
          throw new Error('GEMINI_API_KEY nicht gesetzt');
        }
        genAI = new GoogleGenAI({apiKey: API_KEY});
      } else {
        ollama = new Ollama({host: 'http://127.0.0.1:11434'});
      }
      
      // Log den Prompt
      await logLLMCommunication('prompt', {
        prompt,
        filePath,
        model: modelToUse,
        attempt,
        timestamp: new Date().toISOString()
      });

      // Nur wenn eine Datei angegeben ist, versuche sie hochzuladen oder aus Cache zu holen
      let contents = [{ parts: [{ text: prompt }] }];
      if (filePath && typeof filePath === 'string') {
        try {
          const file = await uploadFileOrGetFromCache(filePath, genAI);
          
          // Füge die Datei zum Content hinzu
          contents[0].parts.push({
            fileData: {
              fileUri: file.uri,
              mimeType: file.mimeType
            }
          });
        } catch (fileError) {
          // Log den Fehler
          await logLLMCommunication('error', {
            error: 'File upload failed',
            message: fileError.message,
            filePath,
            attempt
          });
          console.log("File upload failed, continuing without file:", fileError.message);
        }
      }
      
      // Versuche, die Antwort mit dem angegebenen Modell zu generieren
      let response;
      try {
        if (modelToUse === 'ollama') {
          response = await ollama.chat({
            model: 'llama3.2',
            messages: [{ role: 'user', content: prompt }],
          });
          
          // Ollama Response-Format normalisieren
          response = {
            text: response.message?.content || response.content || '',
            result: response.message?.content || response.content || ''
          };
        } else {
          response = await genAI.models.generateContent({ 
            model: modelToUse,
            contents: contents,
            generationConfig: {
              temperature: 0.1
            }
          });
          
          // Gemini Response-Format normalisieren
          response = {
            text: response.response?.text() || response.text || '',
            result: response.response?.text() || response.text || ''
          };
        }
        
        // Validiere Response
        if (!response.text && !response.result) {
          throw new Error('Leere Response vom LLM erhalten');
        }
        
        // Log die erfolgreiche Antwort
        await logLLMCommunication('response', {
          response: response.text || response.result,
          model: modelToUse,
          attempt,
          temperature: 0.1
        });

        return {
          result: response.text || response.result || '',
          file_uploaded: !!filePath
        };
        
      } catch (modelError) {
        console.log(`Modell-spezifischer Fehler (Versuch ${attempt}):`, modelError.message);
        
        // Wenn das Modell nicht verfügbar ist, falle auf Ollama zurück
        if (modelError.message.includes('Model not found') || modelError.message.includes('not found')) {
          console.log(`Fallback auf Ollama (Versuch ${attempt})`);
          try {
            const ollamaResponse = await ollama.chat({
              model: 'llama3.2',
              messages: [{ role: 'user', content: prompt }],
            });
            
            const fallbackResponse = {
              text: ollamaResponse.message?.content || ollamaResponse.content || '',
              result: ollamaResponse.message?.content || ollamaResponse.content || ''
            };
            
            await logLLMCommunication('response', {
              response: fallbackResponse.text || fallbackResponse.result,
              model: 'ollama-fallback',
              attempt,
              temperature: 0.1
            });
            
            return {
              result: fallbackResponse.text || fallbackResponse.result || '',
              file_uploaded: !!filePath
            };
          } catch (fallbackError) {
            console.log(`Ollama Fallback fehlgeschlagen (Versuch ${attempt}):`, fallbackError.message);
            throw fallbackError;
          }
        } else {
          throw modelError;
        }
      }
      
    } catch (error) {
      console.error(`LLM API Fehler (Versuch ${attempt}):`, error.message);
      
      // Log den Fehler
      await logLLMCommunication('error', {
        error: 'LLM API call failed',
        message: error.message,
        model: modelToUse,
        attempt,
        timestamp: new Date().toISOString()
      });
      
      if (attempt < maxRetries) {
        // Warte kurz vor dem nächsten Versuch
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
        continue;
      } else {
        // Letzter Versuch fehlgeschlagen
        throw new Error(`LLM API call failed after ${maxRetries} attempts: ${error.message}`);
      }
    }
  }
  
  // Fallback: Sollte nie erreicht werden
  throw new Error(`LLM API call failed after ${maxRetries} attempts`);
}

export async function testLLMAPI(prompt, model) {  
  let result;
  if (model === 'ollama') {
    const ollama = new Ollama({host: 'http://127.0.0.1:11434'});
    result = await ollama.chat({
      model: 'llama3.2',
      messages: [{ role: 'user', content: prompt }],
    });
  } else if (model === 'gemini-2.5-flash-lite') {
    const API_KEY = process.env.GEMINI_API_KEY;
    const genAI = new GoogleGenAI({apiKey: API_KEY});
    result = await genAI.models.generateContent({ 
      model: "gemini-2.5-flash-lite",
      contents: prompt,
      generationConfig: {
        temperature: 0.0
      }
    });
  }
  return {
    ollama: result.message.content || '',
    gemini: result.text || ''
  };
}