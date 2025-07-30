import { GoogleGenAI } from '@google/genai';
import path from 'path';
import { logLLMCommunication } from './log.js';

// Verfügbare Gemini-Modelle
export const GEMINI_MODELS = {
  'gemini-1.5-flash': 'gemini-1.5-flash',
  'gemini-2.0-flash': 'gemini-2.0-flash', 
  'gemini-2.5-flash': 'gemini-2.5-flash'
};

// Aktuell gewähltes Modell (Standard)
let currentModel = 'gemini-2.0-flash';

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

// LLM API Call für Script-Generierung (filePath optional)
export async function callLLMAPI(prompt, filePath = null, customModel = null) {  
  const API_KEY = process.env.GEMINI_API_KEY;
  
  const genAI = new GoogleGenAI({apiKey: API_KEY});
  
  // Verwende custom Model oder aktuell gewähltes Modell
  const modelToUse = customModel || currentModel;
  
  try {
    let contents = [{ role: "user", parts: [{ text: prompt }] }];
    
    // Log den Prompt
    await logLLMCommunication('prompt', {
      prompt,
      filePath,
      model: modelToUse,
      timestamp: new Date().toISOString()
    });

    // Nur wenn eine Datei angegeben ist, versuche sie hochzuladen
    if (filePath && typeof filePath === 'string') {
      try {
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

        // console.log(`Uploading file with MIME type: ${mimeType}`);

        const file = await genAI.files.upload({
          file: filePath,
          config: { mimeType: mimeType },
        });
      
        console.log("File with MIME type:", mimeType, "uploaded successfully:", file.uri);
        
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
          filePath
        });
        console.log("File upload failed, continuing without file:", fileError.message);
      }
    }
    
    const result = await genAI.models.generateContent({ 
      model: modelToUse,
      contents: contents,
      generationConfig: {
        temperature: 0.1
      }
    });
    
    // Log die Antwort
    await logLLMCommunication('response', {
      response: result.text,
      model: modelToUse,
      temperature: 0.1
    });

    return result.text || '';
    
  } catch (error) {
    // Log den Fehler
    await logLLMCommunication('error', {
      error: 'LLM API Error',
      message: error.message,
      prompt,
      filePath,
      model: modelToUse
    });

    console.error('LLM API Error:', error);
    // Fallback für einfache Text-Prompts ohne Datei
    if (!filePath) {
      try {
        const result = await genAI.models.generateContent({ 
          model: modelToUse,
          contents: prompt,
          generationConfig: {
            temperature: 0.1
          }
        });
        
        // Log die Fallback-Antwort
        await logLLMCommunication('response', {
          response: result.text,
          model: modelToUse,
          temperature: 0.1,
          fallback: true
        });

        return result.text || '';
      } catch (fallbackError) {
        // Log den Fallback-Fehler
        await logLLMCommunication('error', {
          error: 'Fallback LLM API Error',
          message: fallbackError.message,
          prompt
        });

        console.error('Fallback LLM API Error:', fallbackError);
        return 'LLM-Analyse nicht verfügbar';
      }
    }
    
    return 'LLM-Analyse nicht verfügbar';
  }
}

export async function testLLMAPI(prompt) {  
  const API_KEY = process.env.GEMINI_API_KEY;
  
  const genAI = new GoogleGenAI({apiKey: API_KEY});
    
  const result = await genAI.models.generateContent({ 
    model: "gemini-2.5-flash",
    contents: prompt,
    generationConfig: {
      temperature: 0.0
    }
  });

  return result.text || '';
}