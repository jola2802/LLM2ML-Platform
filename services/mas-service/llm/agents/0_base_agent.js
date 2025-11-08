/**
 * Basis-Worker-Klasse für alle spezialisierten Worker-Agents
 * 
 * Diese Klasse stellt die gemeinsame Funktionalität für alle Worker-Agents bereit:
 * - LLM-API-Aufrufe
 * - Fehlerbehandlung
 * - Logging
 * - Validierung
 */

import { callLLMAPI } from '../api/llm.js';
import { getAgentConfig, getAgentModel, logAgentCall } from './config_agent_network.js';
import { BASE_WORKER_TEST_PROMPT, formatPrompt } from './prompts.js';

export class BaseWorker {
  constructor(agentKey) {
    this.agentKey = agentKey;
    this.config = getAgentConfig(agentKey);
    this.model = getAgentModel(agentKey);
  }

  /**
   * Führt den Worker-Agent aus
   * Diese Methode muss von jeder Worker-Klasse überschrieben werden
   */
  async execute(pipelineState) {
    throw new Error(`execute() Methode muss in ${this.agentKey} implementiert werden`);
  }

  /**
   * Testet die Verbindung des Worker-Agents
   */
  async test() {
    const testPrompt = formatPrompt(BASE_WORKER_TEST_PROMPT, {
      agentName: this.config.name
    });
    
    try {
      const response = await callLLMAPI(testPrompt, null, this.model, 1);
      const result = typeof response === 'string' ? response : response?.result || '';
      return result.toLowerCase().includes('ok');
    } catch (error) {
      console.error(`Test für ${this.agentKey} fehlgeschlagen:`, error.message);
      return false;
    }
  }

  /**
   * Ruft die LLM-API auf mit Agent-spezifischen Einstellungen
   */
  async callLLM(prompt, context = null, maxTokens = null) {
    logAgentCall(this.agentKey, this.model, 'LLM-Aufruf');
    
    const tokens = maxTokens || this.config.maxTokens;
    
    try {
      const response = await callLLMAPI(prompt, context, this.model, tokens);
      return response;
    } catch (error) {
      console.error(`LLM-Aufruf für ${this.agentKey} fehlgeschlagen:`, error.message);
      throw error;
    }
  }

  /**
   * Validiert das Ergebnis eines Worker-Agents
   */
  validateResult(result, expectedType = 'string') {
    if (!result) {
      throw new Error(`${this.agentKey}: Kein Ergebnis erhalten`);
    }

    if (expectedType === 'string' && typeof result !== 'string') {
      throw new Error(`${this.agentKey}: Erwarteter String, erhalten: ${typeof result}`);
    }

    if (expectedType === 'object' && typeof result !== 'object') {
      throw new Error(`${this.agentKey}: Erwartetes Objekt, erhalten: ${typeof result}`);
    }

    return true;
  }

  /**
   * Bereinigt generierten Code
   */
  cleanCode(code) {
    if (!code || typeof code !== 'string') {
      return '';
    }

    let cleaned = code
      .replace(/^\s*```[\w]*\s*/gm, '') // Markdown code blocks
      .replace(/\s*```\s*$/gm, '')
      .replace(/^\s*# Output:\s*$/gm, '') // Output-Kommentare
      .replace(/^\s*# Ausgabe:\s*$/gm, '')
      .trim();

    return cleaned;
  }

  /**
   * Extrahiert JSON aus einer LLM-Antwort
   */
  extractJSON(text) {
    try {
      // Versuche direktes JSON-Parsing
      return JSON.parse(text);
    } catch (error) {
      // Versuche JSON-Extraktion mit Regex
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        try {
          return JSON.parse(jsonMatch[0]);
        } catch (parseError) {
          console.warn(`${this.agentKey}: JSON-Extraktion fehlgeschlagen:`, parseError.message);
          return {};
        }
      }
      
      console.warn(`${this.agentKey}: Kein JSON in der Antwort gefunden`);
      return {};
    }
  }

  /**
   * Extrahiert Code aus einer LLM-Antwort
   */
  extractCode(text) {
    const block = text.match(/'start code'\n([\s\S]*?)\n'end code'/i);
    return block ? block[1].trim() : text.trim();
  }

  /**
   * Loggt Worker-Aktivitäten
   */
  log(level, message, data = null) {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] ${this.agentKey}: ${message}`;
    
    switch (level) {
      case 'info':
        console.log(`ℹ️ ${logMessage}`);
        break;
      case 'warn':
        console.warn(`⚠️ ${logMessage}`);
        break;
      case 'error':
        console.error(`❌ ${logMessage}`);
        break;
      case 'success':
        console.log(`✅ ${logMessage}`);
        break;
      default:
        console.log(logMessage);
    }
    
    if (data) {
      console.log('   Daten:', data);
    }
  }
}
