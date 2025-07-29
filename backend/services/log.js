import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Logging-Verzeichnis erstellen
const logsDir = path.join(__dirname, '..', 'logs');

// Initialisierung des Log-Verzeichnisses
export async function initializeLogging() {
  try {
    await fs.mkdir(logsDir, { recursive: true });
  } catch (err) {
    console.log('Logs directory already exists or error creating it:', err.message);
  }
}

// Hilfsfunktion zum Loggen der LLM-Kommunikation
export async function logLLMCommunication(type, data) {
  const timestamp = new Date().toISOString();
  const logEntry = {
    timestamp,
    type,
    data
  };

  // Log-Dateiname mit Datum
  const date = new Date().toISOString().split('T')[0];
  const logFile = path.join(logsDir, `llm_communication_${date}.log`);

  // Farbige Konsolenausgabe
  const colors = {
    prompt: '\x1b[36m', // Cyan für Prompts
    response: '\x1b[32m', // Grün für Antworten
    error: '\x1b[31m', // Rot für Fehler
    reset: '\x1b[0m'
  };

  // Konsolenausgabe
  // console.log(`${colors[type]}[LLM ${type.toUpperCase()}]${colors.reset}`);
  // console.log(JSON.stringify(data, null, 2));
  // console.log('-'.repeat(80));

  try {
    // An Log-Datei anhängen
    await fs.appendFile(
      logFile,
      JSON.stringify(logEntry, null, 2) + ',\n',
      'utf-8'
    );
  } catch (error) {
    console.error('Error writing to log file:', error);
  }
}
