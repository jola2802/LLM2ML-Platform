import { apiService } from './apiService';

export interface LLMStatus {
  success: boolean;
  activeProvider: string;
  ollama: {
    connected: boolean;
    available: boolean;
    error?: string;
    model: string;
  };
  gemini: {
    connected: boolean;
    available: boolean;
    hasApiKey: boolean;
    error?: string;
    model: string;
  };
  lastTested: string;
}

class LLMStatusService {
  private status: LLMStatus | null = null;
  private lastFetch: number = 0;
  private cacheDuration: number = 60000; // 1 Minute Cache
  private isLoading: boolean = false;
  private listeners: Set<(status: LLMStatus) => void> = new Set();

  // Status mit Caching abrufen
  async getStatus(forceRefresh: boolean = false): Promise<LLMStatus> {
    const now = Date.now();
    
    // Verwende Cache, wenn verfügbar und nicht abgelaufen
    if (!forceRefresh && this.status && (now - this.lastFetch) < this.cacheDuration) {
      return this.status;
    }

    // Verhindere mehrfache gleichzeitige Requests
    if (this.isLoading && !forceRefresh) {
      // Warte auf laufenden Request
      return new Promise((resolve) => {
        const checkStatus = () => {
          if (!this.isLoading && this.status) {
            resolve(this.status);
          } else {
            setTimeout(checkStatus, 100);
          }
        };
        checkStatus();
      });
    }

    try {
      this.isLoading = true;
      console.log('📡 LLM Status wird abgerufen...');
      
      const newStatus = await apiService.getLLMStatus();
      this.status = newStatus;
      this.lastFetch = now;
      
      // Benachrichtige alle Listener
      this.notifyListeners();
      
      return newStatus;
    } catch (error) {
      console.error('Fehler beim Abrufen des LLM-Status:', error);
      
      // Fallback-Status
      const fallbackStatus: LLMStatus = {
        success: false,
        activeProvider: 'ollama',
        ollama: {
          connected: false,
          available: false,
          error: error instanceof Error ? error.message : 'Unbekannter Fehler',
          model: 'mistral:latest'
        },
        gemini: {
          connected: false,
          available: false,
          hasApiKey: false,
          error: error instanceof Error ? error.message : 'Unbekannter Fehler',
          model: 'gemini-2.5-flash-lite'
        },
        lastTested: new Date().toISOString()
      };
      
      this.status = fallbackStatus;
      this.lastFetch = now;
      
      return fallbackStatus;
    } finally {
      this.isLoading = false;
    }
  }

  // Listener für Status-Updates registrieren
  subscribe(callback: (status: LLMStatus) => void): () => void {
    this.listeners.add(callback);
    
    // Sofort aktuellen Status senden, falls verfügbar
    if (this.status) {
      callback(this.status);
    }
    
    // Return Unsubscribe-Funktion
    return () => {
      this.listeners.delete(callback);
    };
  }

  // Alle Listener benachrichtigen
  private notifyListeners(): void {
    if (this.status) {
      this.listeners.forEach(callback => {
        try {
          callback(this.status!);
        } catch (error) {
          console.error('Fehler beim Benachrichtigen des LLM-Status-Listeners:', error);
        }
      });
    }
  }

  // Cache invalidieren
  invalidateCache(): void {
    this.status = null;
    this.lastFetch = 0;
  }

  // Provider wechseln und Status aktualisieren
  async setProvider(provider: string): Promise<{success: boolean, message?: string, error?: string}> {
    try {
      const result = await apiService.setLLMProvider(provider);
      
      if (result.success) {
        // Status sofort aktualisieren nach Provider-Wechsel
        await this.getStatus(true);
      }
      
      return result;
    } catch (error) {
      throw error;
    }
  }

  // Aktueller Status (ohne API-Call)
  getCurrentStatus(): LLMStatus | null {
    return this.status;
  }

  // Prüfen ob Cache gültig ist
  isCacheValid(): boolean {
    return this.status !== null && (Date.now() - this.lastFetch) < this.cacheDuration;
  }
}

// Singleton-Instance exportieren
export const llmStatusService = new LLMStatusService();
