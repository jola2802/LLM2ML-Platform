// API-URL für Docker Deployment
const getApiBaseUrl = () => {
  // In der Entwicklung: Nutze Vite-Proxy (läuft auf gleicher Domain mit /api prefix)
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    // Nutze den Vite-Proxy - /api wird an localhost:3001 weitergeleitet
    return '/api';
  }
  
  // In Produktion: gleiche Domain mit /api prefix (wird durch nginx proxy gehandhabt)
  return '/api';
};

const API_BASE_URL = getApiBaseUrl();

// Debug-Logging für URL
console.log('API_BASE_URL:', API_BASE_URL);
console.log('Window location:', window.location.hostname, window.location.port);

export interface ApiProject {
  id: string;
  name: string;
  status: string;
  modelType: string;
  dataSourceName: string;
  targetVariable: string;
  features: string[];
  createdAt: string;
  performanceMetrics?: {
    [key: string]: number;
  };
  pythonCode?: string;
  originalPythonCode?: string;
  modelPath?: string;
  algorithm?: string;
  hyperparameters?: { [key: string]: any };
  recommendations?: any;
}

export interface CreateProjectRequest {
  name: string;
  modelType: string;
  dataSourceName: string;
  targetVariable: string;
  features: string[];
  csvFilePath?: string;
  algorithm?: string;
  hyperparameters?: object;
}

export interface SmartCreateProjectRequest {
  name: string;
  csvFilePath: string;
  recommendations: any;
}

export interface CsvAnalysisResult {
  fileName: string;
  filePath: string;
  columns: string[];
  rowCount: number;
  dataTypes: { [key: string]: string };
  sampleData: string[][];
  llmAnalysis?: string;
  recommendations?: any;
}

class ApiService {
  // Alle Projekte abrufen
  async getProjects(): Promise<ApiProject[]> {
    const response = await fetch(`${API_BASE_URL}/projects`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  // Einzelnes Projekt abrufen
  async getProject(id: string): Promise<ApiProject> {
    const response = await fetch(`${API_BASE_URL}/projects/${id}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  // Neues Projekt erstellen (alte Methode)
  async createProject(projectData: CreateProjectRequest): Promise<ApiProject> {
    const response = await fetch(`${API_BASE_URL}/projects`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(projectData),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  // Intelligentes Projekt erstellen (mit LLM-Empfehlungen)
  async createSmartProject(projectData: SmartCreateProjectRequest): Promise<ApiProject> {
    const response = await fetch(`${API_BASE_URL}/projects/smart-create`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(projectData),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  // Python-Code bearbeiten
  async updatePythonCode(projectId: string, pythonCode: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/projects/${projectId}/code`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ pythonCode }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
  }

  // Python-Code und Hyperparameter für ein Projekt aktualisieren
  async updateProjectCodeAndHyperparameters(projectId: string, pythonCode: string, hyperparameters: { [key: string]: any }): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/projects/${projectId}/code`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ pythonCode, hyperparameters }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
  }

  // Projekt re-trainieren
  async retrainProject(projectId: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/projects/${projectId}/retrain`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
  }

  // Neue Performance-Evaluation Funktionen
  async evaluatePerformance(projectId: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/projects/${projectId}/evaluate-performance`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  // Projekt löschen
  async deleteProject(id: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/projects/${id}`, {
      method: 'DELETE',
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
  }

  // Datei hochladen und Basis-Analyse (ohne LLM)
  async uploadFile(file: File): Promise<CsvAnalysisResult> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  // LLM-Empfehlungen für manipulierte Daten
  async analyzeData(filePath: string, excludedColumns?: string[], excludedFeatures?: string[]): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/analyze-data`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        filePath,
        excludedColumns,
        excludedFeatures
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  // Prediction mit echtem API-Endpoint
  async predict(projectId: string, features: { [key: string]: string | string }): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/predict/${projectId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ features }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  // Modell herunterladen
  async downloadModel(id: string, projectName: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/projects/${id}/download`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    // Blob aus Response erstellen
    const blob = await response.blob();
    
    // Download auslösen
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${projectName}_model.pkl`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }

  // Erweiterte Datenstatistiken abrufen
  async getDataStatistics(projectId: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/projects/${projectId}/data-statistics`);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  // ===== NEUE LLM API METHODEN =====

  // LLM-Konfiguration abrufen
  async getLLMConfig(): Promise<{success: boolean, config: any}> {
    try {
      const response = await fetch(`${API_BASE_URL}/llm/config`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      return response.json();
    } catch (error) {
      console.error('Fehler beim Abrufen der LLM-Konfiguration:', error);
      throw error;
    }
  }

  // LLM-Status abrufen
  async getLLMStatus(): Promise<any> {
    try {
      const response = await fetch(`${API_BASE_URL}/llm/status`);
      
      // Prüfe Content-Type um sicherzustellen, dass wir JSON bekommen
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        console.warn('Server antwortet nicht mit JSON:', contentType);
        throw new Error(`Server antwortet nicht mit JSON (Content-Type: ${contentType})`);
      }
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      
      return response.json();
    } catch (error) {
      console.error('Fehler beim Abrufen des LLM-Status:', error);
      
      // Fallback: Standard-Status zurückgeben
      const errorMessage = error instanceof Error ? error.message : 'Unbekannter Fehler';
      return {
        success: false,
        activeProvider: 'ollama',
        ollama: {
          connected: false,
          available: false,
          error: errorMessage,
          model: 'mistral:latest'
        },
        gemini: {
          connected: false,
          available: false,
          hasApiKey: false,
          error: errorMessage,
          model: 'gemini-2.5-flash-lite'
        },
        lastTested: new Date().toISOString()
      };
    }
  }

  // Aktiven Provider setzen
  async setLLMProvider(provider: string): Promise<{success: boolean, message?: string, error?: string}> {
    try {
      const response = await fetch(`${API_BASE_URL}/llm/provider`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ provider }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      return response.json();
    } catch (error) {
      console.error('Fehler beim Setzen des LLM-Providers:', error);
      throw error;
    }
  }

  // Ollama-Modelle abrufen
  async getOllamaModels(): Promise<{success: boolean, models: any[], error?: string}> {
    try {
      const response = await fetch(`${API_BASE_URL}/llm/ollama/models`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      return response.json();
    } catch (error) {
      console.error('Fehler beim Abrufen der Ollama-Modelle:', error);
      throw error;
    }
  }

  // Ollama-Verbindung testen
  async testOllamaConnection(): Promise<{success: boolean, connected: boolean, error?: string}> {
    try {
      const response = await fetch(`${API_BASE_URL}/llm/ollama/test`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      return response.json();
    } catch (error) {
      console.error('Fehler beim Testen der Ollama-Verbindung:', error);
      throw error;
    }
  }

  // Ollama-Konfiguration aktualisieren
  async updateOllamaConfig(config: any): Promise<{success: boolean, message?: string, error?: string}> {
    try {
      const response = await fetch(`${API_BASE_URL}/llm/ollama/config`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      return response.json();
    } catch (error) {
      console.error('Fehler beim Aktualisieren der Ollama-Konfiguration:', error);
      throw error;
    }
  }

  // Gemini-Verbindung testen
  async testGeminiConnection(): Promise<{success: boolean, connected: boolean, error?: string}> {
    try {
      const response = await fetch(`${API_BASE_URL}/llm/gemini/test`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      return response.json();
    } catch (error) {
      console.error('Fehler beim Testen der Gemini-Verbindung:', error);
      throw error;
    }
  }

  // Gemini-Konfiguration aktualisieren
  async updateGeminiConfig(config: any): Promise<{success: boolean, message?: string, error?: string}> {
    try {
      const response = await fetch(`${API_BASE_URL}/llm/gemini/config`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      return response.json();
    } catch (error) {
      console.error('Fehler beim Aktualisieren der Gemini-Konfiguration:', error);
      throw error;
    }
  }

  // ===== VERALTETE METHODEN (für Kompatibilität) =====

  // Gemini-Verbindungsstatus prüfen (veraltet)
  async checkGeminiStatus(): Promise<{connected: boolean, hasApiKey: boolean, error?: string, lastTested?: string}> {
    const response = await fetch(`${API_BASE_URL}/gemini/status`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  // API-Key setzen
  async setGeminiApiKey(apiKey: string): Promise<{success: boolean, connected: boolean, message?: string, error?: string}> {
    const response = await fetch(`${API_BASE_URL}/gemini/api-key`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ apiKey }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  // API-Key-Status abrufen
  async getGeminiApiKeyStatus(): Promise<{hasApiKey: boolean, keyPreview?: string}> {
    const response = await fetch(`${API_BASE_URL}/gemini/api-key-status`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  // Verfügbare Gemini-Modelle abrufen
  async getGeminiModels(): Promise<{availableModels: string[], currentModel: string, customModelSupported: boolean}> {
    const response = await fetch(`${API_BASE_URL}/gemini/models`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  // Gemini-Modell setzen
  async setGeminiModel(model: string): Promise<{success: boolean, model: string, isCustomModel: boolean, tested: boolean, working: boolean, message?: string, error?: string}> {
    const response = await fetch(`${API_BASE_URL}/gemini/model`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ model }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  // Aktuelles Gemini-Modell abrufen
  async getCurrentGeminiModel(): Promise<{currentModel: string, isCustomModel: boolean}> {
    const response = await fetch(`${API_BASE_URL}/gemini/current-model`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }
}

export const apiService = new ApiService(); 