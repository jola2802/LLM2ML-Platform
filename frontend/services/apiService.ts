// API-URL für Vercel Deployment
const getApiBaseUrl = () => {
  // In der Entwicklung: localhost:3001
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return 'http://localhost:3001/api';
  }
  
  // In Vercel: gleiche Domain mit /api prefix
  return '/api';
};

const API_BASE_URL = getApiBaseUrl();

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

  // Datei hochladen und intelligente Analyse
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

  // Gemini-Verbindungsstatus prüfen
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