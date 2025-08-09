import React, { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';

interface LLMSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfigUpdated?: () => void;
}

interface LLMConfig {
  activeProvider: 'ollama' | 'gemini';
  ollama: {
    host: string;
    defaultModel: string;
    availableModels: string[];
  };
  gemini: {
    apiKey: string | null;
    defaultModel: string;
    availableModels: string[];
  };
}

interface LLMStatus {
  success: boolean;
  activeProvider: string;
  ollama: {
    connected: boolean;
    available: boolean;
    error?: string;
    model?: string;
  };
  gemini: {
    connected: boolean;
    available: boolean;
    hasApiKey: boolean;
    error?: string;
    model?: string;
  };
  lastTested: string;
}

const LLMSettingsModal: React.FC<LLMSettingsModalProps> = ({ 
  isOpen, 
  onClose, 
  onConfigUpdated 
}) => {
  const [config, setConfig] = useState<LLMConfig | null>(null);
  const [status, setStatus] = useState<LLMStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState<{
    type: 'success' | 'error' | 'info';
    text: string;
  } | null>(null);

  // Ollama-spezifische States
  const [ollamaHost, setOllamaHost] = useState('');
  const [selectedOllamaModel, setSelectedOllamaModel] = useState('');
  const [availableOllamaModels, setAvailableOllamaModels] = useState<Array<{
    name: string;
    size: number;
    modified_at: string;
  }>>([]);

  // Gemini-spezifische States
  const [geminiApiKey, setGeminiApiKey] = useState('');
  const [selectedGeminiModel, setSelectedGeminiModel] = useState('');

  // Lade Konfiguration und Status
  const loadConfigAndStatus = async () => {
    try {
      setIsLoading(true);
      
      // Lade Konfiguration
      const configResponse = await apiService.getLLMConfig();
      console.log('Konfiguration geladen:', configResponse);
      setConfig(configResponse.config);
      
      // Setze lokale States basierend auf der geladenen Konfiguration
      if (configResponse.config) {
        setOllamaHost(configResponse.config.ollama?.host || '');
        setSelectedOllamaModel(configResponse.config.ollama?.defaultModel || '');
        setSelectedGeminiModel(configResponse.config.gemini?.defaultModel || '');
      }
      
      // Lade Status
      const statusResponse = await apiService.getLLMStatus();
      console.log('Status geladen:', statusResponse);
      setStatus(statusResponse);
      
      // Lade verfügbare Ollama-Modelle
      await loadOllamaModels();
      
    } catch (error) {
      console.error('Fehler beim Laden der Konfiguration:', error);
      setMessage({
        type: 'error',
        text: 'Fehler beim Laden der Konfiguration: ' + (error instanceof Error ? error.message : 'Unbekannter Fehler')
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Lade verfügbare Ollama-Modelle
  const loadOllamaModels = async () => {
    try {
      const response = await apiService.getOllamaModels();
      console.log('Ollama-Modelle Response:', response);
      if (response.success && response.models && Array.isArray(response.models)) {
        setAvailableOllamaModels(response.models);
        console.log(`${response.models.length} Ollama-Modelle geladen`);
      } else {
        console.log('Keine Ollama-Modelle verfügbar:', response.error);
        setAvailableOllamaModels([]);
      }
    } catch (error) {
      console.error('Fehler beim Laden der Ollama-Modelle:', error);
      setAvailableOllamaModels([]);
      // Zeige Fehler in der UI an
      setMessage({
        type: 'error',
        text: `Fehler beim Laden der Ollama-Modelle: ${error instanceof Error ? error.message : 'Unbekannter Fehler'}`
      });
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadConfigAndStatus();
      setMessage(null);
      setGeminiApiKey('');
    }
  }, [isOpen]);

  // Provider wechseln
  const handleProviderChange = async (provider: 'ollama' | 'gemini') => {
    try {
      setIsLoading(true);
      setMessage(null);
      
      console.log(`Wechsle Provider zu: ${provider}`);
      const result = await apiService.setLLMProvider(provider);
      console.log('Provider-Wechsel Ergebnis:', result);
      
      if (result.success) {
        setMessage({
          type: 'success',
          text: `Provider erfolgreich auf ${provider} gewechselt`
        });
        
        // Aktualisiere Konfiguration und Status
        await loadConfigAndStatus();
        
        if (onConfigUpdated) {
          onConfigUpdated();
        }
      } else {
        setMessage({
          type: 'error',
          text: result.error || 'Provider konnte nicht gewechselt werden'
        });
      }
    } catch (error) {
      console.error('Fehler beim Provider-Wechsel:', error);
      setMessage({
        type: 'error',
        text: `Fehler beim Provider-Wechsel: ${error instanceof Error ? error.message : 'Unbekannter Fehler'}`
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Ollama-Konfiguration speichern
  const handleOllamaConfigSave = async () => {
    try {
      setIsLoading(true);
      setMessage(null);
      
      const config: any = {};
      if (ollamaHost) config.host = ollamaHost;
      if (selectedOllamaModel) config.defaultModel = selectedOllamaModel;
      
      console.log('Speichere Ollama-Konfiguration:', config);
      const result = await apiService.updateOllamaConfig(config);
      console.log('Ollama-Konfiguration Ergebnis:', result);
      
      if (result.success) {
        setMessage({
          type: 'success',
          text: 'Ollama-Konfiguration erfolgreich gespeichert'
        });
        
        // Aktualisiere Konfiguration und lade Modelle neu
        await loadConfigAndStatus();
        
        if (onConfigUpdated) {
          onConfigUpdated();
        }
      } else {
        setMessage({
          type: 'error',
          text: result.error || 'Konfiguration konnte nicht gespeichert werden'
        });
      }
    } catch (error) {
      console.error('Fehler beim Speichern der Ollama-Konfiguration:', error);
      setMessage({
        type: 'error',
        text: `Fehler beim Speichern: ${error instanceof Error ? error.message : 'Unbekannter Fehler'}`
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Gemini-Konfiguration speichern
  const handleGeminiConfigSave = async () => {
    try {
      setIsLoading(true);
      setMessage(null);
      
      const config: any = {};
      if (geminiApiKey) config.apiKey = geminiApiKey;
      if (selectedGeminiModel) config.defaultModel = selectedGeminiModel;
      
      console.log('Speichere Gemini-Konfiguration:', { ...config, apiKey: config.apiKey ? '***' : 'nicht gesetzt' });
      const result = await apiService.updateGeminiConfig(config);
      console.log('Gemini-Konfiguration Ergebnis:', result);
      
      if (result.success) {
        setMessage({
          type: 'success',
          text: 'Gemini-Konfiguration erfolgreich gespeichert'
        });
        
        setGeminiApiKey('');
        await loadConfigAndStatus();
        
        if (onConfigUpdated) {
          onConfigUpdated();
        }
      } else {
        setMessage({
          type: 'error',
          text: result.error || 'Konfiguration konnte nicht gespeichert werden'
        });
      }
    } catch (error) {
      console.error('Fehler beim Speichern der Gemini-Konfiguration:', error);
      setMessage({
        type: 'error',
        text: `Fehler beim Speichern: ${error instanceof Error ? error.message : 'Unbekannter Fehler'}`
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Verbindung testen
  const handleTestConnection = async (provider: 'ollama' | 'gemini') => {
    try {
      setIsLoading(true);
      setMessage(null);
      
      console.log(`Teste ${provider} Verbindung...`);
      let result;
      if (provider === 'ollama') {
        result = await apiService.testOllamaConnection();
      } else {
        result = await apiService.testGeminiConnection();
      }
      
      console.log(`${provider} Test Ergebnis:`, result);
      
      if (result.success && result.connected) {
        setMessage({
          type: 'success',
          text: `${provider} Verbindung erfolgreich!`
        });
        
        // Aktualisiere Status nach erfolgreichem Test
        await loadConfigAndStatus();
      } else {
        setMessage({
          type: 'error',
          text: result.error || `${provider} Verbindung fehlgeschlagen`
        });
      }
    } catch (error) {
      console.error(`Fehler beim ${provider} Verbindungstest:`, error);
      setMessage({
        type: 'error',
        text: `${provider} Test fehlgeschlagen: ${error instanceof Error ? error.message : 'Unbekannter Fehler'}`
      });
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-full max-w-4xl mx-4 border border-gray-700 max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-white">LLM-Verwaltung</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl"
            disabled={isLoading}
          >
            ×
          </button>
        </div>

        {/* Aktueller Status */}
        {status && (
          <div className="mb-6 p-4 bg-gray-700 rounded border border-gray-600">
            <h3 className="font-semibold text-gray-200 mb-3">Aktueller Status</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Aktiver Provider:</span>
                  <span className={`px-2 py-1 rounded text-sm font-mono ${
                    status.activeProvider === 'ollama' ? 'bg-blue-600 text-blue-200' : 'bg-green-600 text-green-200'
                  }`}>
                    {status.activeProvider}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Ollama Status:</span>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded text-sm ${
                      status.ollama.connected ? 'bg-green-600 text-green-200' : 'bg-red-600 text-red-200'
                    }`}>
                      {status.ollama.connected ? 'Verbunden' : 'Nicht verbunden'}
                    </span>
                    {status.ollama.error && (
                      <span className="text-xs text-red-400" title={status.ollama.error}>
                        ⚠️
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Gemini Status:</span>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded text-sm ${
                      status.gemini.connected ? 'bg-green-600 text-green-200' : 'bg-red-600 text-red-200'
                    }`}>
                      {status.gemini.connected ? 'Verbunden' : 'Nicht verbunden'}
                    </span>
                    {status.gemini.error && (
                      <span className="text-xs text-red-400" title={status.gemini.error}>
                        ⚠️
                      </span>
                    )}
                  </div>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Ollama Modell:</span>
                  <span className="text-sm font-mono text-gray-200">{status.ollama.model || 'N/A'}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Gemini Modell:</span>
                  <span className="text-sm font-mono text-gray-200">{status.gemini.model || 'N/A'}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Letzter Test:</span>
                  <span className="text-sm text-gray-400">
                    {new Date(status.lastTested).toLocaleString('de-DE')}
                  </span>
                </div>
                {status.gemini.hasApiKey && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-300">Gemini API-Key:</span>
                    <span className="text-sm text-green-400">✓ Konfiguriert</span>
                  </div>
                )}
              </div>
            </div>
            {/* Fehler-Details */}
            {(status.ollama.error || status.gemini.error) && (
              <div className="mt-3 p-3 bg-red-900 bg-opacity-50 rounded border border-red-600">
                <h4 className="text-sm font-medium text-red-200 mb-2">Fehler-Details:</h4>
                {status.ollama.error && (
                  <p className="text-xs text-red-300 mb-1">
                    <strong>Ollama:</strong> {status.ollama.error}
                  </p>
                )}
                {status.gemini.error && (
                  <p className="text-xs text-red-300">
                    <strong>Gemini:</strong> {status.gemini.error}
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {/* Provider-Auswahl */}
        <div className="mb-6 p-4 bg-gray-700 rounded border border-gray-600">
          <h3 className="font-semibold text-gray-200 mb-3">Aktiver Provider</h3>
          <div className="flex space-x-4">
            <button
              onClick={() => handleProviderChange('ollama')}
              disabled={isLoading || status?.activeProvider === 'ollama'}
              className={`px-4 py-2 rounded font-medium transition-colors ${
                status?.activeProvider === 'ollama'
                  ? 'bg-blue-600 text-blue-200 cursor-not-allowed'
                  : 'bg-gray-600 text-gray-200 hover:bg-blue-600 hover:text-blue-200'
              }`}
            >
              Ollama
            </button>
            <button
              onClick={() => handleProviderChange('gemini')}
              disabled={isLoading || status?.activeProvider === 'gemini'}
              className={`px-4 py-2 rounded font-medium transition-colors ${
                status?.activeProvider === 'gemini'
                  ? 'bg-green-600 text-green-200 cursor-not-allowed'
                  : 'bg-gray-600 text-gray-200 hover:bg-green-600 hover:text-green-200'
              }`}
            >
              Gemini
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Ollama-Konfiguration */}
          <div className="p-4 bg-gray-700 rounded border border-gray-600">
            <h3 className="font-semibold text-gray-200 mb-4">Ollama-Konfiguration</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-200 mb-2">
                  Host URL
                </label>
                <input
                  type="text"
                  value={ollamaHost}
                  onChange={(e) => setOllamaHost(e.target.value)}
                  placeholder="http://127.0.0.1:11434"
                  className="w-full px-3 py-2 bg-gray-600 border border-gray-500 text-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={isLoading}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-200 mb-2">
                  Standard-Modell
                </label>
                {availableOllamaModels.length > 0 ? (
                  <select
                    value={selectedOllamaModel}
                    onChange={(e) => setSelectedOllamaModel(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-600 border border-gray-500 text-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    disabled={isLoading}
                  >
                    <option value="">Modell auswählen...</option>
                    {availableOllamaModels.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.name} ({Math.round((model.size || 0) / 1024 / 1024)}MB)
                      </option>
                    ))}
                  </select>
                ) : (
                  <div className="w-full px-3 py-2 bg-gray-600 border border-gray-500 text-gray-400 rounded-md">
                    Keine Modelle verfügbar - Ollama nicht verbunden oder keine Modelle installiert
                  </div>
                )}
                <p className="text-xs text-gray-400 mt-1">
                  Verfügbare Modelle: {availableOllamaModels.length}
                  {status?.ollama?.error && (
                    <span className="text-red-400 ml-2">Fehler: {status.ollama.error}</span>
                  )}
                </p>
              </div>

              <div className="flex space-x-2">
                <button
                  onClick={() => handleTestConnection('ollama')}
                  disabled={isLoading}
                  className="px-4 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  {isLoading ? 'Teste...' : 'Verbindung testen'}
                </button>
                <button
                  onClick={handleOllamaConfigSave}
                  disabled={isLoading}
                  className="px-4 py-2 bg-green-600 text-white text-sm rounded hover:bg-green-700 disabled:opacity-50"
                >
                  {isLoading ? 'Speichere...' : 'Speichern'}
                </button>
              </div>
            </div>
          </div>

          {/* Gemini-Konfiguration */}
          <div className="p-4 bg-gray-700 rounded border border-gray-600">
            <h3 className="font-semibold text-gray-200 mb-4">Gemini-Konfiguration</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-200 mb-2">
                  API-Key
                </label>
                <input
                  type="password"
                  value={geminiApiKey}
                  onChange={(e) => setGeminiApiKey(e.target.value)}
                  placeholder="Geben Sie Ihren Gemini API-Key ein..."
                  className="w-full px-3 py-2 bg-gray-600 border border-gray-500 text-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={isLoading}
                />
                {config?.gemini.apiKey && (
                  <p className="text-xs text-gray-400 mt-1">
                    Aktueller Key: {config.gemini.apiKey}
                  </p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-200 mb-2">
                  Standard-Modell
                </label>
                <select
                  value={selectedGeminiModel}
                  onChange={(e) => setSelectedGeminiModel(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-600 border border-gray-500 text-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={isLoading}
                >
                  {config?.gemini.availableModels?.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  )) || [
                    'gemini-2.5-flash-lite',
                    'gemini-2.5-flash',
                    'gemini-2.0-flash-lite',
                    'gemini-2.0-flash'
                  ].map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-gray-400 mt-1">
                  Standard-Gemini-Modelle verfügbar
                </p>
              </div>

              <div className="flex space-x-2">
                <button
                  onClick={() => handleTestConnection('gemini')}
                  disabled={isLoading}
                  className="px-4 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  {isLoading ? 'Teste...' : 'Verbindung testen'}
                </button>
                <button
                  onClick={handleGeminiConfigSave}
                  disabled={isLoading}
                  className="px-4 py-2 bg-green-600 text-white text-sm rounded hover:bg-green-700 disabled:opacity-50"
                >
                  {isLoading ? 'Speichere...' : 'Speichern'}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Nachricht anzeigen */}
        {message && (
          <div className={`mt-6 p-4 rounded border ${
            message.type === 'success' ? 'bg-green-800 text-green-200 border-green-600' :
            message.type === 'error' ? 'bg-red-800 text-red-200 border-red-600' :
            'bg-blue-800 text-blue-200 border-blue-600'
          }`}>
            {message.text}
          </div>
        )}

        {/* Schließen Button */}
        <div className="mt-6 flex justify-end">
          <button
            onClick={onClose}
            disabled={isLoading}
            className="px-6 py-2 text-gray-300 border border-gray-600 rounded hover:bg-gray-700 disabled:opacity-50 transition-colors"
          >
            Schließen
          </button>
        </div>
      </div>
    </div>
  );
};

export default LLMSettingsModal; 