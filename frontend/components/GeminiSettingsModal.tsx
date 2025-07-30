import React, { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';

interface GeminiSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onApiKeyUpdated?: () => void;
}

const GeminiSettingsModal: React.FC<GeminiSettingsModalProps> = ({ 
  isOpen, 
  onClose, 
  onApiKeyUpdated 
}) => {
  const [apiKey, setApiKey] = useState('');
  const [currentKeyPreview, setCurrentKeyPreview] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [currentModel, setCurrentModel] = useState('');
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState<{
    type: 'success' | 'error' | 'info';
    text: string;
  } | null>(null);

  // Lade aktuellen API-Key-Status
  const loadCurrentKeyStatus = async () => {
    try {
      const status = await apiService.getGeminiApiKeyStatus();
      setCurrentKeyPreview(status.keyPreview || null);
    } catch (error) {
      console.error('Fehler beim Laden des API-Key-Status:', error);
    }
  };

  // Lade verfügbare Modelle und aktuelles Modell
  const loadModels = async () => {
    try {
      setIsLoadingModels(true);
      const models = await apiService.getGeminiModels();
      setAvailableModels(models.availableModels);
      setCurrentModel(models.currentModel);
      setSelectedModel(models.currentModel);
      
      // Prüfe ob das aktuelle Modell in der Liste ist, falls nicht, zeige erstes verfügbares
      if (!models.availableModels.includes(models.currentModel) && models.availableModels.length > 0) {
        setSelectedModel(models.availableModels[0]);
      }
    } catch (error) {
      console.error('Fehler beim Laden der Modelle:', error);
    } finally {
      setIsLoadingModels(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadCurrentKeyStatus();
      loadModels();
      setMessage(null);
      setApiKey('');
    }
  }, [isOpen]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!apiKey.trim()) {
      setMessage({
        type: 'error',
        text: 'Bitte geben Sie einen API-Key ein'
      });
      return;
    }

    setIsLoading(true);
    setMessage(null);

    try {
      const result = await apiService.setGeminiApiKey(apiKey.trim());
      
      if (result.success && result.connected) {
        setMessage({
          type: 'success',
          text: result.message || 'API-Key erfolgreich gesetzt und getestet!'
        });
        
        // Aktualisiere den Vorschau-Status
        await loadCurrentKeyStatus();
        
        // Benachrichtige Parent-Component
        if (onApiKeyUpdated) {
          onApiKeyUpdated();
        }
        
        // Schließe Modal nach kurzer Zeit
        setTimeout(() => {
          onClose();
        }, 1500);
      } else {
        setMessage({
          type: 'error',
          text: result.error || 'API-Key konnte nicht gesetzt werden'
        });
      }
    } catch (error) {
      setMessage({
        type: 'error',
        text: error instanceof Error ? error.message : 'Unbekannter Fehler'
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Modell wechseln
  const handleModelChange = async () => {
    if (!selectedModel.trim()) {
      setMessage({
        type: 'error',
        text: 'Bitte wählen Sie ein Modell aus'
      });
      return;
    }

    setIsLoading(true);
    setMessage(null);

    try {
      const result = await apiService.setGeminiModel(selectedModel.trim());
      
      if (result.success && result.working) {
        setMessage({
          type: 'success',
          text: result.message || 'Modell erfolgreich gewechselt!'
        });
        
        // Aktualisiere den aktuellen Modell-Status
        setCurrentModel(result.model);
        
        // Benachrichtige Parent-Component
        if (onApiKeyUpdated) {
          onApiKeyUpdated();
        }
      } else {
        setMessage({
          type: 'error',
          text: result.error || 'Modell konnte nicht gewechselt werden'
        });
      }
    } catch (error) {
      setMessage({
        type: 'error',
        text: error instanceof Error ? error.message : 'Unbekannter Fehler beim Modellwechsel'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleTestConnection = async () => {
    if (!currentKeyPreview) {
      setMessage({
        type: 'error',
        text: 'Kein API-Key gesetzt'
      });
      return;
    }

    setIsLoading(true);
    setMessage(null);

    try {
      const status = await apiService.checkGeminiStatus();
      
      if (status.connected) {
        setMessage({
          type: 'success',
          text: 'Verbindung erfolgreich!'
        });
      } else {
        setMessage({
          type: 'error',
          text: status.error || 'Verbindung fehlgeschlagen'
        });
      }
    } catch (error) {
      setMessage({
        type: 'error',
        text: error instanceof Error ? error.message : 'Test fehlgeschlagen'
      });
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-full max-w-md mx-4 border border-gray-700">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-white">Gemini AI Einstellungen</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl"
            disabled={isLoading}
          >
            ×
          </button>
        </div>

        {/* Aktueller Status */}
        {currentKeyPreview && (
          <div className="mb-4 p-3 bg-gray-700 rounded border border-gray-600">
            <h3 className="font-semibold text-gray-200 mb-2">Aktueller API-Key</h3>
            <p className="text-sm text-gray-300 font-mono">{currentKeyPreview}</p>
            <button
              onClick={handleTestConnection}
              disabled={isLoading}
              className="mt-2 px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {isLoading ? 'Teste...' : 'Verbindung testen'}
            </button>
          </div>
        )}

        {/* Modell-Auswahl */}
        <div className="mb-4 p-3 bg-gray-700 rounded border border-gray-600">
          <h3 className="font-semibold text-gray-200 mb-3">Gemini Modell</h3>
          
          {isLoadingModels ? (
            <div className="text-sm text-gray-400">Lade verfügbare Modelle...</div>
          ) : (
            <>
              {/* Aktuelles Modell anzeigen */}
              {currentModel && (
                <div className="mb-3 text-sm">
                  <span className="text-gray-300">Aktuell: </span>
                  <span className="font-mono bg-gray-600 text-gray-200 px-2 py-1 rounded border border-gray-500">{currentModel}</span>
                </div>
              )}

              {/* Modell-Auswahl */}
              <div className="space-y-3">
                {availableModels.map((model) => (
                  <label key={model} className="flex items-center space-x-3 cursor-pointer hover:bg-gray-600 p-2 rounded">
                    <input
                      type="radio"
                      name="model"
                      value={model}
                      checked={selectedModel === model}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      disabled={isLoading}
                      className="text-blue-500 focus:ring-blue-500 focus:ring-2"
                    />
                    <span className="text-sm font-mono text-gray-200">{model}</span>
                  </label>
                ))}
              </div>

              {/* Modell wechseln Button */}
              <button
                onClick={handleModelChange}
                disabled={isLoading || selectedModel === currentModel}
                className="mt-4 w-full px-4 py-2 bg-green-600 text-white text-sm rounded hover:bg-green-700 disabled:opacity-50 transition-colors"
              >
                {isLoading ? 'Wechsle...' : 'Modell wechseln'}
              </button>
            </>
          )}
        </div>

        {/* API-Key Eingabe */}
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-200 mb-2">
              Gemini API-Key
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Geben Sie Ihren Gemini API-Key ein..."
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 text-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              disabled={isLoading}
            />
            <p className="text-xs text-gray-400 mt-1">
              Sie können Ihren API-Key in der Google AI Studio Console erhalten.
            </p>
          </div>

          {/* Nachricht anzeigen */}
          {message && (
            <div className={`mb-4 p-3 rounded border ${
              message.type === 'success' ? 'bg-green-800 text-green-200 border-green-600' :
              message.type === 'error' ? 'bg-red-800 text-red-200 border-red-600' :
              'bg-blue-800 text-blue-200 border-blue-600'
            }`}>
              {message.text}
            </div>
          )}

          {/* Buttons */}
          <div className="flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              disabled={isLoading}
              className="px-4 py-2 text-gray-300 border border-gray-600 rounded hover:bg-gray-700 disabled:opacity-50 transition-colors"
            >
              Abbrechen
            </button>
            <button
              type="submit"
              disabled={isLoading || !apiKey.trim()}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {isLoading ? 'Teste...' : 'API-Key setzen'}
            </button>
          </div>
        </form>

        {/* Hilfetext */}
        <div className="mt-4 text-xs text-gray-400">
          <p><strong>Hinweis:</strong> Der API-Key wird nur für diese Session gespeichert. 
          Für eine permanente Konfiguration setzen Sie die Umgebungsvariable GEMINI_API_KEY.</p>
        </div>
      </div>
    </div>
  );
};

export default GeminiSettingsModal;