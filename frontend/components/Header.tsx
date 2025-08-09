
import React, { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';
import LLMSettingsModal from './LLMSettingsModal';
import LLMSwitch from './LLMSwitch';

interface HeaderProps {
  title: string;
}

const Header: React.FC<HeaderProps> = ({ title }) => {
  const [llmStatus, setLlmStatus] = useState<any>(null);
  const [isLlmModalOpen, setIsLlmModalOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Lade LLM-Status
  const loadLLMStatus = async () => {
    try {
      setIsLoading(true);
      const status = await apiService.getLLMStatus();
      setLlmStatus(status);
    } catch (error) {
      console.error('Fehler beim Laden des LLM-Status:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadLLMStatus();
    
    // Aktualisiere Status alle 30 Sekunden
    const interval = setInterval(loadLLMStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  // LLM-Status-Indikator
  const getLLMStatusIndicator = () => {
    if (!llmStatus) {
      return (
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
          <span className="text-sm text-gray-400">Lade...</span>
        </div>
      );
    }

    const activeProvider = llmStatus.activeProvider;
    const isConnected = activeProvider === 'ollama' 
      ? llmStatus.ollama.connected 
      : llmStatus.gemini.connected;

    return (
      <div className="flex items-center space-x-2">
        <div className={`w-2 h-2 rounded-full ${
          isConnected ? 'bg-green-400' : 'bg-red-400'
        }`}></div>
        <span className="text-sm text-gray-300">
          {activeProvider === 'ollama' ? 'Ollama' : 'Gemini'}
        </span>
        <span className={`text-xs px-2 py-1 rounded ${
          isConnected 
            ? 'bg-green-600 text-green-200' 
            : 'bg-red-600 text-red-200'
        }`}>
          {isConnected ? 'Verbunden' : 'Nicht verbunden'}
        </span>
      </div>
    );
  };

  return (
    <>
      <header className="bg-gray-900 text-white p-4 border-b border-gray-700">
        <div className="flex justify-between items-center">
          {/* Links: Logo und Titel */}
          <div className="flex items-center space-x-4">
            <img 
              src="/idpm.png" 
              alt="IDPM Logo" 
              className="h-8 w-8 object-contain"
            />
            <h1 className="text-xl font-bold">{title}</h1>
          </div>

          {/* Rechts: LLM-Status und Einstellungen */}
          <div className="flex items-center space-x-4">
            {/* LLM-Switch mit Status */}
            <LLMSwitch onProviderChange={loadLLMStatus} />
            
            {/* LLM-Status-Anzeige */}
            {/* {getLLMStatusIndicator()} */}
            
            <button
              onClick={() => setIsLlmModalOpen(true)}
              className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-colors"
              title="LLM-Einstellungen"
            >
              <i className="fa-solid fa-gear"></i> Settings
            </button>
            
          </div>
        </div>
      </header>

      {/* LLM-Einstellungen Modal */}
      <LLMSettingsModal
        isOpen={isLlmModalOpen}
        onClose={() => setIsLlmModalOpen(false)}
        onConfigUpdated={loadLLMStatus}
      />
    </>
  );
};

export default Header;
