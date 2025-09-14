
import React, { useState, useEffect } from 'react';
import LLMSettingsModal from './LLMSettingsModal';
import FileManagementModal from './FileManagementModal';
import LLMSwitch from './LLMSwitch';
import { llmStatusService, LLMStatus } from '../services/llmStatusService';

interface HeaderProps {
  title: string;
}

const Header: React.FC<HeaderProps> = ({ title }) => {
  const [llmStatus, setLlmStatus] = useState<LLMStatus | null>(null);
  const [isLlmModalOpen, setIsLlmModalOpen] = useState(false);
  const [isFileModalOpen, setIsFileModalOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    let unsubscribe: (() => void) | null = null;

    const initializeStatus = async () => {
      setIsLoading(true);
      
      // Status abonnieren für automatische Updates
      unsubscribe = llmStatusService.subscribe((status: LLMStatus) => {
        setLlmStatus(status);
        setIsLoading(false);
      });

      // Initial laden
      try {
        await llmStatusService.getStatus();
      } catch (error) {
        console.error('Fehler beim Laden des LLM-Status:', error);
        setIsLoading(false);
      }
    };

    initializeStatus();
    
    // Aktualisiere Status alle 10 Minuten (reduziert von 5 Minuten)
    const interval = setInterval(() => {
      llmStatusService.getStatus(true);
    }, 600000);

    return () => {
      if (unsubscribe) unsubscribe();
      clearInterval(interval);
    };
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
      : llmStatus.ollama.connected;

    return (
      <div className="flex items-center space-x-2">
        <div className={`w-2 h-2 rounded-full ${
          isConnected ? 'bg-green-400' : 'bg-red-400'
        }`}></div>
        <span className="text-sm text-gray-300">
          {activeProvider === 'ollama' ? 'Ollama' : 'Ollama'}
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
          <div>
            <button
              onClick={() => setIsFileModalOpen(true)}
              className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-colors flex items-center gap-2"
              title="Datei-Management"
            >
              <i className="fa-solid fa-folder-open"></i>📂 Files
            </button>
          </div>

          {/* Rechts: LLM-Status und Einstellungen */}
          <div className="flex items-center space-x-4">
            {/* LLM-Switch mit Status */}
            <LLMSwitch onProviderChange={() => {}} />
            
            {/* LLM-Status-Anzeige */}
            {/* {getLLMStatusIndicator()} */}
            
            <button
              onClick={() => setIsLlmModalOpen(true)}
              className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-colors"
              title="LLM-Einstellungen"
            >
              <i className="fa-solid fa-gear"></i>⚙️
            </button>                        
          </div>
        </div>
      </header>

      {/* Datei-Management Modal */}
      <FileManagementModal
        isOpen={isFileModalOpen}
        onClose={() => setIsFileModalOpen(false)}
      />

      {/* LLM-Einstellungen Modal */}
      <LLMSettingsModal
        isOpen={isLlmModalOpen}
        onClose={() => setIsLlmModalOpen(false)}
        onConfigUpdated={() => llmStatusService.getStatus(true)}
      />
    </>
  );
};

export default Header;
