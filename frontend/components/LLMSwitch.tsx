import React, { useState, useEffect } from 'react';
import { llmStatusService, LLMStatus } from '../services/llmStatusService';
import { apiService } from '../services/apiService';

interface LLMSwitchProps {
  onProviderChange?: (provider: string) => void;
  className?: string;
}

const LLMSwitch: React.FC<LLMSwitchProps> = ({ onProviderChange, className = '' }) => {
  const [activeProvider, setActiveProvider] = useState<string>('ollama');
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<LLMStatus | null>(null);

  useEffect(() => {
    // Status abonnieren für automatische Updates
    const unsubscribe = llmStatusService.subscribe((newStatus: LLMStatus) => {
      setStatus(newStatus);
      if (newStatus.activeProvider) {
        setActiveProvider(newStatus.activeProvider);
      }
    });

    // Initial laden (verwendet Cache falls verfügbar)
    llmStatusService.getStatus().then((initialStatus) => {
      setStatus(initialStatus);
      if (initialStatus.activeProvider) {
        setActiveProvider(initialStatus.activeProvider);
      }
    }).catch((error) => {
      console.error('Fehler beim Laden des LLM-Status:', error);
    });

    return unsubscribe;
  }, []);

  // Provider wechseln
  const handleProviderChange = async (provider: string) => {
    if (provider === activeProvider || isLoading) return;

    try {
      setIsLoading(true);
      const result = await llmStatusService.setProvider(provider);
      
      if (result.success) {
        setActiveProvider(provider);
        
        if (onProviderChange) {
          onProviderChange(provider);
        }
      }
    } catch (error) {
      console.error('Fehler beim Wechseln des Providers:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Status-Indikator
  const getConnectionStatus = () => {
    if (!status) return { connected: false, color: 'gray' };
    
    // Sicherheitsprüfung für Status-Struktur
    const ollamaStatus = status.ollama || { connected: false, available: false };

    const isConnected = ollamaStatus.connected;
    
    return {
      connected: isConnected,
      color: isConnected ? 'green' : 'red'
    };
  };

  const connectionStatus = getConnectionStatus();

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      {/* Status-Indikator */}
      <div className="flex items-center space-x-1">
        <div className={`w-3 h-3 rounded-full ${
          connectionStatus.connected ? 'bg-green-400' : 'bg-red-400'
        }`}></div>
        <span className="text-xs text-gray-400">
          {connectionStatus.connected ? 'Verbunden' : 'Getrennt'}
        </span>
      </div>
      
      {/* Provider-Buttons */}
      <div className="flex bg-gray-700 rounded-lg p-1">
        <button
          onClick={() => handleProviderChange('ollama')}
          disabled={isLoading}
          className={`px-3 py-1 text-sm rounded-md transition-colors ${
            activeProvider === 'ollama'
              ? 'bg-blue-600 text-blue-200'
              : 'text-gray-300 hover:text-white hover:bg-gray-600'
          } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          Ollama
        </button>
      </div>
      
      {/* Loading-Indikator */}
      {isLoading && (
        <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
      )}
    </div>
  );
};

export default LLMSwitch;