import React, { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';

interface ConnectionStatusProps {
  onRefresh?: () => void;
}

const LLMConnectionStatus: React.FC<ConnectionStatusProps> = ({ onRefresh }) => {
  const [status, setStatus] = useState<{
    connected: boolean;
    hasApiKey: boolean;
    error?: string;
    loading: boolean;
    currentModel?: string;
  }>({
    connected: false,
    hasApiKey: false,
    loading: true
  });

  const checkStatus = async () => {
    try {
      setStatus(prev => ({ ...prev, loading: true }));
      
      // Gemini Status und aktuelles Modell parallel abrufen
      const [connectionResponse, modelResponse] = await Promise.all([
        apiService.checkGeminiStatus(),
        apiService.getCurrentGeminiModel().catch(() => ({ currentModel: 'unbekannt', isCustomModel: false }))
      ]);
      
      setStatus({
        connected: connectionResponse.connected,
        hasApiKey: connectionResponse.hasApiKey,
        error: connectionResponse.error,
        currentModel: modelResponse.currentModel,
        loading: false
      });
      
      if (onRefresh) {
        onRefresh();
      }
    } catch (error) {
      setStatus({
        connected: false,
        hasApiKey: false,
        error: error instanceof Error ? error.message : 'Unbekannter Fehler',
        loading: false
      });
    }
  };

  useEffect(() => {
    checkStatus();
    
    // Prüfe Status alle 30 Sekunden
    const interval = setInterval(checkStatus, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const getStatusIndicator = () => {
    if (status.loading) {
      return (
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-amber-400 rounded-full animate-pulse"></div>
          <span className="text-sm text-slate-300">Prüfe...</span>
        </div>
      );
    }

    if (!status.hasApiKey) {
      return (
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-red-500 rounded-full"></div>
          <span className="text-sm text-red-300">Kein API-Key</span>
        </div>
      );
    }

    if (status.connected) {
      return (
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-emerald-500 rounded-full animate-pulse"></div>
          <div className="flex flex-col">
            <span className="text-sm text-emerald-300">LLM verbunden</span>
            {status.currentModel && (
              <span className="text-xs text-slate-400 font-mono">{status.currentModel}</span>
            )}
          </div>
        </div>
      );
    }

    return (
      <div className="flex items-center space-x-2">
        <div className="w-3 h-3 bg-red-500 rounded-full"></div>
        <span className="text-sm text-red-300">Verbindung fehlgeschlagen</span>
      </div>
    );
  };

  const getTooltipText = () => {
    if (status.loading) return 'Verbindungsstatus wird geprüft...';
    if (!status.hasApiKey) return 'Kein Gemini API-Key konfiguriert. Klicken Sie auf Einstellungen um einen zu setzen.';
    if (status.connected) {
      const baseText = 'Gemini AI ist verbunden und einsatzbereit';
      return status.currentModel ? `${baseText}\nModell: ${status.currentModel}` : baseText;
    }
    return status.error || 'Verbindung zu Gemini AI fehlgeschlagen';
  };

  return (
    <div 
      className="cursor-pointer" 
      title={getTooltipText()}
      onClick={checkStatus}
    >
      {getStatusIndicator()}
    </div>
  );
};

export default LLMConnectionStatus;