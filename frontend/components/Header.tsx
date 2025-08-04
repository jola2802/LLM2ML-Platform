
import React, { useState } from 'react';
import GeminiConnectionStatus from './GeminiConnectionStatus';
import GeminiSettingsModal from './GeminiSettingsModal';
import SettingsIcon from './icons/SettingsIcon';

const Header: React.FC = () => {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  const handleConnectionRefresh = () => {
    // Trigger refresh of connection status
    // This will be handled by the GeminiConnectionStatus component itself
  };

  const handleApiKeyUpdated = () => {
    // Callback when API key is updated
    handleConnectionRefresh();
  };

  return (
    <>
      <header className="bg-slate-800 shadow-lg border-b border-slate-700">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <img 
                src="/idpm.png" 
                alt="IDPM Logo" 
                className="h-8 w-8 object-contain"
              />
              <h1 className="text-2xl font-bold text-white ml-3">No-Code ML Platform</h1>
            </div>
            
            {/* Rechte Seite mit Gemini-Status und Einstellungen */}
            <div className="flex items-center space-x-4">
              {/* Gemini Verbindungsstatus */}
              <GeminiConnectionStatus onRefresh={handleConnectionRefresh} />
              
              {/* Einstellungen Button */}
              <button
                onClick={() => setIsSettingsOpen(true)}
                className="p-2 text-gray-300 hover:text-white hover:bg-slate-700 rounded-md transition-colors duration-200"
                title="Gemini AI Einstellungen"
              >
                <SettingsIcon className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Settings Modal */}
      <GeminiSettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onApiKeyUpdated={handleApiKeyUpdated}
      />
    </>
  );
};

export default Header;
