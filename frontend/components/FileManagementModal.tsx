import React, { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';
import { TrashIcon } from './icons/TrashIcon';

interface FileInfo {
  name: string;
  size: number;
  lastModified: string;
  type: 'script' | 'model' | 'upload';
  path: string;
}

interface FileManagementModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const FileManagementModal: React.FC<FileManagementModalProps> = ({ isOpen, onClose }) => {
  const [files, setFiles] = useState<{
    scripts: FileInfo[];
    models: FileInfo[];
    uploads: FileInfo[];
  }>({
    scripts: [],
    models: [],
    uploads: []
  });
  const [activeTab, setActiveTab] = useState<'scripts' | 'models' | 'uploads'>('uploads');
  const [isLoading, setIsLoading] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  const loadFiles = async () => {
    setIsLoading(true);
    try {
      const [scriptsResponse, modelsResponse, uploadsResponse] = await Promise.all([
        apiService.getFiles('scripts'),
        apiService.getFiles('models'),
        apiService.getFiles('uploads')
      ]);

      setFiles({
        scripts: scriptsResponse,
        models: modelsResponse,
        uploads: uploadsResponse
      });
    } catch (error) {
      console.error('Fehler beim Laden der Dateien:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadFiles();
    }
  }, [isOpen]);

  const handleDeleteFile = async (filePath: string, fileType: 'scripts' | 'models' | 'uploads') => {
    try {
      await apiService.deleteFile(filePath, fileType);
      await loadFiles(); // Neu laden nach Löschung
      setDeleteConfirm(null);
    } catch (error) {
      console.error('Fehler beim Löschen der Datei:', error);
      alert('Fehler beim Löschen der Datei: ' + (error instanceof Error ? error.message : 'Unbekannter Fehler'));
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleString('de-DE');
  };

  const getTabCount = (tab: 'scripts' | 'models' | 'uploads'): number => {
    return files[tab].length;
  };

  const getCurrentFiles = (): FileInfo[] => {
    return files[activeTab];
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-4xl max-h-[80vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-600">
          <div>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Datei-Management
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Verwalte Scripts, Modelle und Upload-Dateien
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200 dark:border-gray-600">
          <nav className="flex space-x-8 px-6">
            {[
              { key: 'uploads' as const, label: 'Uploads', icon: '📁' },
              { key: 'scripts' as const, label: 'Scripts', icon: '📄' },
              { key: 'models' as const, label: 'Modelle', icon: '🤖' }
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.key
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
                }`}
              >
                <span className="flex items-center space-x-2">
                  <span>{tab.icon}</span>
                  <span>{tab.label}</span>
                  <span className="bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 px-2 py-1 rounded-full text-xs">
                    {getTabCount(tab.key)}
                  </span>
                </span>
              </button>
            ))}
          </nav>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto" style={{ maxHeight: 'calc(80vh - 200px)' }}>
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
              <span className="ml-3 text-gray-600 dark:text-gray-400">Lade Dateien...</span>
            </div>
          ) : getCurrentFiles().length === 0 ? (
            <div className="text-center py-12">
              <div className="text-4xl mb-4">📂</div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                Keine Dateien gefunden
              </h3>
              <p className="text-gray-600 dark:text-gray-400">
                Es wurden keine {activeTab === 'uploads' ? 'Upload-Dateien' : activeTab === 'scripts' ? 'Scripts' : 'Modelle'} gefunden.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {getCurrentFiles().map((file, index) => (
                <div
                  key={`${file.path}-${index}`}
                  className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-3">
                      <div className="text-2xl">
                        {activeTab === 'uploads' && '📄'}
                        {activeTab === 'scripts' && '</>'}
                        {activeTab === 'models' && '🤖'}
                      </div>
                      <div className="min-w-0 flex-1">
                        <h4 className="text-sm font-medium text-gray-900 dark:text-white truncate">
                          {file.name}
                        </h4>
                        <div className="text-xs text-gray-500 dark:text-gray-400 space-x-4">
                          <span>{formatFileSize(file.size)}</span>
                          <span>•</span>
                          <span>Geändert: {formatDate(file.lastModified)}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {deleteConfirm === file.path ? (
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-red-600 dark:text-red-400">Sicher?</span>
                        <button
                          onClick={() => handleDeleteFile(file.path, activeTab)}
                          className="px-3 py-1 bg-red-600 text-white text-xs rounded hover:bg-red-700 transition-colors"
                        >
                          Ja
                        </button>
                        <button
                          onClick={() => setDeleteConfirm(null)}
                          className="px-3 py-1 bg-gray-500 text-white text-xs rounded hover:bg-gray-600 transition-colors"
                        >
                          Nein
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => setDeleteConfirm(file.path)}
                        className="p-2 text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
                        title="Datei löschen"
                      >
                        <TrashIcon className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t border-gray-200 dark:border-gray-600 bg-gray-50 dark:bg-gray-700">
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Insgesamt: {files.scripts.length + files.models.length + files.uploads.length} Dateien
          </div>
          <div className="flex space-x-3">
            <button
              onClick={loadFiles}
              className="px-4 py-2 text-sm bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-500 transition-colors"
            >
              🔄 Aktualisieren
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
            >
              Schließen
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FileManagementModal;
