
import React, { useState, useRef } from 'react';
import { ModelType } from '../types';
import { apiService, CsvAnalysisResult } from '../services/apiService';
import { Spinner } from './ui/Spinner';
import { CloudUploadIcon } from './icons/CloudUploadIcon';

interface ProjectWizardProps {
  onBack: () => void;
  onSubmit: (projectData: any) => void;
}

const ProjectWizard: React.FC<ProjectWizardProps> = ({ onBack, onSubmit }) => {
  const [step, setStep] = useState(1);
  const [projectName, setProjectName] = useState('');
  const [dataSource, setDataSource] = useState<File | null>(null);
  const [csvAnalysis, setCsvAnalysis] = useState<CsvAnalysisResult | null>(null);
  const [llmRecommendations, setLlmRecommendations] = useState<any>(null);
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setDataSource(file);
      setIsAnalyzing(true);
      setAnalysisError(null);
      setCsvAnalysis(null);
      setLlmRecommendations(null);

      try {
        // Datei an Backend senden und intelligente Analyse
        const analysis = await apiService.uploadFile(file);
        setCsvAnalysis(analysis);
        
        if (analysis.columns.length === 0) {
          throw new Error("Keine Spalten in der Datei gefunden. Bitte eine gültige Datei mit Überschriften bereitstellen.");
        }

        // LLM-Empfehlungen automatisch setzen
        if (analysis.recommendations) {
          setLlmRecommendations(analysis.recommendations);
          
          // Automatischen Projektnamen vorschlagen
          if (analysis.recommendations.dataSourceName && !projectName) {
            setProjectName(`${analysis.recommendations.dataSourceName} - ${analysis.recommendations.modelType} Model`);
          }
        }

      } catch (error) {
        setAnalysisError(error instanceof Error ? error.message : "Fehler beim Analysieren der Daten.");
      } finally {
        setIsAnalyzing(false);
      }
    }
  };

  const handleCreateProject = async () => {
    if (!projectName || !csvAnalysis || !llmRecommendations) return;
    
    setIsCreating(true);
    try {
      // Intelligentes Projekt mit LLM-Empfehlungen erstellen
      const project = await apiService.createSmartProject({
        name: projectName,
        csvFilePath: csvAnalysis.filePath,
        recommendations: llmRecommendations
      });
      
      onSubmit(project);
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : "Fehler beim Erstellen des Projekts.");
    } finally {
      setIsCreating(false);
    }
  };

  const nextStep = () => setStep(s => s + 1);
  const prevStep = () => setStep(s => s - 1);

  const renderStep1 = () => (
    <div>
      <h3 className="text-xl font-semibold text-white mb-2">🚀 ML-Projekt erstellen</h3>
      <p className="text-sm text-gray-400 mb-6">
        Unser KI-Experte analysiert Ihre Daten automatisch und wählt den optimalen Algorithmus und die besten Features aus.
      </p>
      
      <div className="space-y-6">
        <div>
          <h4 className="text-lg font-medium text-white mb-2">Projektname</h4>
          <input
            type="text"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            placeholder="z.B. Kundenabwanderung Vorhersage"
            className="w-full bg-gray-700 border-gray-600 rounded-md p-3 text-white focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div>
          <h4 className="text-lg font-medium text-white mb-2">Datei hochladen</h4>
          <div 
            className="mt-1 flex justify-center px-6 pt-8 pb-8 border-2 border-gray-600 border-dashed rounded-lg cursor-pointer hover:border-blue-500 transition-all duration-200 hover:bg-gray-800/30"
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="space-y-2 text-center">
              <CloudUploadIcon className="mx-auto h-16 w-16 text-gray-400" />
              <div className="text-gray-400">
                <p className="text-lg">{dataSource ? `📄 ${dataSource.name}` : 'Für Datei-Upload hier klicken'}</p>
                <p className="text-sm">{dataSource ? `${(dataSource.size / 1024).toFixed(2)} KB` : 'Maximal 10MB'}</p>
              </div>
              <input 
                ref={fileInputRef} 
                id="file-upload" 
                name="file-upload" 
                type="file" 
                className="sr-only" 
                onChange={handleFileChange} 
                accept=".csv, .json, .xlsx, .xls, .txt, .pdf, .xml, .docx, .doc" 
              />
            </div>
          </div>
        </div>

        {isAnalyzing && (
          <div className="bg-blue-900/30 border border-blue-500/50 rounded-lg p-6">
            <div className="flex items-center space-x-4">
              <Spinner size="sm" />
              <div>
                <h4 className="text-blue-400 font-medium">🧠 KI-Experte analysiert Ihre Daten...</h4>
                <p className="text-blue-300 text-sm">Bestimme optimalen Algorithmus und Features</p>
              </div>
            </div>
          </div>
        )}

        {csvAnalysis && (
          <div className="bg-gray-900/50 p-6 rounded-lg border border-gray-700">
            <h4 className="font-semibold text-green-400 mb-4">📊 Datenanalyse</h4>
            <div className="grid grid-cols-2 gap-6 text-sm">
              <div>
                <p className="text-gray-400">Zeilen: <span className="text-white font-mono">{csvAnalysis.rowCount.toLocaleString()}</span></p>
                <p className="text-gray-400">Spalten: <span className="text-white font-mono">{csvAnalysis.columns.length}</span></p>
              </div>
              <div>
                <p className="text-gray-400">Numerische Spalten: <span className="text-blue-400 font-mono">{Object.values(csvAnalysis.dataTypes).filter(t => t === 'numeric').length}</span></p>
                <p className="text-gray-400">Kategorische Spalten: <span className="text-green-400 font-mono">{Object.values(csvAnalysis.dataTypes).filter(t => t === 'categorical').length}</span></p>
              </div>
            </div>
          </div>
        )}

        {llmRecommendations && (
          <div className="bg-gradient-to-r from-green-900/40 to-blue-900/40 border border-green-500/50 rounded-lg p-6">
            <h4 className="text-green-400 font-bold mb-4 flex items-center">
              🎯 KI-Experten-Empfehlung
              <span className="ml-2 px-2 py-1 bg-green-600 text-green-100 text-xs rounded-full">
                Automatisch optimiert
              </span>
            </h4>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div className="space-y-3">
                <div>
                  <p className="text-gray-300 font-medium">Zielvariable:</p>
                  <p className="text-green-300 font-mono">{llmRecommendations.targetVariable}</p>
                </div>
                <div>
                  <p className="text-gray-300 font-medium">Model-Typ:</p>
                  <p className="text-green-300">{llmRecommendations.modelType}</p>
                </div>
                <div>
                  <p className="text-gray-300 font-medium">Algorithmus:</p>
                  <p className="text-green-300">{llmRecommendations.algorithm}</p>
                </div>
              </div>
              
              <div className="space-y-3">
                <div>
                  <p className="text-gray-300 font-medium">Features ({llmRecommendations.features?.length || 0}):</p>
                  <div className="text-green-300 text-xs max-h-20 overflow-y-auto">
                    {llmRecommendations.features?.join(', ') || 'Keine Features'}
                  </div>
                </div>
              </div>
            </div>
            
            <details className="mt-4">
              <summary className="cursor-pointer text-green-200 hover:text-green-100 font-medium">
                💡 Begründung anzeigen
              </summary>
              <div className="mt-3 p-4 bg-green-950/30 rounded-lg">
                <p className="text-green-100 text-sm leading-relaxed">
                  {llmRecommendations.reasoning || 'Keine Begründung verfügbar'}
                </p>
              </div>
            </details>
          </div>
        )}

        {analysisError && (
          <div className="bg-red-900/30 border border-red-500/50 rounded-lg p-4">
            <p className="text-red-400">❌ {analysisError}</p>
          </div>
        )}
      </div>
    </div>
  );

  const renderStep2 = () => (
    <div>
      <h3 className="text-xl font-semibold text-white mb-4">🔍 Projektübersicht</h3>
      <p className="text-gray-400 mb-6">Überprüfen Sie die automatischen Empfehlungen vor dem Start des Trainings.</p>
      
      <div className="space-y-6">
        <div className="bg-gray-900/50 p-6 rounded-lg border border-gray-700">
          <h4 className="font-semibold text-white mb-4">Projekt-Konfiguration</h4>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400 font-medium">Projektname:</span>
              <span className="text-white">{projectName}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400 font-medium">Datenquelle:</span>
              <span className="text-white">{dataSource?.name}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400 font-medium">Zeilen:</span>
              <span className="text-blue-400">{csvAnalysis?.rowCount.toLocaleString()}</span>
            </div>
          </div>
        </div>

        {llmRecommendations && (
          <div className="bg-gradient-to-r from-purple-900/40 to-blue-900/40 border border-purple-500/50 rounded-lg p-6">
            <h4 className="font-semibold text-purple-400 mb-4">🤖 Automatische ML-Konfiguration</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-gray-300"><strong>Model-Typ:</strong> {llmRecommendations.modelType}</p>
                <p className="text-gray-300"><strong>Algorithmus:</strong> {llmRecommendations.algorithm}</p>
                <p className="text-gray-300"><strong>Zielvariable:</strong> {llmRecommendations.targetVariable}</p>
              </div>
              <div>
                <p className="text-gray-300"><strong>Features:</strong> {llmRecommendations.features?.length || 0} ausgewählt</p>
                <p className="text-gray-300"><strong>Hyperparameter:</strong> {Object.keys(llmRecommendations.hyperparameters || {}).length} konfiguriert</p>
              </div>
            </div>
          </div>
        )}

        <div className="bg-yellow-900/30 border border-yellow-500/50 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <span className="text-yellow-400 text-xl">⚡</span>
            <div>
              <h5 className="text-yellow-400 font-medium">Automatisches Training</h5>
              <p className="text-yellow-200 text-sm mt-1">
                Das Training startet automatisch nach der Erstellung. Sie können den Fortschritt im Dashboard verfolgen.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const steps = [
    { 
      num: 1, 
      title: 'Upload & Analyse', 
      content: renderStep1(), 
      canProceed: !!(projectName && csvAnalysis && llmRecommendations && !isAnalyzing) 
    },
    { 
      num: 2, 
      title: 'Bestätigung', 
      content: renderStep2(), 
      canProceed: true 
    },
  ];
  
  return (
    <div className="max-w-4xl mx-auto bg-gray-800 rounded-lg shadow-xl p-6 sm:p-8 animate-fade-in">
      <div className="flex justify-between items-start mb-6">
        <div>
          <h2 className="text-3xl font-bold text-white">Intelligentes ML-Projekt</h2>
          <p className="text-gray-400">Schritt {step} von {steps.length}: {steps[step-1].title}</p>
        </div>
        <button onClick={onBack} className="text-gray-400 hover:text-white text-xl">&times;</button>
      </div>

      <div className="py-6">
        {steps[step-1].content}
      </div>
      
      <div className="flex justify-between items-center pt-6 border-t border-gray-700">
        <button
          onClick={step === 1 ? onBack : prevStep}
          className="px-6 py-2 border border-gray-600 text-sm font-medium rounded-md text-gray-300 hover:bg-gray-700 transition-colors"
        >
          {step === 1 ? 'Abbrechen' : 'Zurück'}
        </button>
        
        {step < steps.length ? (
          <button
            onClick={nextStep}
            disabled={!steps[step-1].canProceed || isAnalyzing}
            className="px-6 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-500 disabled:cursor-not-allowed transition-colors"
          >
            Weiter
          </button>
        ) : (
          <button
            onClick={handleCreateProject}
            disabled={isCreating}
            className="px-6 py-3 border border-transparent text-sm font-medium rounded-md text-white bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {isCreating ? (
              <div className="flex items-center space-x-2">
                <Spinner size="sm" />
                <span>Projekt wird erstellt...</span>
              </div>
            ) : (
              '🚀 Training starten'
            )}
          </button>
        )}
      </div>
    </div>
  );
};

export default ProjectWizard;
