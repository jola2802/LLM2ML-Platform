
import React, { useState, useRef } from 'react';
import { ModelType } from '../types';
import { apiService, CsvAnalysisResult } from '../services/apiService';
import { Spinner } from './ui/Spinner';
import { CloudUploadIcon } from './icons/CloudUploadIcon';
import { TrashIcon } from './icons/TrashIcon';

interface ProjectWizardProps {
  onBack: () => void;
  onSubmit: (projectData: any) => void;
}

// Verfügbare ML-Algorithmen
const ALGORITHMS: { [key: string]: { name: string; type: string } } = {
  // Klassifikation
  'RandomForestClassifier': { name: 'Random Forest Classifier', type: 'Classification' },
  'SVM': { name: 'Support Vector Machine', type: 'Classification' },
  'LogisticRegression': { name: 'Logistic Regression', type: 'Classification' },
  'XGBoostClassifier': { name: 'XGBoost Classifier', type: 'Classification' },
  'NeuralNetworkClassifier': { name: 'Neural Network Classifier', type: 'Classification' },
  
  // Regression
  'RandomForestRegressor': { name: 'Random Forest Regressor', type: 'Regression' },
  'SVR': { name: 'Support Vector Regression', type: 'Regression' },
  'LinearRegression': { name: 'Linear Regression', type: 'Regression' },
  'XGBoostRegressor': { name: 'XGBoost Regressor', type: 'Regression' },
  'NeuralNetworkRegressor': { name: 'Neural Network Regressor', type: 'Regression' },
};

const ProjectWizard: React.FC<ProjectWizardProps> = ({ onBack, onSubmit }) => {
  const [step, setStep] = useState(1);
  const [projectName, setProjectName] = useState('');
  const [dataSource, setDataSource] = useState<File | null>(null);
  const [csvAnalysis, setCsvAnalysis] = useState<CsvAnalysisResult | null>(null);
  const [llmRecommendations, setLlmRecommendations] = useState<any>(null);
  
  // Neue States für erweiterte Funktionalität
  const [excludedColumns, setExcludedColumns] = useState<string[]>([]);
  const [excludedFeatures, setExcludedFeatures] = useState<string[]>([]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('');
  const [selectedTargetVariable, setSelectedTargetVariable] = useState<string>('');
  const [selectedModelType, setSelectedModelType] = useState<string>('');
  const [manipulatedData, setManipulatedData] = useState<any>(null);
  const [availableFeatures, setAvailableFeatures] = useState<string[]>([]);
  const [userPreferences, setUserPreferences] = useState<string>('');
  
  // State für ausklappbare Sektionen
  const [showColumnManagement, setShowColumnManagement] = useState(false);
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [isProcessingData, setIsProcessingData] = useState(false);
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
      setExcludedColumns([]);
      setExcludedFeatures([]);

      try {
        // Datei an Backend senden für Basis-Analyse (ohne LLM)
        const analysis = await apiService.uploadFile(file);
        setCsvAnalysis(analysis);
        
        if (analysis.columns.length === 0) {
          throw new Error("Keine Spalten in der Datei gefunden. Bitte eine gültige Datei mit Überschriften bereitstellen.");
        }

        // Automatischen Projektnamen vorschlagen
        if (!projectName && analysis.fileName) {
          const fileName = analysis.fileName.replace(/\.[^/.]+$/, ""); // Remove extension
          setProjectName(`${fileName} - ML Model`);
        }

      } catch (error) {
        setAnalysisError(error instanceof Error ? error.message : "Fehler beim Analysieren der Daten.");
      } finally {
        setIsAnalyzing(false);
      }
    }
  };

  const handleDataManipulation = async () => {
    if (!csvAnalysis) return;
    
    setIsProcessingData(true);
    setAnalysisError(null);
    
    try {
      // LLM-Empfehlungen für manipulierte Daten abrufen
      const result = await apiService.analyzeData(
        csvAnalysis.filePath,
        excludedColumns,
        excludedFeatures,
        userPreferences,
      );
      
      console.log('AnalyzeData Result:', result); // Debug-Log
      
      setManipulatedData(result.analysis); // Geändert von result.manipulatedData
      
      // LLM-empfohlene Features verwenden - das sind die Features, die das LLM für das Training empfiehlt
      let features = [];
      if (result.recommendations && result.recommendations.features) {
        features = result.recommendations.features;
      } else if (result.availableFeatures && result.availableFeatures.length > 0) {
        // Fallback: Alle verfügbaren Features außer der Zielvariable
        features = result.availableFeatures.filter((f: string) => f !== result.recommendations?.targetVariable);
      } else if (result.analysis && result.analysis.columns) {
        // Fallback: Alle Spalten außer der Zielvariable
        features = result.analysis.columns.filter((f: string) => f !== result.recommendations?.targetVariable);
      }
      setAvailableFeatures(features);
      
      setLlmRecommendations(result.recommendations);
      
      // Automatisch empfohlene Werte setzen
      if (result.recommendations) {
        if (result.recommendations.targetVariable && !selectedTargetVariable) {
          setSelectedTargetVariable(result.recommendations.targetVariable);
        }
        if (result.recommendations.algorithm && !selectedAlgorithm) {
          setSelectedAlgorithm(result.recommendations.algorithm);
        }
        if (result.recommendations.modelType && !selectedModelType) {
          setSelectedModelType(result.recommendations.modelType);
        }
      }
      
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : "Fehler bei der Datenmanipulation.");
    } finally {
      setIsProcessingData(false);
    }
  };

  const toggleColumnExclusion = (column: string) => {
    setExcludedColumns(prev => 
      prev.includes(column) 
        ? prev.filter(col => col !== column)
        : [...prev, column]
    );
  };

  const toggleFeatureExclusion = (feature: string) => {
    setExcludedFeatures(prev => 
      prev.includes(feature) 
        ? prev.filter(feat => feat !== feature)
        : [...prev, feature]
    );
  };

  const handleCreateProject = async () => {
    if (!projectName || !csvAnalysis || !selectedAlgorithm || !selectedTargetVariable) return;
    
    setIsCreating(true);
    try {
      // Projekt mit benutzerdefinierten Einstellungen erstellen
      const finalRecommendations = {
        ...llmRecommendations,
        algorithm: selectedAlgorithm,
        targetVariable: selectedTargetVariable,
        modelType: selectedModelType,
        excludedColumns: excludedColumns,
        excludedFeatures: excludedFeatures,
        features: (availableFeatures || []).filter(f => 
          f !== selectedTargetVariable && !excludedFeatures.includes(f)
        )
      };

      const project = await apiService.createSmartProject({
        name: projectName,
        csvFilePath: csvAnalysis.filePath,
        recommendations: finalRecommendations
      });
      
      onSubmit(project);
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : "Fehler beim Erstellen des Projekts.");
    } finally {
      setIsCreating(false);
    }
  };

  const nextStep = async () => {
    if (step === 2) {
      // Beim Übergang von Schritt 2 zu 3: LLM-Empfehlungen abrufen
      await handleDataManipulation();
    }
    setStep(s => s + 1);
  };
  
  const prevStep = () => setStep(s => s - 1);

  const renderStep1 = () => (
    <div>
      <h3 className="text-xl font-semibold text-white mb-2">📁 Datei hochladen & Basis-Analyse</h3>
      <p className="text-sm text-slate-400 mb-6">
        Laden Sie Ihre Daten hoch und lassen Sie uns eine erste Analyse durchführen.
      </p>
      
      <div className="space-y-6">
        <div>
          <h4 className="text-lg font-medium text-white mb-2">Projektname</h4>
          <input
            type="text"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            placeholder="z.B. Kundenabwanderung Vorhersage"
            className="w-full bg-slate-700 border-slate-600 rounded-md p-3 text-white focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div>
          <h4 className="text-lg font-medium text-white mb-2">Datei hochladen</h4>
          <div 
            className="mt-1 flex justify-center px-6 pt-8 pb-8 border-2 border-slate-600 border-dashed rounded-lg cursor-pointer hover:border-blue-500 transition-all duration-200 hover:bg-slate-800/30"
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="space-y-2 text-center">
              <CloudUploadIcon className="mx-auto h-16 w-16 text-slate-400" />
              <div className="text-slate-400">
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
                <h4 className="text-blue-400 font-medium">📊 Analysiere Datenstruktur...</h4>
                <p className="text-blue-300 text-sm">Erkenne Spalten und Datentypen</p>
              </div>
            </div>
          </div>
        )}

        {csvAnalysis && (
          <div className="bg-slate-900/50 p-6 rounded-lg border border-slate-700">
            <h4 className="font-semibold text-green-400 mb-4">📊 Datenübersicht</h4>
            <div className="grid grid-cols-2 gap-6 text-sm">
              <div>
                <p className="text-slate-400">Zeilen: <span className="text-white font-mono">{csvAnalysis.rowCount.toLocaleString()}</span></p>
                <p className="text-slate-400">Spalten: <span className="text-white font-mono">{csvAnalysis.columns.length}</span></p>
              </div>
              <div>
                <p className="text-slate-400">Numerische Spalten: <span className="text-blue-400 font-mono">{Object.values(csvAnalysis.dataTypes).filter(t => t === 'numeric').length}</span></p>
                <p className="text-slate-400">Kategorische Spalten: <span className="text-green-400 font-mono">{Object.values(csvAnalysis.dataTypes).filter(t => t === 'categorical').length}</span></p>
              </div>
            </div>
            
            <div className="mt-4">
              <h5 className="text-sm font-medium text-slate-300 mb-2">Verfügbare Spalten:</h5>
              <div className="flex flex-wrap gap-2">
                {csvAnalysis.columns.map((column) => (
                  <span 
                    key={column}
                    className={`px-2 py-1 rounded text-xs ${
                      csvAnalysis.dataTypes[column] === 'numeric' 
                        ? 'bg-blue-900/30 text-blue-300 border border-blue-500/30'
                        : 'bg-green-900/30 text-green-300 border border-green-500/30'
                    }`}
                  >
                    {column}
                  </span>
                ))}
              </div>
            </div>
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
      <h3 className="text-xl font-semibold text-white mb-2">✂️ Datenmanipulation & Feature-Auswahl</h3>
      <p className="text-sm text-slate-400 mb-6">
        Entfernen Sie unerwünschte Spalten und Features bevor die KI-Analyse startet.
      </p>

      <div className="space-y-6">
        {/* Anforderungen an die KI */}
        <div className="bg-slate-900/50 p-6 rounded-lg border border-slate-700">
            <h4 className="font-semibold text-white mb-2">📝 Anforderungen</h4>
            <p className="text-sm text-slate-400 mb-4">
              Beschreiben Sie hier spezielle Anforderungen, z. B. gewünschte Zielvariable, zu priorisierende Features, bevorzugter Modelltyp/Algorithmus, Metriken oder sonstige Constraints.
            </p>
            <textarea
              value={userPreferences}
              onChange={(e) => setUserPreferences(e.target.value)}
              placeholder="Beispiel: Zielvariable = churn; Bevorzugt Klassifikation; Algorithmus: LogisticRegression;"
              className="w-full bg-slate-700 border-slate-600 rounded-md p-3 text-white focus:ring-blue-500 focus:border-blue-500 min-h-[100px]"
            />
            <p className="text-xs text-slate-400 mt-2">Diese Hinweise werden beim Erstellen der Empfehlungen berücksichtigt.</p>
        </div>

        {csvAnalysis && (
          <>
            {/* <div className="bg-slate-900/50 p-6 rounded-lg border border-slate-700">
              <button
                onClick={() => setShowColumnManagement(!showColumnManagement)}
                className="w-full flex items-center justify-between text-left"
              >
                <div>
                  <h4 className="font-semibold text-white">🗂️ Spalten verwalten</h4>
                  <p className="text-sm text-slate-400 mt-1">
                    {excludedColumns.length > 0 
                      ? `${excludedColumns.length} Spalte(n) entfernt` 
                      : 'Klicken Sie auf Spalten, um sie aus dem Dataset zu entfernen.'
                    }
                  </p>
                </div>
                <div className={`transform transition-transform duration-200 ${showColumnManagement ? 'rotate-180' : ''}`}>
                  <svg className="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </button>
              
              {showColumnManagement && (
                <div className="mt-4 pt-4 border-t border-slate-600">
                  <p className="text-sm text-slate-400 mb-4">
                    Klicken Sie auf Spalten, um sie aus dem Dataset zu entfernen.
                  </p>
                  
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {csvAnalysis.columns.map((column) => (
                      <div
                        key={column}
                        onClick={() => toggleColumnExclusion(column)}
                        className={`cursor-pointer p-3 rounded-lg border transition-all ${
                          excludedColumns.includes(column)
                            ? 'bg-red-900/30 border-red-500/50 text-red-300'
                            : `${csvAnalysis.dataTypes[column] === 'numeric' 
                              ? 'bg-blue-900/30 border-blue-500/30 text-blue-300 hover:bg-blue-800/40'
                              : 'bg-green-900/30 border-green-500/30 text-green-300 hover:bg-green-800/40'
                            }`
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium">{column}</span>
                          {excludedColumns.includes(column) && (
                            <TrashIcon className="h-4 w-4" />
                          )}
                        </div>
                        <div className="text-xs opacity-75 mt-1">
                          {csvAnalysis.dataTypes[column]}
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  {excludedColumns.length > 0 && (
                    <div className="mt-4 p-3 bg-red-900/20 border border-red-500/30 rounded">
                      <p className="text-red-300 text-sm">
                        <strong>Entfernte Spalten ({excludedColumns.length}):</strong> {excludedColumns.join(', ')}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div> */}

            <div className="bg-slate-900/50 p-6 rounded-lg border border-slate-700">
              <h4 className="font-semibold text-white mb-4">🎯 Features für ML ausschließen</h4>
              <p className="text-sm text-slate-400 mb-4">
                Schließen Sie Features aus, die nicht für das Machine Learning verwendet werden sollen.
              </p>
              
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {csvAnalysis.columns
                  .filter(col => !excludedColumns.includes(col))
                  .map((feature) => (
                    <div
                      key={feature}
                      onClick={() => toggleFeatureExclusion(feature)}
                      className={`cursor-pointer p-3 rounded-lg border transition-all ${
                        excludedFeatures.includes(feature)
                          ? 'bg-orange-900/30 border-orange-500/50 text-orange-300'
                          : `${csvAnalysis.dataTypes[feature] === 'numeric' 
                            ? 'bg-blue-900/30 border-blue-500/30 text-blue-300 hover:bg-blue-800/40'
                            : 'bg-green-900/30 border-green-500/30 text-green-300 hover:bg-green-800/40'
                          }`
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">{feature}</span>
                        {excludedFeatures.includes(feature) && (
                          <span className="text-xs">🚫</span>
                        )}
                      </div>
                      <div className="text-xs opacity-75 mt-1">
                        {csvAnalysis.dataTypes[feature]}
                      </div>
                    </div>
                  ))}
              </div>
              
              {excludedFeatures.length > 0 && (
                <div className="mt-4 p-3 bg-orange-900/20 border border-orange-500/30 rounded">
                  <p className="text-orange-300 text-sm">
                    <strong>Ausgeschlossene Features ({excludedFeatures.length}):</strong> {excludedFeatures.join(', ')}
                  </p>
                </div>
              )}
            </div>

            <div className="bg-blue-900/30 border border-blue-500/50 rounded-lg p-6">
              <h4 className="text-blue-400 font-medium mb-2">📊 Manipulierte Datenübersicht</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-400">Verbleibende Spalten: <span className="text-white font-mono">{csvAnalysis.columns.length - excludedColumns.length}</span></p>
                  <p className="text-gray-400">Verfügbare Features: <span className="text-blue-400 font-mono">{csvAnalysis.columns.length - excludedColumns.length - excludedFeatures.length}</span></p>
                </div>
                <div>
                  <p className="text-gray-400">Entfernte Spalten: <span className="text-red-400 font-mono">{excludedColumns.length}</span></p>
                  <p className="text-gray-400">Ausgeschlossene Features: <span className="text-orange-400 font-mono">{excludedFeatures.length}</span></p>
                </div>
              </div>
            </div>
          </>
        )}

        {analysisError && (
          <div className="bg-red-900/30 border border-red-500/50 rounded-lg p-4">
            <p className="text-red-400">❌ {analysisError}</p>
          </div>
        )}
      </div>
    </div>
  );

  const renderStep3 = () => (
    <div>
      <h3 className="text-xl font-semibold text-white mb-2">⚙️ Hyperparameter & ML-Konfiguration</h3>
      <p className="text-sm text-gray-400 mb-6">
        Wählen Sie Algorithmus, Zielvariable und weitere ML-Parameter.
      </p>
      
      <div className="space-y-6">
        {isProcessingData && (
          <div className="bg-blue-900/30 border border-blue-500/50 rounded-lg p-6">
            <div className="flex items-center space-x-4">
              <Spinner size="sm" />
              <div>
                <h4 className="text-blue-400 font-medium">🧠 KI analysiert manipulierte Daten...</h4>
                <p className="text-blue-300 text-sm">Erstelle optimierte Empfehlungen für Ihre Auswahl</p>
              </div>
            </div>
          </div>
        )}

        <div className="bg-gray-900/50 p-6 rounded-lg border border-gray-700">
          <h4 className="font-semibold text-white mb-4">🎯 Zielvariable auswählen</h4>
          <p className="text-sm text-gray-400 mb-4">
            Wählen Sie die Variable, die vorhergesagt werden soll.
          </p>
          
          <select
            value={selectedTargetVariable}
            onChange={(e) => setSelectedTargetVariable(e.target.value)}
            className="w-full bg-gray-700 border-gray-600 rounded-md p-3 text-white focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="">-- Zielvariable wählen --</option>
            {csvAnalysis?.columns
              .filter(col => !excludedColumns.includes(col) && !excludedFeatures.includes(col))
              .map((column) => (
                <option key={column} value={column}>
                  {column} ({csvAnalysis.dataTypes[column]})
                </option>
              ))}
          </select>
          
          {llmRecommendations?.targetVariable && (
            <div className="mt-3 p-3 bg-green-900/20 border border-green-500/30 rounded">
              <p className="text-green-300 text-sm">
                💡 <strong>KI-Empfehlung:</strong> {llmRecommendations.targetVariable}
                {llmRecommendations.targetVariable !== selectedTargetVariable && selectedTargetVariable && (
                  <button 
                    onClick={() => setSelectedTargetVariable(llmRecommendations.targetVariable)}
                    className="ml-2 text-xs px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700"
                  >
                    Übernehmen
                  </button>
                )}
              </p>
            </div>
          )}
        </div>

        <div className="bg-gray-900/50 p-6 rounded-lg border border-gray-700">
          <h4 className="font-semibold text-white mb-4">🤖 Modell-Typ bestimmen</h4>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div
              onClick={() => setSelectedModelType('Classification')}
              className={`p-4 rounded-lg border cursor-pointer transition-all ${
                selectedModelType === 'Classification'
                  ? 'bg-blue-900/40 border-blue-500 text-blue-300'
                  : 'bg-gray-800/50 border-gray-600 text-gray-300 hover:bg-gray-700/50'
              }`}
            >
              <h5 className="font-medium">🏷️ Klassifikation</h5>
              <p className="text-sm mt-1 opacity-75">
                Kategorien vorhersagen (z.B. Spam/Nicht-Spam, Krebsdiagnose)
              </p>
            </div>
            
            <div
              onClick={() => setSelectedModelType('Regression')}
              className={`p-4 rounded-lg border cursor-pointer transition-all ${
                selectedModelType === 'Regression'
                  ? 'bg-purple-900/40 border-purple-500 text-purple-300'
                  : 'bg-gray-800/50 border-gray-600 text-gray-300 hover:bg-gray-700/50'
              }`}
            >
              <h5 className="font-medium">📈 Regression</h5>
              <p className="text-sm mt-1 opacity-75">
                Numerische Werte vorhersagen (z.B. Preise, Temperaturen)
              </p>
            </div>
          </div>
          
          {llmRecommendations?.modelType && (
            <div className="p-3 bg-green-900/20 border border-green-500/30 rounded">
              <p className="text-green-300 text-sm">
                💡 <strong>KI-Empfehlung:</strong> {llmRecommendations.modelType}
                {llmRecommendations.modelType !== selectedModelType && selectedModelType && (
                  <button 
                    onClick={() => setSelectedModelType(llmRecommendations.modelType)}
                    className="ml-2 text-xs px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700"
                  >
                    Übernehmen
                  </button>
                )}
              </p>
            </div>
          )}
        </div>

        {selectedModelType && (
          <div className="bg-gray-900/50 p-6 rounded-lg border border-gray-700">
            <h4 className="font-semibold text-white mb-4">🧮 Algorithmus auswählen</h4>
            <p className="text-sm text-gray-400 mb-4">
              Wählen Sie den ML-Algorithmus für Ihr {selectedModelType}-Problem.
            </p>
            
            <select
              value={selectedAlgorithm}
              onChange={(e) => setSelectedAlgorithm(e.target.value)}
              className="w-full bg-gray-700 border-gray-600 rounded-md p-3 text-white focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">-- Algorithmus wählen --</option>
              {Object.entries(ALGORITHMS)
                .filter(([_, algo]) => algo.type === selectedModelType)
                .map(([key, algo]) => (
                  <option key={key} value={key}>
                    {algo.name}
                  </option>
                ))}
            </select>
            
            {llmRecommendations?.algorithm && (
              <div className="mt-3 p-3 bg-green-900/20 border border-green-500/30 rounded">
                <p className="text-green-300 text-sm">
                  💡 <strong>KI-Empfehlung:</strong> {ALGORITHMS[llmRecommendations.algorithm]?.name || llmRecommendations.algorithm}
                  {llmRecommendations.algorithm !== selectedAlgorithm && selectedAlgorithm && (
                    <button 
                      onClick={() => setSelectedAlgorithm(llmRecommendations.algorithm)}
                      className="ml-2 text-xs px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700"
                    >
                      Übernehmen
                    </button>
                  )}
                </p>
              </div>
            )}
          </div>
        )}

        {llmRecommendations && (
          <div className="bg-gradient-to-r from-green-900/40 to-blue-900/40 border border-green-500/50 rounded-lg p-6">
            <h4 className="text-green-400 font-bold mb-4 flex items-center">
              🎯 KI-Empfehlungen
            </h4>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div className="space-y-3">
                <div>
                  <p className="text-gray-300 font-medium">Empfohlene Features ({(availableFeatures || []).filter(f => f !== selectedTargetVariable && !excludedFeatures.includes(f)).length}):</p>
                  <div className="text-green-300 text-xs max-h-20 overflow-y-auto">
                    {(availableFeatures || [])
                      .filter(f => f !== selectedTargetVariable && !excludedFeatures.includes(f))
                      .join(', ') || 'Keine Features verfügbar'}
                  </div>
                </div>
              </div>
              
              <div className="space-y-3">
                <div>
                  <p className="text-gray-300 font-medium">Begründung:</p>
                  <div className="text-green-300 text-xs max-h-20 overflow-y-auto">
                    {llmRecommendations.reasoning || 'Keine Begründung verfügbar'}
                  </div>
                </div>
              </div>
            </div>
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

  const renderStep4 = () => (
    <div>
      <h3 className="text-xl font-semibold text-white mb-4">✅ Bestätigung & Training starten</h3>
      <p className="text-gray-400 mb-6">Überprüfen Sie Ihre Konfiguration vor dem Training.</p>
      
      <div className="space-y-6">
        <div className="bg-gray-900/50 p-6 rounded-lg border border-gray-700">
          <h4 className="font-semibold text-white mb-4">📋 Projekt-Zusammenfassung</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3 text-sm">
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
              <div className="flex justify-between">
                <span className="text-gray-400 font-medium">Ursprüngliche Spalten:</span>
                <span className="text-white">{csvAnalysis?.columns.length}</span>
              </div>
            </div>
            
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400 font-medium">Modell-Typ:</span>
                <span className="text-purple-400">{selectedModelType}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400 font-medium">Algorithmus:</span>
                <span className="text-blue-400">{ALGORITHMS[selectedAlgorithm]?.name || selectedAlgorithm}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400 font-medium">Zielvariable:</span>
                <span className="text-green-400">{selectedTargetVariable}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400 font-medium">Features:</span>
                <span className="text-cyan-400">
                  {(availableFeatures || []).filter(f => f !== selectedTargetVariable && !excludedFeatures.includes(f)).length}
                </span>
              </div>
            </div>
          </div>
        </div>

        {excludedColumns.length > 0 && (
          <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
            <h5 className="text-red-400 font-medium mb-2">🗑️ Entfernte Spalten ({excludedColumns.length})</h5>
            <p className="text-red-300 text-sm">{excludedColumns.join(', ')}</p>
          </div>
        )}

        {excludedFeatures.length > 0 && (
          <div className="bg-orange-900/20 border border-orange-500/30 rounded-lg p-4">
            <h5 className="text-orange-400 font-medium mb-2">🚫 Ausgeschlossene Features ({excludedFeatures.length})</h5>
            <p className="text-orange-300 text-sm">{excludedFeatures.join(', ')}</p>
          </div>
        )}

        <div className="bg-blue-900/40 border border-blue-500/50 rounded-lg p-6">
          <h5 className="text-blue-400 font-medium mb-2">🔬 Verwendete Features für ML</h5>
          <div className="text-blue-300 text-sm">
            {(availableFeatures || [])
              .filter(f => f !== selectedTargetVariable && !excludedFeatures.includes(f))
              .join(', ') || 'Keine Features verfügbar'}
          </div>
        </div>

        {/* <div className="bg-yellow-900/30 border border-yellow-500/50 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <span className="text-yellow-400 text-xl">⚡</span>
            <div>
              <h5 className="text-yellow-400 font-medium">Automatisches Training</h5>
              <p className="text-yellow-200 text-sm mt-1">
                Das Training startet nach der Erstellung. Sie können den Fortschritt im Dashboard verfolgen.
              </p>
            </div>
          </div>
        </div> */}
      </div>
    </div>
  );

  const steps = [
    { 
      num: 1, 
      title: 'Upload & Analyse', 
      content: renderStep1(), 
      canProceed: !!(projectName && csvAnalysis && !isAnalyzing) 
    },
    { 
      num: 2, 
      title: 'Datenmanipulation', 
      content: renderStep2(), 
      canProceed: !!(csvAnalysis) 
    },
    { 
      num: 3, 
      title: 'ML-Konfiguration', 
      content: renderStep3(), 
      canProceed: !!(selectedTargetVariable && selectedAlgorithm && selectedModelType && !isProcessingData)
    },
    { 
      num: 4, 
      title: 'Bestätigung', 
      content: renderStep4(), 
      canProceed: true 
    },
  ];
  
  return (
    <div className="max-w-4xl mx-auto bg-slate-800 rounded-lg shadow-xl p-6 sm:p-8 animate-fade-in">
      <div className="flex justify-between items-start mb-6">
        <div>
          <h2 className="text-3xl font-bold text-white">Intelligenter ML-Wizard</h2>
          <p className="text-slate-400">Schritt {step} von {steps.length}: {steps[step-1].title}</p>
        </div>
        <button onClick={onBack} className="text-slate-400 hover:text-white text-xl">&times;</button>
      </div>

      {/* Fortschrittsanzeige */}
      <div className="flex items-center justify-between mb-6">
        {steps.map((stepInfo, index) => (
          <div key={index} className="flex items-center">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              index + 1 < step 
                ? 'bg-emerald-600 text-white' 
                : index + 1 === step 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-slate-600 text-slate-300'
            }`}>
              {index + 1 < step ? '✓' : index + 1}
            </div>
            <span className={`ml-2 text-sm ${
              index + 1 <= step ? 'text-white' : 'text-slate-400'
            }`}>
              {stepInfo.title}
            </span>
            {index < steps.length - 1 && (
              <div className={`mx-4 h-0.5 w-16 ${
                index + 1 < step ? 'bg-emerald-600' : 'bg-slate-600'
              }`} />
            )}
          </div>
        ))}
      </div>

      <div className="py-6">
        {steps[step-1].content}
      </div>
      
      <div className="flex justify-between items-center pt-6 border-t border-slate-700">
        <button
          onClick={step === 1 ? onBack : prevStep}
          className="px-6 py-2 border border-slate-600 text-sm font-medium rounded-md text-slate-300 hover:bg-slate-700 transition-colors"
        >
          {step === 1 ? 'Abbrechen' : 'Zurück'}
        </button>
        
        {step < steps.length ? (
          <button
            onClick={nextStep}
            disabled={!steps[step-1].canProceed || isAnalyzing || isProcessingData}
            className="px-6 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 disabled:bg-slate-500 disabled:cursor-not-allowed transition-colors"
          >
            {step === 2 && isProcessingData ? (
              <div className="flex items-center space-x-2">
                <Spinner size="sm" />
                <span>Analysiere...</span>
              </div>
            ) : (
              'Weiter'
            )}
          </button>
        ) : (
          <button
            onClick={handleCreateProject}
            disabled={isCreating}
            className="px-6 py-3 border border-transparent text-sm font-medium rounded-md text-white bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
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
