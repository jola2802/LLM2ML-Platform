
import React, { useState, useRef, useEffect } from 'react';
import { ModelType } from '../types';
import { apiService, CsvAnalysisResult } from '../services/apiService';
import { Spinner } from './ui/Spinner';
import { CloudUploadIcon } from './icons/CloudUploadIcon';
import { TrashIcon } from './icons/TrashIcon';

// Komponente zum Hinzufügen von Features
const AddFeatureForm: React.FC<{
  availableColumns: string[];
  onAdd: (feature: any) => void;
  onCancel: () => void;
}> = ({ availableColumns, onAdd, onCancel }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [formula, setFormula] = useState('');
  const [reasoning, setReasoning] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !formula) {
      alert('Bitte Name und Formel eingeben');
      return;
    }
    onAdd({ name, description, formula, reasoning });
    setName('');
    setDescription('');
    setFormula('');
    setReasoning('');
  };

  const insertColumn = (column: string) => {
    setFormula((prev: string) => prev + `data['${column}']`);
  };

  return (
    <form onSubmit={handleSubmit} className="bg-slate-800/50 border border-slate-600 rounded-lg p-4 space-y-4">
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-1">Feature-Name *</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full bg-slate-700 border-slate-600 rounded-md p-2 text-white text-sm"
          placeholder="z.B. feature_age_income_ratio"
          required
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-1">Beschreibung</label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          className="w-full bg-slate-700 border-slate-600 rounded-md p-2 text-white text-sm"
          placeholder="Was stellt dieses Feature dar?"
          rows={2}
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-1">Python-Formel *</label>
        <div className="mb-2">
          <p className="text-xs text-slate-400 mb-2">Available columns (click to insert):</p>
          <div className="flex flex-wrap gap-2">
            {availableColumns.map((col) => (
              <button
                key={col}
                type="button"
                onClick={() => insertColumn(col)}
                className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded"
              >
                {col}
              </button>
            ))}
          </div>
        </div>
        <textarea
          value={formula}
          onChange={(e) => setFormula(e.target.value)}
          className="w-full bg-slate-700 border-slate-600 rounded-md p-2 text-white text-sm font-mono"
          placeholder="z.B. data['age'] / data['income']"
          rows={3}
          required
        />
        <p className="text-xs text-slate-400 mt-1">Use 'data['columnname']' for column access</p>
      </div>
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-1">Reasoning</label>
        <textarea
          value={reasoning}
          onChange={(e) => setReasoning(e.target.value)}
          className="w-full bg-slate-700 border-slate-600 rounded-md p-2 text-white text-sm"
          placeholder="Why does this feature improve the model performance?"
          rows={2}
        />
      </div>
      <div className="flex gap-2">
        <button
          type="submit"
          className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-md"
        >
          Add feature
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white text-sm rounded-md"
        >
          Cancel
        </button>
      </div>
    </form>
  );
};

// Komponente zum Bearbeiten von Features
const EditFeatureForm: React.FC<{
  feature: any;
  availableColumns: string[];
  onSave: (feature: any) => void;
  onCancel: () => void;
}> = ({ feature, availableColumns, onSave, onCancel }) => {
  const [name, setName] = useState(feature.name || '');
  const [description, setDescription] = useState(feature.description || '');
  const [formula, setFormula] = useState(feature.formula || '');
  const [reasoning, setReasoning] = useState(feature.reasoning || '');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !formula) {
      alert('Please enter name and formula');
      return;
    }
    onSave({ name, description, formula, reasoning });
  };

  const insertColumn = (column: string) => {
    setFormula((prev: string) => prev + `data['${column}']`);
  };

  return (
    <form onSubmit={handleSubmit} className="mt-4 bg-slate-800/50 border border-slate-600 rounded-lg p-4 space-y-4">
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-1">Feature name *</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full bg-slate-700 border-slate-600 rounded-md p-2 text-white text-sm"
          required
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-1">Description</label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          className="w-full bg-slate-700 border-slate-600 rounded-md p-2 text-white text-sm"
          rows={2}
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-1">Python formula *</label>
        <div className="mb-2">
          <p className="text-xs text-slate-400 mb-2">Available columns (click to insert):</p>
          <div className="flex flex-wrap gap-2">
            {availableColumns.map((col) => (
              <button
                key={col}
                type="button"
                onClick={() => insertColumn(col)}
                className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded"
              >
                {col}
              </button>
            ))}
          </div>
        </div>
        <textarea
          value={formula}
          onChange={(e) => setFormula(e.target.value)}
          className="w-full bg-slate-700 border-slate-600 rounded-md p-2 text-white text-sm font-mono"
          rows={3}
          required
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-1">AI reasoning</label>
        <textarea
          value={reasoning}
          onChange={(e) => setReasoning(e.target.value)}
          className="w-full bg-slate-700 border-slate-600 rounded-md p-2 text-white text-sm"
          rows={2}
        />
      </div>
      <div className="flex gap-2">
        <button
          type="submit"
          className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-md"
        >
          Save
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white text-sm rounded-md"
        >
          Cancel
        </button>
      </div>
    </form>
  );
};

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
  const [isGeneratingFeatures, setIsGeneratingFeatures] = useState(false);
  const [featureEngineeringError, setFeatureEngineeringError] = useState<string | null>(null);

  // State für Feature Engineering Interaktion
  const [selectedGeneratedFeatures, setSelectedGeneratedFeatures] = useState<Set<number>>(new Set());
  const [customFeatures, setCustomFeatures] = useState<any[]>([]);
  const [showAddFeatureForm, setShowAddFeatureForm] = useState(false);
  const [editingFeatureIndex, setEditingFeatureIndex] = useState<number | null>(null);

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
          throw new Error("No columns found in the file. Please provide a valid file with headers.");
        }

        // Automatischen Projektnamen vorschlagen
        if (!projectName && analysis.fileName) {
          const fileName = analysis.fileName.replace(/\.[^/.]+$/, ""); // Remove extension
          setProjectName(`${fileName} - ML Model`);
        }

      } catch (error) {
        setAnalysisError(error instanceof Error ? error.message : "Error during data analysis.");
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
      // Basis-Analyse für manipulierte Daten (ohne Feature Engineering)
      // Verwende selectedTargetVariable wenn vorhanden, sonst userPreferences
      const targetVariableHint = selectedTargetVariable
        ? `The target variable is: ${selectedTargetVariable}. ${userPreferences || ''}`
        : userPreferences;

      const result = await apiService.analyzeData(
        csvAnalysis.filePath,
        excludedColumns,
        excludedFeatures,
        targetVariableHint,
      );

      console.log('AnalyzeData result:', result);

      setManipulatedData(result.analysis);

      // WICHTIG: Setze LLM-Empfehlungen (targetVariable, algorithm, modelType, etc.)
      if (result.recommendations) {
        setLlmRecommendations(result.recommendations);
        console.log('✅ LLM-Empfehlungen gesetzt:', result.recommendations);

        // Setze selectedTargetVariable wenn in Empfehlungen vorhanden und noch nicht gesetzt
        if (result.recommendations.targetVariable && !selectedTargetVariable) {
          setSelectedTargetVariable(result.recommendations.targetVariable);
        }
      }

      // Basis-Features setzen
      let features = [];
      if (result.analysis && result.analysis.columns) {
        features = result.analysis.columns.filter((f: string) =>
          !excludedColumns.includes(f) && !excludedFeatures.includes(f)
        );
      }
      setAvailableFeatures(features);

    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : "Error during data manipulation.");
    } finally {
      setIsProcessingData(false);
    }
  };

  const handleFeatureEngineering = async () => {
    if (!csvAnalysis) return;

    setIsGeneratingFeatures(true);
    setFeatureEngineeringError(null);

    try {
      // Feature Engineering API aufrufen - NUR Feature-Generierung, keine ML-Konfiguration
      // Verwende selectedTargetVariable wenn vorhanden
      const targetVariableHint = selectedTargetVariable
        ? `The target variable is: ${selectedTargetVariable}. ${userPreferences || ''}`
        : userPreferences;

      const result = await apiService.generateFeatures(
        csvAnalysis.filePath,
        excludedColumns,
        excludedFeatures,
        targetVariableHint,
      );

      console.log('🔍 Feature Engineering Result (complete):', result);
      console.log('🔍 Feature Engineering Result.generatedFeatures:', result.generatedFeatures);
      console.log('🔍 Feature Engineering Result.reasoning:', result.reasoning);
      console.log('🔍 Result Keys:', Object.keys(result));

      // API gibt { success: true, generatedFeatures, reasoning } zurück
      // Extrahiere die Features aus der Antwort
      const generatedFeatures = result.generatedFeatures || (result.success ? [] : []);

      console.log('🔍 Generierte Features (verarbeitet):', generatedFeatures);
      console.log('🔍 Number of generated features:', generatedFeatures.length);
      console.log('🔍 Ist Array?', Array.isArray(generatedFeatures));

      // Setze die generierten Features in llmRecommendations
      // WICHTIG: Behalte bestehende Empfehlungen (targetVariable, algorithm, modelType, etc.) bei
      const newRecommendations = {
        ...llmRecommendations, // Behalte bestehende Empfehlungen
        generatedFeatures: generatedFeatures,
        reasoning: result.reasoning || llmRecommendations?.reasoning || 'Feature Engineering completed'
      };

      console.log('🔍 New recommendations:', newRecommendations);
      setLlmRecommendations(newRecommendations);

      // Alle generierten Features standardmäßig auswählen
      if (generatedFeatures.length > 0) {
        setSelectedGeneratedFeatures(new Set(generatedFeatures.map((_: any, index: number) => index)));
        console.log(`✅ ${generatedFeatures.length} Features generated:`, generatedFeatures);
      } else {
        console.log('ℹ️ No generated features proposed');
        console.log('🔍 Result structure:', JSON.stringify(result, null, 2));
      }

    } catch (error) {
      console.error('Feature Engineering Error:', error);
      setFeatureEngineeringError(error instanceof Error ? error.message : "Error during Feature Engineering.");
    } finally {
      setIsGeneratingFeatures(false);
    }
  };

  const toggleGeneratedFeature = (index: number) => {
    setSelectedGeneratedFeatures(prev => {
      const newSet = new Set(prev);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  };

  const handleAddCustomFeature = (feature: any) => {
    setCustomFeatures(prev => [...prev, feature]);
    setShowAddFeatureForm(false);
  };

  const handleEditCustomFeature = (index: number, updatedFeature: any) => {
    setCustomFeatures(prev => {
      const newFeatures = [...prev];
      newFeatures[index] = updatedFeature;
      return newFeatures;
    });
    setEditingFeatureIndex(null);
  };

  const handleDeleteCustomFeature = (index: number) => {
    setCustomFeatures(prev => prev.filter((_, i) => i !== index));
  };

  // Berechne finale Feature-Liste
  const getFinalFeatures = () => {
    const selectedGenerated = llmRecommendations?.generatedFeatures?.filter((_: any, index: number) =>
      selectedGeneratedFeatures.has(index)
    ) || [];
    return [...selectedGenerated, ...customFeatures];
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
      // Finale Features zusammenstellen
      const finalFeaturesList = getFinalFeatures();

      // Projekt mit benutzerdefinierten Einstellungen erstellen
      const finalRecommendations = {
        ...llmRecommendations,
        generatedFeatures: finalFeaturesList, // Verwende die finalen ausgewählten Features
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

      // Verfügbare Spalten für Feature Engineering löschen (Cleanup)
      setAvailableFeatures([]);
      setManipulatedData(null);
      setLlmRecommendations(null);
      setSelectedGeneratedFeatures(new Set());
      setCustomFeatures([]);

      onSubmit(project);
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : "Fehler beim Erstellen des Projekts.");
    } finally {
      setIsCreating(false);
    }
  };

  const nextStep = async () => {
    if (step === 2) {
      // Beim Übergang von Schritt 2 zu 3: Basis-Datenmanipulation durchführen
      await handleDataManipulation();
    }
    setStep(s => s + 1);
  };

  // Automatisches Feature Engineering beim Erscheinen von Schritt 3
  useEffect(() => {
    if (step === 3 && csvAnalysis && !isGeneratingFeatures && !llmRecommendations?.generatedFeatures) {
      console.log('🔄 Auto-starting Feature Engineering in Step 3...');
      // Kleine Verzögerung, damit der UI-State aktualisiert werden kann
      const timer = setTimeout(() => {
        handleFeatureEngineering();
      }, 100);
      return () => clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, csvAnalysis]);

  const prevStep = () => setStep(s => s - 1);

  const renderStep1 = () => (
    <div>
      <h3 className="text-xl font-semibold text-white mb-2">📁 Upload file & basic analysis</h3>
      <p className="text-sm text-slate-400 mb-6">
        Upload your data and let us perform a basic analysis.
      </p>

      <div className="space-y-6">
        <div>
          <h4 className="text-lg font-medium text-white mb-2">Project name</h4>
          <input
            type="text"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            placeholder="e.g. Customer churn prediction"
            className="w-full bg-slate-700 border-slate-600 rounded-md p-3 text-white focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div>
          <h4 className="text-lg font-medium text-white mb-2">Upload file</h4>
          <div
            className="mt-1 flex justify-center px-6 pt-8 pb-8 border-2 border-slate-600 border-dashed rounded-lg cursor-pointer hover:border-blue-500 transition-all duration-200 hover:bg-slate-800/30"
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="space-y-2 text-center">
              <CloudUploadIcon className="mx-auto h-16 w-16 text-slate-400" />
              <div className="text-slate-400">
                <p className="text-lg">{dataSource ? `📄 ${dataSource.name}` : 'Click for file upload'}</p>
                <p className="text-sm">{dataSource ? `${(dataSource.size / 1024).toFixed(2)} KB` : 'Max. 10MB'}</p>
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
                <h4 className="text-blue-400 font-medium">📊 Analyze data structure...</h4>
                <p className="text-blue-300 text-sm">Recognize columns and data types</p>
              </div>
            </div>
          </div>
        )}

        {csvAnalysis && (
          <div className="bg-slate-900/50 p-6 rounded-lg border border-slate-700">
            <h4 className="font-semibold text-green-400 mb-4">📊 Data overview</h4>
            <div className="grid grid-cols-2 gap-6 text-sm">
              <div>
                <p className="text-slate-400">Rows: <span className="text-white font-mono">{csvAnalysis.rowCount.toLocaleString()}</span></p>
                <p className="text-slate-400">Columns: <span className="text-white font-mono">{csvAnalysis.columns.length}</span></p>
              </div>
              <div>
                <p className="text-slate-400">Numerical columns: <span className="text-blue-400 font-mono">{Object.values(csvAnalysis.dataTypes).filter(t => t === 'numeric').length}</span></p>
                <p className="text-slate-400">Categorical columns: <span className="text-green-400 font-mono">{Object.values(csvAnalysis.dataTypes).filter(t => t === 'categorical').length}</span></p>
              </div>
            </div>

            <div className="mt-4">
              <h5 className="text-sm font-medium text-slate-300 mb-2">Available columns:</h5>
              <div className="flex flex-wrap gap-2">
                {csvAnalysis.columns.map((column) => (
                  <span
                    key={column}
                    className={`px-2 py-1 rounded text-xs ${csvAnalysis.dataTypes[column] === 'numeric'
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
      <h3 className="text-xl font-semibold text-white mb-2">✂️ Data manipulation & feature selection</h3>
      <p className="text-sm text-slate-400 mb-6">
        Remove unwanted columns and features before the AI analysis starts.
      </p>

      <div className="space-y-6">
        {/* Target Variable Auswahl - IMMER sichtbar */}
        <div className="bg-slate-900/50 p-6 rounded-lg border border-slate-700">
          <h4 className="font-semibold text-white mb-2">🎯 Select Target Variable</h4>
          <p className="text-sm text-slate-400 mb-4">
            Select the variable that should be predicted by the model.
          </p>
          {csvAnalysis ? (
            <>
              <select
                value={selectedTargetVariable}
                onChange={(e) => setSelectedTargetVariable(e.target.value)}
                className="w-full bg-slate-700 border-slate-600 rounded-md p-3 text-white focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">-- Select target variable --</option>
                {csvAnalysis.columns
                  .filter(col => !excludedColumns.includes(col) && !excludedFeatures.includes(col))
                  .map((column) => (
                    <option key={column} value={column}>
                      {column} ({csvAnalysis.dataTypes[column]})
                    </option>
                  ))}
              </select>
              {selectedTargetVariable && (
                <p className="text-xs text-green-400 mt-2">
                  ✅ Selected: <strong>{selectedTargetVariable}</strong>
                </p>
              )}
            </>
          ) : (
            <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-lg p-3">
              <p className="text-yellow-300 text-sm">
                ⚠️ Please upload a file in Step 1 first to select the target variable.
              </p>
            </div>
          )}
        </div>

        {/* Anforderungen an die KI */}
        <div className="bg-slate-900/50 p-6 rounded-lg border border-slate-700">
          <h4 className="font-semibold text-white mb-2">📝 Additional Requirements (Optional)</h4>
          <p className="text-sm text-slate-400 mb-4">
            Describe here additional requirements, e.g. preferred features, preferred model type/algorithm, metrics or other constraints.
          </p>
          <textarea
            value={userPreferences}
            onChange={(e) => setUserPreferences(e.target.value)}
            placeholder="Example: I prefer using Random Forest algorithm. Focus on interpretability."
            className="w-full bg-slate-700 border-slate-600 rounded-md p-3 text-white focus:ring-blue-500 focus:border-blue-500 min-h-[100px]"
          />
          <p className="text-xs text-slate-400 mt-2">These hints will be considered when creating the recommendations.</p>
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
              <h4 className="font-semibold text-white mb-4">🎯 Columns to exclude</h4>
              <p className="text-sm text-slate-400 mb-4">
                Exclude columns that are not used for the project.
              </p>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {csvAnalysis.columns
                  .filter(col => !excludedColumns.includes(col))
                  .map((feature) => (
                    <div
                      key={feature}
                      onClick={() => toggleFeatureExclusion(feature)}
                      className={`cursor-pointer p-3 rounded-lg border transition-all ${excludedFeatures.includes(feature)
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
                    <strong>Excluded features ({excludedFeatures.length}):</strong> {excludedFeatures.join(', ')}
                  </p>
                </div>
              )}
            </div>

            <div className="bg-blue-900/30 border border-blue-500/50 rounded-lg p-6">
              <h4 className="text-blue-400 font-medium mb-2">📊 Overview of manipulated data</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-400">Remaining columns: <span className="text-white font-mono">{csvAnalysis.columns.length - excludedColumns.length}</span></p>
                  <p className="text-gray-400">Available features: <span className="text-blue-400 font-mono">{csvAnalysis.columns.length - excludedColumns.length - excludedFeatures.length}</span></p>
                </div>
                <div>
                  <p className="text-gray-400">Excluded columns: <span className="text-red-400 font-mono">{excludedColumns.length}</span></p>
                  <p className="text-gray-400">Excluded features: <span className="text-orange-400 font-mono">{excludedFeatures.length}</span></p>
                </div>
              </div>

            </div>
          </>
        )}
      </div>
    </div>
  );

  const renderStep3 = () => (
    <div>
      <h3 className="text-xl font-semibold text-white mb-2">✨ Feature Engineering</h3>

      <div className="space-y-6">
        {/* Debug: Zeige aktuellen State */}
        {process.env.NODE_ENV === 'development' && llmRecommendations && (
          <div className="bg-gray-900/50 p-3 rounded text-xs text-gray-400 mb-4">
            <p>Debug: llmRecommendations vorhanden: {llmRecommendations ? 'Ja' : 'Nein'}</p>
            <p>Debug: generatedFeatures vorhanden: {llmRecommendations.generatedFeatures ? 'Ja' : 'Nein'}</p>
            <p>Debug: Ist Array: {Array.isArray(llmRecommendations.generatedFeatures) ? 'Ja' : 'Nein'}</p>
            <p>Debug: Anzahl: {llmRecommendations.generatedFeatures?.length || 0}</p>
          </div>
        )}

        {/* Generierte Features anzeigen mit Interaktionsmöglichkeiten */}
        {llmRecommendations?.generatedFeatures && Array.isArray(llmRecommendations.generatedFeatures) && llmRecommendations.generatedFeatures.length > 0 && (
          <div className="bg-gradient-to-r from-green-900/40 to-blue-900/40 border border-green-500/50 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-green-400 font-bold flex items-center">
                ✨ AI-generated features ({llmRecommendations.generatedFeatures.length})
              </h4>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setSelectedGeneratedFeatures(new Set(llmRecommendations.generatedFeatures.map((_: any, i: number) => i)))}
                  className="text-xs px-2 py-1 bg-green-700 hover:bg-green-600 text-white rounded"
                >
                  Select all
                </button>
                <button
                  onClick={() => setSelectedGeneratedFeatures(new Set())}
                  className="text-xs px-2 py-1 bg-slate-700 hover:bg-slate-600 text-white rounded"
                >
                  Deselect all
                </button>
              </div>
            </div>
            {llmRecommendations.reasoning && (
              <div className="bg-green-900/20 border border-green-500/30 rounded-lg p-3 mb-4">
                <p className="text-green-200 text-sm italic">
                  💡 <strong>AI Strategy:</strong> {llmRecommendations.reasoning}
                </p>
              </div>
            )}
            <div className="bg-green-900/20 border border-green-500/30 rounded-lg p-3 mb-4">
              <p className="text-green-200 text-sm">
                ✅ <strong>{selectedGeneratedFeatures.size} von {llmRecommendations.generatedFeatures.length}</strong> Features selected
              </p>
              <p className="text-green-300 text-xs mt-1">
                Select the features that should be used for training. You can change them at any time.
              </p>
            </div>
            <div className="space-y-4">
              {llmRecommendations.generatedFeatures.map((feature: any, index: number) => {
                const isSelected = selectedGeneratedFeatures.has(index);
                return (
                  <div
                    key={index}
                    className={`bg-green-900/20 border rounded-lg p-5 transition-all ${isSelected
                      ? 'border-green-500/50 bg-green-900/30'
                      : 'border-green-500/20 bg-green-900/10 opacity-60'
                      }`}
                  >
                    <div className="flex items-start gap-3">
                      {/* Checkbox */}
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggleGeneratedFeature(index)}
                        className="mt-1 w-5 h-5 text-green-600 bg-slate-700 border-slate-600 rounded focus:ring-green-500"
                      />
                      <div className="flex-1">
                        {/* Feature Name */}
                        <div className="flex items-center mb-2">
                          <span className="text-green-400 font-bold text-lg mr-2">#{index + 1}</span>
                          <p className="text-green-200 text-base font-bold">{feature.name || `Feature ${index + 1}`}</p>
                        </div>

                        {/* Description */}
                        {feature.description && (
                          <p className="text-green-300 text-sm mb-3 leading-relaxed">{feature.description}</p>
                        )}

                        {/* Kombinierte Spalten */}
                        {feature.formula && (() => {
                          const columnMatches = feature.formula.match(/data\['([^']+)'\]/g) || [];
                          const columns = columnMatches.map((match: string) => match.replace(/data\['|'\]/g, ''));
                          return columns.length > 0 ? (
                            <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-3 mb-3">
                              <p className="text-blue-300 text-xs font-semibold mb-2">🔗 Kombinierte Spalten:</p>
                              <div className="flex flex-wrap gap-2">
                                {columns.map((col: string, colIndex: number) => (
                                  <span key={colIndex} className="bg-blue-800/50 text-blue-200 text-xs px-2 py-1 rounded border border-blue-500/50">
                                    {col}
                                  </span>
                                ))}
                              </div>
                            </div>
                          ) : null;
                        })()}

                        {/* Formula */}
                        {feature.formula && (
                          <div className="bg-green-900/40 border border-green-500/50 p-4 rounded-lg mb-3">
                            <p className="text-green-400 text-xs font-semibold mb-2 uppercase tracking-wide">🐍 Python Formel:</p>
                            <p className="text-green-300 text-sm font-mono break-all bg-green-950/50 p-2 rounded border border-green-500/30">
                              {feature.formula}
                            </p>
                          </div>
                        )}

                        {/* Reasoning */}
                        {feature.reasoning && (
                          <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-3">
                            <p className="text-blue-300 text-xs leading-relaxed">
                              <span className="font-semibold">💡 Reasoning:</span> {feature.reasoning}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Manuell hinzugefügte Features */}
        <div className="bg-slate-900/50 p-6 rounded-lg border border-slate-700">
          <div className="flex items-center justify-between mb-4">
            <h4 className="font-semibold text-white">➕ Eigene Features hinzufügen</h4>
            <button
              onClick={() => setShowAddFeatureForm(!showAddFeatureForm)}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-md transition-colors"
            >
              {showAddFeatureForm ? 'Abbrechen' : '+ Neues Feature'}
            </button>
          </div>

          {showAddFeatureForm && (
            <AddFeatureForm
              availableColumns={csvAnalysis?.columns.filter((col: string) => !excludedColumns.includes(col) && !excludedFeatures.includes(col)) || []}
              onAdd={handleAddCustomFeature}
              onCancel={() => setShowAddFeatureForm(false)}
            />
          )}

          {customFeatures.length > 0 && (
            <div className="mt-4 space-y-3">
              <p className="text-slate-300 text-sm font-medium mb-2">Deine eigenen Features ({customFeatures.length}):</p>
              {customFeatures.map((feature: any, index: number) => (
                <div key={index} className="bg-slate-800/50 border border-slate-600 rounded-lg p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center mb-2">
                        <span className="text-blue-400 font-bold mr-2">#{index + 1}</span>
                        <p className="text-blue-200 font-bold">{feature.name}</p>
                      </div>
                      {feature.description && (
                        <p className="text-slate-300 text-sm mb-2">{feature.description}</p>
                      )}
                      {feature.formula && (
                        <p className="text-slate-400 text-xs font-mono bg-slate-900/50 p-2 rounded">{feature.formula}</p>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => setEditingFeatureIndex(index)}
                        className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded"
                      >
                        Bearbeiten
                      </button>
                      <button
                        onClick={() => handleDeleteCustomFeature(index)}
                        className="text-xs px-2 py-1 bg-red-600 hover:bg-red-700 text-white rounded"
                      >
                        Löschen
                      </button>
                    </div>
                  </div>
                  {editingFeatureIndex === index && (
                    <EditFeatureForm
                      feature={feature}
                      availableColumns={csvAnalysis?.columns.filter((col: string) => !excludedColumns.includes(col) && !excludedFeatures.includes(col)) || []}
                      onSave={(updated) => handleEditCustomFeature(index, updated)}
                      onCancel={() => setEditingFeatureIndex(null)}
                    />
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Vorschau der finalen Features */}
        {getFinalFeatures().length > 0 && (
          <div className="bg-gradient-to-r from-purple-900/40 to-indigo-900/40 border border-purple-500/50 rounded-lg p-6">
            <h4 className="text-purple-400 font-bold mb-4 flex items-center">
              📋 Preview: Final feature list ({getFinalFeatures().length})
            </h4>
            <p className="text-purple-200 text-sm mb-4">
              These features will be used for training:
            </p>
            <div className="space-y-2">
              {getFinalFeatures().map((feature: any, index: number) => (
                <div key={index} className="bg-purple-900/20 border border-purple-500/30 rounded-lg p-3">
                  <div className="flex items-center gap-2">
                    <span className="text-purple-400 font-bold">#{index + 1}</span>
                    <p className="text-purple-200 font-medium">{feature.name}</p>
                    {feature.formula && (
                      <span className="text-purple-400 text-xs font-mono ml-auto">{feature.formula}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {!llmRecommendations && !isGeneratingFeatures && (
          <div className="bg-blue-900/30 border border-blue-500/50 rounded-lg p-6">
            <p className="text-blue-300 text-sm text-center">
              Click on "🚀 Start feature engineering" above to start the AI analysis.
            </p>
          </div>
        )}

        {llmRecommendations && (!llmRecommendations.generatedFeatures || !Array.isArray(llmRecommendations.generatedFeatures) || llmRecommendations.generatedFeatures.length === 0) && !isGeneratingFeatures && (
          <div className="bg-yellow-900/30 border border-yellow-500/50 rounded-lg p-6">
            <p className="text-yellow-300 text-sm text-center">
              ℹ️ The AI has not proposed any new features. The existing columns will be used directly.
            </p>
            {llmRecommendations.reasoning && (
              <p className="text-yellow-200 text-xs text-center mt-2 italic">
                {llmRecommendations.reasoning}
              </p>
            )}
          </div>
        )}

        {/* Verfügbare Spalten anzeigen */}
        {/* {csvAnalysis && (
          <div className="bg-slate-900/50 p-6 rounded-lg border border-slate-700">
            <h4 className="font-semibold text-white mb-3">📊 Available columns for feature engineering</h4>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {csvAnalysis.columns
                .filter(col => !excludedColumns.includes(col) && !excludedFeatures.includes(col))
                .map((column) => (
                  <div key={column} className="bg-slate-800/50 border border-slate-600 rounded p-2">
                    <p className="text-slate-300 text-xs font-medium">{column}</p>
                    <p className="text-slate-400 text-xs">{csvAnalysis.dataTypes[column]}</p>
                  </div>
                ))}
            </div>
          </div>
        )} */}
      </div>
    </div>
  );

  const renderStep4 = () => (
    <div>
      <h3 className="text-xl font-semibold text-white mb-2">⚙️ Hyperparameters & ML configuration</h3>
      <p className="text-sm text-gray-400 mb-6">
        Select algorithm, target variable and other machine learning parameters.
      </p>

      <div className="space-y-6">
        <div className="bg-gray-900/50 p-6 rounded-lg border border-gray-700">
          <h4 className="font-semibold text-white mb-4">🎯 Select target variable</h4>
          <p className="text-sm text-gray-400 mb-4">
            Select the variable that should be predicted.
          </p>

          <select
            value={selectedTargetVariable}
            onChange={(e) => setSelectedTargetVariable(e.target.value)}
            className="w-full bg-gray-700 border-gray-600 rounded-md p-3 text-white focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="">-- Select target variable --</option>
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
                💡 <strong>Recommendation:</strong> {llmRecommendations.targetVariable}
                {llmRecommendations.targetVariable !== selectedTargetVariable && selectedTargetVariable && (
                  <button
                    onClick={() => setSelectedTargetVariable(llmRecommendations.targetVariable)}
                    className="ml-2 text-xs px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700"
                  >
                    Accept
                  </button>
                )}
              </p>
            </div>
          )}
        </div>

        <div className="bg-gray-900/50 p-6 rounded-lg border border-gray-700">
          <h4 className="font-semibold text-white mb-4">🤖 Select model type</h4>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div
              onClick={() => setSelectedModelType('Classification')}
              className={`p-4 rounded-lg border cursor-pointer transition-all ${selectedModelType === 'Classification'
                ? 'bg-blue-900/40 border-blue-500 text-blue-300'
                : 'bg-gray-800/50 border-gray-600 text-gray-300 hover:bg-gray-700/50'
                }`}
            >
              <h5 className="font-medium">🏷️ Classification</h5>
              <p className="text-sm mt-1 opacity-75">
                Predict categories (e.g. spam/not spam, cancer diagnosis)
              </p>
            </div>

            <div
              onClick={() => setSelectedModelType('Regression')}
              className={`p-4 rounded-lg border cursor-pointer transition-all ${selectedModelType === 'Regression'
                ? 'bg-purple-900/40 border-purple-500 text-purple-300'
                : 'bg-gray-800/50 border-gray-600 text-gray-300 hover:bg-gray-700/50'
                }`}
            >
              <h5 className="font-medium">📈 Regression</h5>
              <p className="text-sm mt-1 opacity-75">
                Predict numerical values (e.g. prices, temperatures)
              </p>
            </div>
          </div>

          {llmRecommendations?.modelType && (
            <div className="p-3 bg-green-900/20 border border-green-500/30 rounded">
              <p className="text-green-300 text-sm">
                💡 <strong>Recommendation:</strong> {llmRecommendations.modelType}
                {llmRecommendations.modelType !== selectedModelType && selectedModelType && (
                  <button
                    onClick={() => setSelectedModelType(llmRecommendations.modelType)}
                    className="ml-2 text-xs px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700"
                  >
                    Accept
                  </button>
                )}
              </p>
            </div>
          )}
        </div>

        {selectedModelType && (
          <div className="bg-gray-900/50 p-6 rounded-lg border border-gray-700">
            <h4 className="font-semibold text-white mb-4">🧮 Select algorithm</h4>
            <p className="text-sm text-gray-400 mb-4">
              Select the machine learning algorithm for your {selectedModelType}-problem.
            </p>

            <select
              value={selectedAlgorithm}
              onChange={(e) => setSelectedAlgorithm(e.target.value)}
              className="w-full bg-gray-700 border-gray-600 rounded-md p-3 text-white focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">-- Select algorithm --</option>
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
                  💡 <strong>Recommendation:</strong> {ALGORITHMS[llmRecommendations.algorithm]?.name || llmRecommendations.algorithm}
                  {llmRecommendations.algorithm !== selectedAlgorithm && selectedAlgorithm && (
                    <button
                      onClick={() => setSelectedAlgorithm(llmRecommendations.algorithm)}
                      className="ml-2 text-xs px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700"
                    >
                      Accept
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
              🎯 Recommendations
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm mb-4">
              <div className="space-y-3">
                <div>
                  <p className="text-gray-300 font-medium">Recommended features ({(availableFeatures || []).filter(f => f !== selectedTargetVariable && !excludedFeatures.includes(f)).length}):</p>
                  <div className="text-green-300 text-xs max-h-20 overflow-y-auto">
                    {(availableFeatures || [])
                      .filter(f => f !== selectedTargetVariable && !excludedFeatures.includes(f))
                      .join(', ') || 'No features available'}
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <div>
                  <p className="text-gray-300 font-medium">Reasoning:</p>
                  <div className="text-green-300 text-xs max-h-20 overflow-y-auto">
                    {llmRecommendations.reasoning || 'No reasoning available'}
                  </div>
                </div>
              </div>
            </div>

            {/* Generierte Features anzeigen */}
            {llmRecommendations.generatedFeatures && llmRecommendations.generatedFeatures.length > 0 && (
              <div className="mt-4 pt-4 border-t border-green-500/30">
                <p className="text-gray-300 font-medium mb-3">
                  ✨ Generierte Features ({llmRecommendations.generatedFeatures.length}):
                </p>
                <div className="space-y-3">
                  {llmRecommendations.generatedFeatures.map((feature: any, index: number) => (
                    <div key={index} className="bg-green-900/20 border border-green-500/30 rounded-lg p-3">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <p className="text-green-300 font-semibold text-sm">{feature.name}</p>
                          <p className="text-green-200 text-xs mt-1">{feature.description}</p>
                          <p className="text-green-400 text-xs mt-2 font-mono bg-green-900/30 p-2 rounded">
                            {feature.formula}
                          </p>
                          {feature.reasoning && (
                            <p className="text-green-300 text-xs mt-2 italic">
                              💡 {feature.reasoning}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <p className="text-green-300 text-xs mt-3 italic">
                  Diese Features werden automatisch aus vorhandenen Spalten generiert, um die Modellleistung zu verbessern.
                </p>
              </div>
            )}
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

  const renderStep5 = () => (
    <div>
      <h3 className="text-xl font-semibold text-white mb-4">✅ Confirm & start training</h3>
      <p className="text-gray-400 mb-6">Check your configuration before training.</p>

      <div className="space-y-6">
        <div className="bg-gray-900/50 p-6 rounded-lg border border-gray-700">
          <h4 className="font-semibold text-white mb-4">📋 Project summary</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400 font-medium">Project name:</span>
                <span className="text-white">{projectName}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400 font-medium">Data source:</span>
                <span className="text-white">{dataSource?.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400 font-medium">Rows:</span>
                <span className="text-blue-400">{csvAnalysis?.rowCount.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400 font-medium">Original columns:</span>
                <span className="text-white">{csvAnalysis?.columns.length}</span>
              </div>
            </div>

            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400 font-medium">Model type:</span>
                <span className="text-purple-400">{selectedModelType}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400 font-medium">Algorithm:</span>
                <span className="text-blue-400">{ALGORITHMS[selectedAlgorithm]?.name || selectedAlgorithm}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400 font-medium">Target variable:</span>
                <span className="text-green-400">{selectedTargetVariable}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400 font-medium">Used features:</span>
                <span className="text-cyan-400">
                  {(availableFeatures || []).filter(f => f !== selectedTargetVariable && !excludedFeatures.includes(f)).length}
                </span>
              </div>
            </div>
          </div>
        </div>

        {excludedColumns.length > 0 && (
          <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
            <h5 className="text-red-400 font-medium mb-2">🗑️ Excluded columns ({excludedColumns.length})</h5>
            <p className="text-red-300 text-sm">{excludedColumns.join(', ')}</p>
          </div>
        )}

        {excludedFeatures.length > 0 && (
          <div className="bg-orange-900/20 border border-orange-500/30 rounded-lg p-4">
            <h5 className="text-orange-400 font-medium mb-2">🚫 Excluded features ({excludedFeatures.length})</h5>
            <p className="text-orange-300 text-sm">{excludedFeatures.join(', ')}</p>
          </div>
        )}

        <div className="bg-blue-900/40 border border-blue-500/50 rounded-lg p-6">
          <h5 className="text-blue-400 font-medium mb-2">🔬 Used features for ML</h5>
          <div className="text-blue-300 text-sm">
            {(availableFeatures || [])
              .filter(f => f !== selectedTargetVariable && !excludedFeatures.includes(f))
              .join(', ') || 'No features available'}
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
      title: 'Upload & Analysis',
      content: renderStep1(),
      canProceed: !!(projectName && csvAnalysis && !isAnalyzing)
    },
    {
      num: 2,
      title: 'Data manipulation',
      content: renderStep2(),
      canProceed: !!(csvAnalysis)
    },
    {
      num: 3,
      title: 'Feature Engineering',
      content: renderStep3(),
      canProceed: true // Optional, kann übersprungen werden (Features werden trotzdem generiert wenn vorhanden)
    },
    {
      num: 4,
      title: 'ML configuration',
      content: renderStep4(),
      canProceed: !!(selectedTargetVariable && selectedAlgorithm && selectedModelType && !isGeneratingFeatures)
    },
    {
      num: 5,
      title: 'Confirm & start training',
      content: renderStep5(),
      canProceed: true
    },
  ];

  return (
    <div className="max-w-4xl mx-auto bg-slate-800 rounded-lg shadow-xl p-6 sm:p-8 animate-fade-in">
      <div className="flex justify-between items-start mb-6">
        <div>
          <h2 className="text-3xl font-bold text-white">ML Wizard</h2>
          <p className="text-slate-400">Step {step} of {steps.length}: {steps[step - 1]?.title || 'Unknown'}</p>
        </div>
        <button onClick={onBack} className="text-slate-400 hover:text-white text-xl">&times;</button>
      </div>

      {/* Fortschrittsanzeige */}
      <div className="flex items-center justify-between mb-6 overflow-x-auto pb-2">
        {steps.map((stepInfo, index) => (
          <div key={index} className="flex items-center flex-shrink-0">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${index + 1 < step
              ? 'bg-emerald-600 text-white'
              : index + 1 === step
                ? 'bg-blue-600 text-white'
                : 'bg-slate-600 text-slate-300'
              }`}>
              {index + 1 < step ? '✓' : index + 1}
            </div>
            <span className={`ml-2 text-xs whitespace-nowrap ${index + 1 <= step ? 'text-white' : 'text-slate-400'
              }`}>
              {stepInfo.title}
            </span>
            {index < steps.length - 1 && (
              <div className={`mx-2 md:mx-4 h-0.5 w-8 md:w-16 flex-shrink-0 ${index + 1 < step ? 'bg-emerald-600' : 'bg-slate-600'
                }`} />
            )}
          </div>
        ))}
      </div>

      <div className="py-6">
        {steps[step - 1].content}
      </div>

      <div className="flex justify-between items-center pt-6 border-t border-slate-700">
        <button
          onClick={step === 1 ? onBack : prevStep}
          className="px-6 py-2 border border-slate-600 text-sm font-medium rounded-md text-slate-300 hover:bg-slate-700 transition-colors"
        >
          {step === 1 ? 'Cancel' : 'Back'}
        </button>

        {step < steps.length ? (
          <button
            onClick={nextStep}
            disabled={!steps[step - 1].canProceed || isAnalyzing || isProcessingData || isGeneratingFeatures}
            className="px-6 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 disabled:bg-slate-500 disabled:cursor-not-allowed transition-colors"
          >
            {(step === 2 && isProcessingData) || (step === 3 && isGeneratingFeatures) ? (
              <div className="flex items-center space-x-2">
                <Spinner size="sm" />
                <span>Analyzing...</span>
              </div>
            ) : (
              'Next'
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
                <span>Project is being created...</span>
              </div>
            ) : (
              '🚀 Start training'
            )}
          </button>
        )}
      </div>
    </div>
  );
};

export default ProjectWizard;
