import React, { useState, useMemo, useCallback } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Project, ProjectStatus, PerformanceInsights as PerformanceInsightsType } from '../types';
import { apiService } from '../services/apiService';
import { ChevronLeftIcon } from './icons/ChevronLeftIcon';
import { Spinner } from './ui/Spinner';
import ErrorBoundary from './ui/ErrorBoundary';
import HyperparameterEditor from './HyperparameterEditor';

interface ProjectViewProps {
  project: Project;
  onBack: () => void;
  onProjectUpdate?: (updatedProject: Project) => void;
}

type Tab = 'predict' | 'performance' | 'data' | 'code' | 'api' | 'export';

const ProjectView: React.FC<ProjectViewProps> = ({ project, onBack, onProjectUpdate }) => {
  const [activeTab, setActiveTab] = useState<Tab>('predict');
  const [predictionInput, setPredictionInput] = useState<{ [key: string]: string }>({});
  const [predictionResult, setPredictionResult] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  
  // Code Editor States
  const [pythonCode, setPythonCode] = useState(project.pythonCode || '');
  const [isCodeModified, setIsCodeModified] = useState(false);
  const [isSavingCode, setIsSavingCode] = useState(false);
  const [isRetraining, setIsRetraining] = useState(false);
  const [codeMessage, setCodeMessage] = useState<string | null>(null);
  
  // Hyperparameter States
  const [currentHyperparameters, setCurrentHyperparameters] = useState<{ [key: string]: any }>(
    project.hyperparameters || {}
  );
  const [showHyperparameterEditor, setShowHyperparameterEditor] = useState(false);

  // Performance Insights State
  const [currentProject, setCurrentProject] = useState<Project>(project);

  // Update currentProject when project prop changes
  React.useEffect(() => {
    setCurrentProject(project);
  }, [project]);

  // Data Insights State
  const [dataStatistics, setDataStatistics] = useState<any>(null);
  const [isLoadingData, setIsLoadingData] = useState(false);
  const [dataError, setDataError] = useState<string | null>(null);

  // Performance Tab State (moved from render function)
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);

  const handleInsightsUpdate = useCallback((insights: PerformanceInsightsType) => {
    const updatedProject = { ...currentProject, performanceInsights: insights };
    setCurrentProject(updatedProject);
    
    if (onProjectUpdate) {
      onProjectUpdate(updatedProject);
    }
  }, [currentProject, onProjectUpdate]);

  const loadDataStatistics = useCallback(async () => {
    setIsLoadingData(true);
    setDataError(null);
    
    try {
      const statistics = await apiService.getDataStatistics(project.id);
      setDataStatistics(statistics);
    } catch (error) {
      setDataError(error instanceof Error ? error.message : 'Fehler beim Laden der Datenstatistiken');
    } finally {
      setIsLoadingData(false);
    }
  }, [project.id]);

  // Performance evaluation handler (moved from render function)
  const handleEvaluatePerformance = useCallback(async () => {
    setIsEvaluating(true);
    setEvaluationError(null);
    
    try {
      const response = await apiService.evaluatePerformance(project.id);
      const insights = response.insights;
      
      handleInsightsUpdate(insights);
    } catch (error) {
      setEvaluationError(error instanceof Error ? error.message : 'Evaluation fehlgeschlagen');
    } finally {
      setIsEvaluating(false);
    }
  }, [project.id, handleInsightsUpdate]);

  // Load data statistics when data tab is accessed
  React.useEffect(() => {
    if (!dataStatistics && !isLoadingData && activeTab === 'data') {
      loadDataStatistics();
    }
  }, [activeTab, dataStatistics, isLoadingData, loadDataStatistics]);

  // Initialize hyperparameters when project changes
  React.useEffect(() => {
    if (project.hyperparameters) {
      setCurrentHyperparameters(project.hyperparameters);
    }
  }, [project.hyperparameters]);

  // Initialize python code when project changes
  React.useEffect(() => {
    if (project.pythonCode) {
      setPythonCode(project.pythonCode);
      setIsCodeModified(false); // Reset modification flag when project updates
    } else if (project.status === ProjectStatus.Completed) {
      // Falls Python-Code fehlt aber Training abgeschlossen ist, versuche ihn zu laden
      loadPythonCodeFromServer();
    }
  }, [project.pythonCode, project.status]);

  // Load Python code from server if missing
  const loadPythonCodeFromServer = async () => {
    try {
      const updatedProject = await apiService.getProject(project.id);
      if (updatedProject.pythonCode) {
        setPythonCode(updatedProject.pythonCode);
        setIsCodeModified(false);
      }
    } catch (error) {
      console.error('Fehler beim Laden des Python-Codes:', error);
    }
  };

  // Utility functions for styling (moved from render functions)
  const getGradeColor = useCallback((grade: string) => {
    switch (grade) {
      case 'Excellent': return 'text-emerald-400 border-emerald-500/50 bg-emerald-900/30';
      case 'Good': return 'text-blue-400 border-blue-500/50 bg-blue-900/30';
      case 'Fair': return 'text-amber-400 border-amber-500/50 bg-amber-900/30';
      case 'Poor': return 'text-orange-400 border-orange-500/50 bg-orange-900/30';
      case 'Critical': return 'text-red-400 border-red-500/50 bg-red-900/30';
      default: return 'text-slate-400 border-slate-500/50 bg-slate-900/30';
    }
  }, []);

  const getImpactColor = useCallback((impact: string) => {
    switch (impact) {
      case 'High': return 'bg-red-900/50 text-red-400 border border-red-500/50';
      case 'Medium': return 'bg-amber-900/50 text-amber-400 border border-amber-500/50';
      case 'Low': return 'bg-emerald-900/50 text-emerald-400 border border-emerald-500/50';
      default: return 'bg-slate-900/50 text-slate-400 border border-slate-500/50';
    }
  }, []);

  const getReadinessColor = useCallback((readiness: string) => {
    switch (readiness) {
      case 'Production Ready': return 'text-emerald-400';
      case 'Needs Improvement': return 'text-amber-400';
      case 'Not Ready': return 'text-red-400';
      default: return 'text-slate-400';
    }
  }, []);
  
  const handleInputChange = (feature: string, value: string) => {
    setPredictionInput(prev => ({ ...prev, [feature]: value }));
  };

  const handlePredict = async () => {
    setIsPredicting(true);
    setPredictionError(null);
    setPredictionResult(null);
    try {
      // Konvertiere Input zu Strings für die API
      const processedInput: { [key: string]: string } = {};
      Object.entries(predictionInput).forEach(([key, value]) => {
        // Versuche als Zahl zu parsen, falls möglich
        const numberValue = parseFloat(value);
        processedInput[key] = isNaN(numberValue) ? value : numberValue.toString();
      });
      
      const response = await apiService.predict(project.id, processedInput);
      setPredictionResult(response.prediction);
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "Vorhersage fehlgeschlagen.");
    } finally {
      setIsPredicting(false);
    }
  };

  const handleCodeChange = (newCode: string) => {
    setPythonCode(newCode);
    setIsCodeModified(newCode !== project.pythonCode);
    setCodeMessage(null);
  };

  const handleSaveCode = async () => {
    if (!isCodeModified) return;
    
    setIsSavingCode(true);
    setCodeMessage(null);
    try {
      // Speichere Code und Hyperparameter
      await apiService.updateProjectCodeAndHyperparameters(project.id, pythonCode, currentHyperparameters);
      
      // Update project in parent component mit Hyperparametern
      if (onProjectUpdate) {
        onProjectUpdate({ 
          ...project, 
          pythonCode,
          hyperparameters: currentHyperparameters 
        });
      }
      
      setIsCodeModified(false);
      setCodeMessage('✅ Code und Hyperparameter erfolgreich gespeichert');
    } catch (error) {
      setCodeMessage(`❌ Fehler beim Speichern: ${error instanceof Error ? error.message : 'Unbekannter Fehler'}`);
    } finally {
      setIsSavingCode(false);
    }
  };

  const handleRetrain = async () => {
    setIsRetraining(true);
    setCodeMessage(null);
    try {
      await apiService.retrainProject(project.id);
      setCodeMessage('🚀 Re-Training gestartet! Verfolgen Sie den Fortschritt im Dashboard.');
      
      // Update project status in parent component
      if (onProjectUpdate) {
        onProjectUpdate({ ...project, status: ProjectStatus['Re-Training'] });
      }
    } catch (error) {
      setCodeMessage(`❌ Fehler beim Re-Training: ${error instanceof Error ? error.message : 'Unbekannter Fehler'}`);
    } finally {
      setIsRetraining(false);
    }
  };

  const handleResetCode = () => {
    if (project.originalPythonCode) {
      setPythonCode(project.originalPythonCode);
      setIsCodeModified(project.originalPythonCode !== project.pythonCode);
      setCodeMessage('🔄 Code auf Original zurückgesetzt');
    }
  };

  const handleHyperparametersChange = (newHyperparameters: { [key: string]: any }) => {
    // Hyperparameter korrekt konvertieren (numerische Werte als Zahlen, nicht als Strings)
    const convertedHyperparameters: { [key: string]: any } = {};
    for (const [key, value] of Object.entries(newHyperparameters)) {
      // Prüfe ob der Wert eine Zahl sein sollte
      if (typeof value === 'string' && !isNaN(Number(value)) && value.trim() !== '') {
        convertedHyperparameters[key] = Number(value);
      } else {
        convertedHyperparameters[key] = value;
      }
    }
    
    setCurrentHyperparameters(convertedHyperparameters);
    
    // Automatisch den Python-Code mit den neuen Hyperparametern aktualisieren
    const updatedCode = updateHyperparametersInCode(pythonCode, convertedHyperparameters);
    setPythonCode(updatedCode);
    setIsCodeModified(true);
  };

  const updateHyperparametersInCode = (code: string, hyperparameters: { [key: string]: any }): string => {
    // JSON-String für Hyperparameter erstellen (mit korrekten Datentypen)
    const hyperparametersJson = JSON.stringify(hyperparameters);
    
    // Suche nach verschiedenen hyperparameters-Formaten und ersetze sie
    const lines = code.split('\n');
    let found = false;
    const updatedLines = lines.map(line => {
      // Verschiedene Formate unterstützen
      if (line.includes('hyperparameters = ')) {
        found = true;
        // Prüfe ob es bereits ein JSON-String ist
        if (line.includes('"') && line.includes('{')) {
          return `    hyperparameters = "${hyperparametersJson.replace(/"/g, '\\"')}"`;
        } else {
          return `    hyperparameters = "${hyperparametersJson.replace(/"/g, '\\"')}"`;
        }
      }
      return line;
    });
    
    // Falls keine hyperparameters-Zeile gefunden wurde, füge sie hinzu
    if (!found) {
      // Suche nach der main()-Funktion und füge hyperparameters hinzu
      for (let i = 0; i < updatedLines.length; i++) {
        if (updatedLines[i].includes('def main():')) {
          // Füge hyperparameters nach der Funktionsdefinition hinzu
          updatedLines.splice(i + 2, 0, `    hyperparameters = "${hyperparametersJson.replace(/"/g, '\\"')}"`);
          break;
        }
      }
      
      // Falls main() nicht gefunden wurde, suche nach anderen Stellen
      if (!found) {
        // Suche nach der target_variable oder features Definition
        for (let i = 0; i < updatedLines.length; i++) {
          if (updatedLines[i].includes('target_variable = ') || updatedLines[i].includes('features = ')) {
            // Füge hyperparameters nach dieser Zeile hinzu
            updatedLines.splice(i + 1, 0, `    hyperparameters = "${hyperparametersJson.replace(/"/g, '\\"')}"`);
            break;
          }
        }
      }
    }
    
    return updatedLines.join('\n');
  };

  const toggleHyperparameterEditor = () => {
    setShowHyperparameterEditor(!showHyperparameterEditor);
  };

  const performanceData = useMemo(() => {
    if (!project.performanceMetrics) return [];
    
    // Debug: Zeige alle verfügbaren Metriken in der Konsole
    console.log('Available Performance Metrics:', project.performanceMetrics);
    
    return Object.entries(project.performanceMetrics).map(([name, value]) => {
      const formattedValue = typeof value === 'number' ? value.toFixed(4) : 'NaN';
      // Verbesserte Namensformatierung für bessere Lesbarkeit
      const displayName = name
        .replace(/_/g, ' ')
        .replace(/\b\w/g, (l) => l.toUpperCase())
        .replace(/Mae/g, 'MAE')
        .replace(/Mse/g, 'MSE')
        .replace(/Rmse/g, 'RMSE')
        .replace(/R2/g, 'R²');
      
      return {
        name: displayName,
        value: parseFloat(formattedValue),
        rawValue: formattedValue,
      };
    });
  }, [project.performanceMetrics]);

  const renderPredictTab = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
      <div>
        <h3 className="text-xl font-semibold text-white mb-4">Input Features</h3>
        <div className="space-y-4">
          {project.features.map(feature => (
            <div key={feature}>
              <label htmlFor={feature} className="block text-sm font-medium text-slate-300 capitalize">{feature.replace(/_/g, ' ')}</label>
              <input
                type="text"
                id={feature}
                value={predictionInput[feature] || ''}
                onChange={(e) => handleInputChange(feature, e.target.value)}
                className="mt-1 block w-full bg-slate-700 border-slate-600 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm text-white p-2"
                disabled={isPredicting}
              />
            </div>
          ))}
          <button
            onClick={handlePredict}
            disabled={isPredicting || project.status !== ProjectStatus.Completed}
            className="w-full inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-500 disabled:cursor-not-allowed"
          >
            {isPredicting ? <Spinner /> : 'Run Prediction'}
          </button>
           {project.status !== ProjectStatus.Completed && 
            <p className="text-center text-yellow-400 text-sm mt-2">Predictions are disabled for projects not in 'Completed' status.</p>}
        </div>
      </div>
      <div className="bg-slate-800 p-6 rounded-lg flex flex-col items-center justify-center">
        <h3 className="text-xl font-semibold text-white mb-4">Prediction Result</h3>
        <div className="flex-grow flex items-center justify-center w-full">
            {isPredicting && <Spinner size="lg"/>}
            {predictionError && <p className="text-red-400 text-center">{predictionError}</p>}
            {predictionResult && (
                <div className="text-center">
                    <p className="text-sm text-slate-400 mb-2">Predicted {project.targetVariable.replace(/_/g, ' ')}:</p>
                    <p className="text-4xl font-bold text-blue-400 break-all">{predictionResult}</p>
                </div>
            )}
            {!isPredicting && !predictionError && !predictionResult && <p className="text-slate-500">Result will appear here</p>}
        </div>
      </div>
    </div>
  );

  const renderPerformanceTab = () => {
    return (
      <div className="space-y-6">
        {/* KI-Performance-Analyse Evaluation Control */}
        {project.performanceMetrics && (
          <div className="bg-gradient-to-r from-slate-800/50 to-blue-900/30 border border-slate-600/50 rounded-lg p-6">
            <div className="flex justify-between items-center">
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">🤖 KI-Performance-Analyse</h3>
                <p className="text-slate-300 text-sm">
                  Lasse die Performance deines Modells intelligent vom LLM bewerten und erhalte detaillierte Insights.
                </p>
              </div>
              <button
                onClick={handleEvaluatePerformance}
                disabled={isEvaluating}
                className="px-6 py-3 bg-gradient-to-r from-blue-600 to-slate-700 hover:from-blue-700 hover:to-slate-800 disabled:from-slate-600 disabled:to-slate-700 text-white font-medium rounded-lg transition-all duration-200 flex items-center space-x-2"
              >
                {isEvaluating ? (
                  <>
                    <Spinner size="sm" />
                    <span>Analysiere...</span>
                  </>
                ) : (
                  <>
                    <span>🔍</span>
                    <span>Performance analysieren</span>
                  </>
                )}
              </button>
            </div>
            
            {evaluationError && (
              <div className="mt-4 p-4 bg-red-900/30 border border-red-500/50 rounded-lg">
                <p className="text-red-400 text-sm">❌ {evaluationError}</p>
              </div>
            )}
          </div>
        )}

        {/* KI-Performance Insights Display */}
        {currentProject.performanceInsights && (
          <div className="space-y-6">
            {/* Overall Score & Grade */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-slate-800/50 rounded-lg p-6 text-center">
                <h4 className="text-slate-400 text-sm font-medium mb-2">Gesamt-Score</h4>
                <div className="text-4xl font-bold text-white mb-2">
                  {currentProject.performanceInsights.overallScore.toFixed(1)}/10
                </div>
                <div className="w-full bg-slate-700 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${(currentProject.performanceInsights.overallScore / 10) * 100}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="bg-slate-800/50 rounded-lg p-6 text-center">
                <h4 className="text-slate-400 text-sm font-medium mb-2">Performance-Grade</h4>
                <div className={`inline-flex items-center px-4 py-2 rounded-lg border text-lg font-semibold ${getGradeColor(currentProject.performanceInsights.performanceGrade)}`}>
                  {currentProject.performanceInsights.performanceGrade}
                </div>
              </div>
            </div>

            {/* Summary */}
            <div className="bg-slate-800/50 border border-slate-600/50 rounded-lg p-6">
              <h4 className="text-slate-300 font-medium mb-3">📊 KI-Zusammenfassung</h4>
              <p className="text-slate-200">{currentProject.performanceInsights.summary}</p>
            </div>

            {/* Detailed Analysis */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="bg-slate-800/50 border border-slate-600/50 rounded-lg p-6">
                <h4 className="text-emerald-400 font-medium mb-3 flex items-center">
                  <span className="mr-2">✅</span>Stärken
                </h4>
                <ul className="space-y-2">
                  {currentProject.performanceInsights.detailedAnalysis.strengths.map((strength, index) => (
                    <li key={index} className="text-slate-200 text-sm flex items-start">
                      <span className="text-emerald-400 mr-2 mt-1">•</span>
                      {strength}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="bg-slate-800/50 border border-slate-600/50 rounded-lg p-6">
                <h4 className="text-amber-400 font-medium mb-3 flex items-center">
                  <span className="mr-2">⚠️</span>Schwächen
                </h4>
                <ul className="space-y-2">
                  {currentProject.performanceInsights.detailedAnalysis.weaknesses.map((weakness, index) => (
                    <li key={index} className="text-slate-200 text-sm flex items-start">
                      <span className="text-amber-400 mr-2 mt-1">•</span>
                      {weakness}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="bg-slate-800/50 border border-slate-600/50 rounded-lg p-6">
                <h4 className="text-blue-400 font-medium mb-3 flex items-center">
                  <span className="mr-2">🔍</span>Wichtige Erkenntnisse
                </h4>
                <ul className="space-y-2">
                  {currentProject.performanceInsights.detailedAnalysis.keyFindings.map((finding, index) => (
                    <li key={index} className="text-slate-200 text-sm flex items-start">
                      <span className="text-blue-400 mr-2 mt-1">•</span>
                      {finding}
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Improvement Suggestions */}
            <div className="bg-slate-800/50 rounded-lg p-6">
              <h4 className="text-white font-medium mb-4 flex items-center">
                <span className="mr-2">💡</span>Verbesserungsvorschläge
              </h4>
              <div className="space-y-4">
                {currentProject.performanceInsights.improvementSuggestions.map((suggestion, index) => (
                  <div key={index} className="bg-slate-700/50 rounded-lg p-4">
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <span className="text-blue-400 text-sm font-medium">{suggestion.category}</span>
                        <div className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ml-3 ${getImpactColor(suggestion.expectedImpact)}`}>
                          {suggestion.expectedImpact} Impact
                        </div>
                      </div>
                    </div>
                    <p className="text-slate-300 mb-2">{suggestion.suggestion}</p>
                    <details className="cursor-pointer">
                      <summary className="text-slate-400 hover:text-slate-300 text-sm">Umsetzung anzeigen</summary>
                      <p className="text-slate-400 text-sm mt-2 pl-4 border-l-2 border-slate-600">
                        {suggestion.implementation}
                      </p>
                    </details>
                  </div>
                ))}
              </div>
            </div>

            {/* Business Impact */}
            <div className="bg-slate-800/50 rounded-lg p-6">
              <h4 className="text-white font-medium mb-4 flex items-center">
                <span className="mr-2">🏢</span>Business Impact
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center">
                  <h5 className="text-slate-400 text-sm mb-2">Produktionsbereitschaft</h5>
                  <span className={`font-medium ${getReadinessColor(currentProject.performanceInsights.businessImpact.readiness)}`}>
                    {currentProject.performanceInsights.businessImpact.readiness}
                  </span>
                </div>
                <div className="text-center">
                  <h5 className="text-slate-400 text-sm mb-2">Risikobewertung</h5>
                  <span className={`font-medium ${currentProject.performanceInsights.businessImpact.riskAssessment === 'Low' ? 'text-emerald-400' : currentProject.performanceInsights.businessImpact.riskAssessment === 'Medium' ? 'text-amber-400' : 'text-red-400'}`}>
                    {currentProject.performanceInsights.businessImpact.riskAssessment}
                  </span>
                </div>
                <div className="text-center md:col-span-1">
                  <h5 className="text-slate-400 text-sm mb-2">Empfehlung</h5>
                  <p className="text-slate-300 text-sm">{currentProject.performanceInsights.businessImpact.recommendation}</p>
                </div>
              </div>
            </div>

            {/* Next Steps
            <div className="bg-gray-800/50 rounded-lg p-6">
              <h4 className="text-white font-medium mb-4 flex items-center">
                <span className="mr-2">🚀</span>Nächste Schritte
              </h4>
              <ol className="space-y-2">
                {currentProject.performanceInsights.nextSteps.map((step, index) => (
                  <li key={index} className="text-gray-300 flex items-start">
                    <span className="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm mr-3 mt-0.5 flex-shrink-0">
                      {index + 1}
                    </span>
                    {step}
                  </li>
                ))}
              </ol>
            </div> */}

            {/* Evaluation Metadata */}
            <div className="bg-gray-800/30 rounded-lg p-4 text-center">
              <p className="text-gray-400 text-sm">
                Evaluiert am {new Date(currentProject.performanceInsights.evaluatedAt).toLocaleString('de-DE')} 
                von {currentProject.performanceInsights.evaluatedBy} (v{currentProject.performanceInsights.version})
              </p>
            </div>
          </div>
        )}

        {/* Traditional Performance Metrics */}
        {project.performanceMetrics ? (
          <div>
            <h4 className="text-white font-medium mb-4 flex items-center">
              <span className="mr-2">📊</span>Performance-Metriken
            </h4>
            <div style={{ width: '100%', height: 400 }}>
              <ResponsiveContainer>
                <BarChart data={performanceData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="name" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                    labelStyle={{ color: '#d1d5db' }}
                  />
                  <Legend wrapperStyle={{ color: '#d1d5db' }}/>
                  <Bar dataKey="value" fill="#60a5fa" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            {/* Alle Performance-Metriken als Karten anzeigen */}
            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {performanceData.map((metric) => (
                <div key={metric.name} className="bg-gray-800/50 rounded-lg p-4">
                  <div className="flex justify-between items-center mb-2">
                    <h5 className="font-medium text-white">{metric.name}</h5>
                    <span className="text-blue-400 font-mono font-semibold">{metric.rawValue}</span>
                  </div>
                  {/* Zeige LLM-Interpretation falls verfügbar */}
                  {currentProject.performanceInsights?.metricsInterpretation?.[metric.name.toLowerCase().replace(/\s+/g, '_').replace(/²/g, '2')] && (
                    <div>
                      <p className="text-gray-300 text-sm mb-1">
                        {currentProject.performanceInsights.metricsInterpretation[metric.name.toLowerCase().replace(/\s+/g, '_').replace(/²/g, '2')].interpretation}
                      </p>
                      <p className="text-gray-400 text-xs">
                        {currentProject.performanceInsights.metricsInterpretation[metric.name.toLowerCase().replace(/\s+/g, '_').replace(/²/g, '2')].benchmarkComparison}
                      </p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <p className="text-gray-400 mb-4">Performance-Metriken sind für dieses Projekt noch nicht verfügbar.</p>
            <p className="text-gray-500 text-sm">Trainiere das Modell, um automatisch Performance-Metriken und KI-Insights zu erhalten.</p>
          </div>
        )}
      </div>
    );
  };

  const renderCodeTab = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-xl font-semibold text-white">🐍 Python Training Code</h3>
        <div className="flex space-x-3">
          {project.originalPythonCode && (
            <button
              onClick={handleResetCode}
              disabled={isSavingCode || isRetraining}
              className="px-4 py-2 border border-gray-600 text-sm font-medium rounded-md text-gray-300 hover:bg-gray-700 disabled:opacity-50"
            >
              🔄 Reset auf Original
            </button>
          )}
          <button
            onClick={handleSaveCode}
            disabled={!isCodeModified || isSavingCode}
            className="px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-500 disabled:cursor-not-allowed"
          >
            {isSavingCode ? (
              <div className="flex items-center space-x-2">
                <Spinner size="sm" />
                <span>Speichern...</span>
              </div>
            ) : (
              '💾 Code speichern'
            )}
          </button>
          <button
            onClick={handleRetrain}
            disabled={isRetraining || project.status === ProjectStatus['Re-Training']}
            className="px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isRetraining ? (
              <div className="flex items-center space-x-2">
                <Spinner size="sm" />
                <span>Re-Training...</span>
              </div>
            ) : (
              '🚀 Re-Training'
            )}
          </button>
          {!pythonCode && project.status === ProjectStatus.Completed && (
            <button
              onClick={loadPythonCodeFromServer}
              className="px-4 py-2 border border-yellow-600 text-sm font-medium rounded-md text-yellow-300 hover:bg-yellow-700 disabled:opacity-50"
            >
              🔄 Code laden
            </button>
          )}
        </div>
      </div>

      {/* Hyperparameter Editor - anzeigen wenn Training abgeschlossen ist oder Hyperparameter vorhanden sind */}
      {(project.status === ProjectStatus.Completed || project.hyperparameters) && project.algorithm && project.status !== ProjectStatus.Training && (
        <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 border border-purple-500/50 rounded-lg p-4">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h4 className="text-lg font-medium text-white">⚙️ Hyperparameter anpassen</h4>
              <p className="text-purple-200 text-sm">
                Passe die Hyperparameter für {project.algorithm} an
              </p>
            </div>
            <button
              onClick={toggleHyperparameterEditor}
              disabled={isRetraining}
              className="px-4 py-2 border border-purple-500 text-sm font-medium rounded-md text-purple-300 hover:bg-purple-700 disabled:opacity-50"
            >
              {showHyperparameterEditor ? '🔽 Ausblenden' : '⚙️ Anzeigen'}
            </button>
          </div>
          
          {showHyperparameterEditor && (
            <HyperparameterEditor
              algorithm={project.algorithm}
              currentHyperparameters={currentHyperparameters}
              onHyperparametersChange={handleHyperparametersChange}
              isRetraining={isRetraining}
            />
          )}
        </div>
      )}

      {codeMessage && (
        <div className={`p-4 rounded-lg ${codeMessage.includes('❌') ? 'bg-red-900/30 border border-red-500/50' : 'bg-green-900/30 border border-green-500/50'}`}>
          <p className={`text-sm ${codeMessage.includes('❌') ? 'text-red-400' : 'text-green-400'}`}>
            {codeMessage}
          </p>
        </div>
      )}

      <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
        <div className="bg-gray-800 px-4 py-2 border-b border-gray-700">
          <div className="flex items-center space-x-3">
            <div className="flex space-x-1">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            </div>
            <span className="text-gray-400 text-sm font-mono">training_script.py</span>
            {isCodeModified && (
              <span className="text-yellow-400 text-xs">● Ungespeicherte Änderungen</span>
            )}
          </div>
        </div>
        
        <textarea
          value={pythonCode}
          onChange={(e) => handleCodeChange(e.target.value)}
          className="w-full h-96 bg-gray-900 text-gray-300 font-mono text-sm p-4 border-0 resize-none focus:ring-0 focus:outline-none"
          style={{ fontFamily: 'Consolas, Monaco, "Courier New", monospace' }}
          placeholder="Python code wird hier angezeigt..."
        />
      </div>

      <div className="bg-blue-900/20 border border-blue-500/50 rounded-lg p-4">
        <h4 className="text-blue-400 font-medium mb-2">💡 Code-Editor Tipps</h4>
        <ul className="text-blue-200 text-sm space-y-1">
          <li>• Bearbeiten Sie den Python-Code nach Ihren Wünschen</li>
          <li>• Speichern Sie Änderungen vor dem Re-Training</li>
          <li>• Re-Training erstellt ein neues Modell mit Ihrem angepassten Code</li>
          <li>• Das alte Modell wird automatisch als Backup gespeichert</li>
        </ul>
      </div>
    </div>
  );

  const renderApiTab = () => {
    const sampleBody = project.features.reduce((acc, f) => ({ ...acc, [f]: "sample_value" }), {});
    const curlCommand = `curl -X POST 'http://localhost:3001/api/predict/${project.id}' \\
    -H 'Content-Type: application/json' \\
    -d '${JSON.stringify({ inputs: sampleBody }, null, 2)}'`; 

    return (
        <div className="space-y-6">
            <div>
                <h3 className="text-lg font-medium text-white">API Endpoint</h3>
                <div className="mt-2 p-3 bg-gray-900 rounded-md font-mono text-sm text-blue-300">
                    POST http://localhost:3001/api/predict/{project.id}
                </div>
            </div>
             <div>
                <h3 className="text-lg font-medium text-white">Example cURL Request</h3>
                <pre className="mt-2 p-4 bg-gray-900 rounded-md text-sm text-gray-300 overflow-x-auto">
                    <code>{curlCommand}</code>
                </pre>
            </div>
        </div>
    );
  };
  
  const handleDownloadModel = async () => {
    try {
      await apiService.downloadModel(project.id, project.name);
    } catch (error) {
      console.error('Fehler beim Download des Modells:', error);
      alert('Fehler beim Download des Modells. Bitte versuchen Sie es später erneut.');
    }
  };

  const renderExportTab = () => (
    <div className="space-y-6">
        <div>
            <h3 className="text-lg font-medium text-white mb-4">Trainiertes Modell herunterladen</h3>
            {project.status === ProjectStatus.Completed && project.modelArtifact ? (
                <button
                    onClick={handleDownloadModel}
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-green-500 transition-colors"
                >
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Modell herunterladen (.pkl)
                </button>
            ) : (
                <div className="text-gray-500 text-center py-4 bg-gray-800 rounded-lg">
                    {project.status === ProjectStatus.Training ? 
                        'Das Modell wird noch trainiert...' : 
                        project.status === ProjectStatus['Re-Training'] ?
                        'Das Modell wird re-trainiert...' :
                        'Kein trainiertes Modell verfügbar'
                    }
                </div>
            )}
        </div>
        
        {/* Projekt-Informationen */}
        {project.recommendations && (
          <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 border border-purple-500/50 rounded-lg p-4">
            <h4 className="text-purple-400 font-medium mb-3">🤖 Ursprüngliche KI-Empfehlungen</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-gray-300"><strong>Algorithmus:</strong> {project.recommendations.algorithm}</p>
                <p className="text-gray-300"><strong>Model-Typ:</strong> {project.recommendations.modelType}</p>
              </div>
              <div>
                <p className="text-gray-300"><strong>Features:</strong> {project.recommendations.features?.length || 0}</p>
                <p className="text-gray-300"><strong>Zielvariable:</strong> {project.recommendations.targetVariable}</p>
              </div>
            </div>
            <details className="mt-3">
              <summary className="cursor-pointer text-purple-200 hover:text-purple-100 text-sm">KI-Begründung anzeigen</summary>
              <p className="mt-2 text-purple-100 text-xs bg-purple-950/30 rounded p-2">
                {project.recommendations.reasoning || 'Keine Begründung verfügbar'}
              </p>
            </details>
          </div>
        )}
    </div>
  );

  const renderDataTab = () => {
    return (
      <div className="space-y-6">
        {/* Load Button */}
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold text-white flex items-center">
            <span className="mr-2">📋</span>Data Insights
          </h3>
          <button
            onClick={loadDataStatistics}
            disabled={isLoadingData}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg transition-colors flex items-center space-x-2"
          >
            {isLoadingData ? (
              <>
                <Spinner size="sm" />
                <span>Lade...</span>
              </>
            ) : (
              <>
                <span>🔄</span>
                <span>Daten neu laden</span>
              </>
            )}
          </button>
        </div>

        {/* Error Display */}
        {dataError && (
          <div className="bg-red-900/30 border border-red-500/50 rounded-lg p-4">
            <p className="text-red-400">❌ {dataError}</p>
          </div>
        )}

        {/* Loading State */}
        {isLoadingData && !dataStatistics && (
          <div className="bg-blue-900/30 border border-blue-500/50 rounded-lg p-6">
            <div className="flex items-center space-x-4">
              <Spinner size="sm" />
              <div>
                <h4 className="text-blue-400 font-medium">📊 Lade Datenstatistiken...</h4>
                <p className="text-blue-300 text-sm">Analysiere ursprüngliche Datei und extrahiere Kennzahlen</p>
              </div>
            </div>
          </div>
        )}

        {/* Data Statistics Display */}
        {dataStatistics && (
          <>
            {/* Basic Info */}
            <div className="bg-gray-800/50 rounded-lg p-6">
              <h4 className="text-white font-medium mb-4 flex items-center">
                <span className="mr-2">📋</span>Grundlegende Dateiinformationen
              </h4>
              
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-gray-700/50 rounded-lg p-4 text-center">
                  <h5 className="text-gray-400 text-sm font-medium mb-1">Dateiname</h5>
                  <p className="text-white font-semibold text-sm">{dataStatistics.basicInfo.fileName}</p>
                </div>
                <div className="bg-gray-700/50 rounded-lg p-4 text-center">
                  <h5 className="text-gray-400 text-sm font-medium mb-1">Zeilen</h5>
                  <p className="text-blue-400 font-semibold">{dataStatistics.basicInfo.rowCount.toLocaleString()}</p>
                </div>
                <div className="bg-gray-700/50 rounded-lg p-4 text-center">
                  <h5 className="text-gray-400 text-sm font-medium mb-1">Spalten</h5>
                  <p className="text-green-400 font-semibold">{dataStatistics.basicInfo.columnCount}</p>
                </div>
                <div className="bg-gray-700/50 rounded-lg p-4 text-center">
                  <h5 className="text-gray-400 text-sm font-medium mb-1">Datentypen</h5>
                  <p className="text-purple-400 font-semibold">
                    {Object.values(dataStatistics.basicInfo.dataTypes).filter(t => t === 'numeric').length}N / 
                    {Object.values(dataStatistics.basicInfo.dataTypes).filter(t => t === 'categorical').length}C
                  </p>
                </div>
              </div>
            </div>

            {/* Column Analysis */}
            <div className="bg-gray-800/50 rounded-lg p-6">
              <h4 className="text-white font-medium mb-4 flex items-center">
                <span className="mr-2">🔢</span>Spalten-Analyse
              </h4>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {dataStatistics.columnAnalysis.map((column: any, index: number) => (
                  <div key={column.name} className="bg-gray-700/50 rounded-lg p-4">
                    <div className="flex justify-between items-center mb-2">
                      <h5 className="font-medium text-white">{column.name}</h5>
                      <div className="flex space-x-1">
                        {column.isTarget && (
                          <span className="text-xs px-2 py-1 bg-green-900/50 text-green-400 rounded">Target</span>
                        )}
                        {column.isFeature && (
                          <span className="text-xs px-2 py-1 bg-blue-900/50 text-blue-400 rounded">Feature</span>
                        )}
                      </div>
                    </div>
                    <div className="text-xs space-y-1">
                      <p className="text-gray-400">
                        Typ: <span className={`font-medium ${column.dataType === 'numeric' ? 'text-blue-300' : 'text-green-300'}`}>
                          {column.dataType}
                        </span>
                      </p>
                      {column.sampleValues.length > 0 && (
                        <div>
                          <p className="text-gray-400">Beispielwerte:</p>
                          <p className="text-gray-300 font-mono text-xs truncate">
                            {column.sampleValues.slice(0, 3).join(', ')}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* ML Configuration */}
            <div className="bg-gray-800/50 rounded-lg p-6">
              <h4 className="text-white font-medium mb-4 flex items-center">
                <span className="mr-2">⚙️</span>ML-Konfiguration
              </h4>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h5 className="text-gray-400 text-sm font-medium mb-3">Algorithmus-Details</h5>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Algorithmus:</span>
                      <span className="text-white font-medium">{dataStatistics.mlConfig.algorithm || 'Standard'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Modell-Typ:</span>
                      <span className="text-purple-400">{dataStatistics.mlConfig.modelType}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Zielvariable:</span>
                      <span className="text-green-400">{dataStatistics.mlConfig.targetVariable}</span>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h5 className="text-gray-400 text-sm font-medium mb-3">Feature-Auswahl</h5>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Verwendete Features:</span>
                      <span className="text-blue-400 font-semibold">{dataStatistics.mlConfig.features.length}</span>
                    </div>
                    {dataStatistics.mlConfig.excludedColumns.length > 0 && (
                      <div className="flex justify-between">
                        <span className="text-gray-400">Entfernte Spalten:</span>
                        <span className="text-red-400 font-semibold">{dataStatistics.mlConfig.excludedColumns.length}</span>
                      </div>
                    )}
                    {dataStatistics.mlConfig.excludedFeatures.length > 0 && (
                      <div className="flex justify-between">
                        <span className="text-gray-400">Ausgeschlossene Features:</span>
                        <span className="text-orange-400 font-semibold">{dataStatistics.mlConfig.excludedFeatures.length}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Sample Data Preview */}
            <div className="bg-gray-800/50 rounded-lg p-6">
              <h4 className="text-white font-medium mb-4 flex items-center">
                <span className="mr-2">👀</span>Datenvorschau (erste {dataStatistics.sampleData.rows.length} Zeilen)
              </h4>
              
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-600">
                      {dataStatistics.sampleData.headers.map((header: string) => (
                        <th key={header} className="text-left py-2 px-3 text-gray-400 font-medium">
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {dataStatistics.sampleData.rows.map((row: any[], index: number) => (
                      <tr key={index} className="border-b border-gray-700 hover:bg-gray-700/30">
                        {row.map((cell: any, cellIndex: number) => (
                          <td key={cellIndex} className="py-2 px-3 text-gray-300 font-mono text-xs">
                            {String(cell).length > 30 ? String(cell).substring(0, 30) + '...' : String(cell)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {/* No Data State */}
        {!dataStatistics && !isLoadingData && !dataError && (
          <div className="bg-gray-800/50 rounded-lg p-6 text-center">
            <div className="flex flex-col items-center space-y-4">
              <span className="text-4xl">📊</span>
              <div>
                <h4 className="text-white font-medium">Keine Datenstatistiken geladen</h4>
                <p className="text-gray-400 text-sm mt-1">
                  Klicken Sie auf "Daten neu laden", um die Analyse zu starten.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const tabs: {id: Tab, name: string, badge?: string}[] = [
      { id: 'predict', name: 'Predict' },
      { id: 'performance', name: 'Performance' },
      { id: 'data', name: 'Data Insights' },
      { id: 'code', name: 'Code Editor' },
      { id: 'api', name: 'API Info' },
      { id: 'export', name: 'Export' }
  ];

  const getTabClasses = (tab: Tab) => {
    const baseClasses = 'px-4 py-2 text-sm font-medium rounded-md transition-colors';
    return activeTab === tab 
      ? `${baseClasses} bg-blue-600 text-white` 
      : `${baseClasses} text-slate-400 hover:text-white hover:bg-slate-800`;
  };



  return (
    <ErrorBoundary>
      <div className="bg-gray-800/50 rounded-lg shadow-xl p-6 sm:p-8 animate-fade-in">
        <button onClick={onBack} className="flex items-center text-sm text-blue-400 hover:text-blue-300 mb-6">
          <ChevronLeftIcon className="h-5 w-5 mr-1" />
          Back to Dashboard
        </button>

        <div className="mb-6">
          <div className="flex justify-between items-start">
            <div>
              <h2 className="text-3xl font-bold tracking-tight text-white">{project.name}</h2>
              <p className="text-gray-400 mt-1">Model Type: {project.modelType}</p>
              <div className="flex items-center mt-2 space-x-4">
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                  project.status === ProjectStatus.Completed ? 'bg-emerald-900/50 text-emerald-400 border border-emerald-500/50' :
                  project.status === ProjectStatus.Training || project.status === ProjectStatus['Re-Training'] ? 'bg-blue-900/50 text-blue-400 border border-blue-500/50' :
                  'bg-red-900/50 text-red-400 border border-red-500/50'
                }`}>
                  {project.status}
                </span>
                {project.algorithm && (
                  <span className="text-gray-400 text-sm">Algorithm: {project.algorithm}</span>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="border-b border-gray-700">
          <nav className="-mb-px flex space-x-8" aria-label="Tabs">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-400'
                    : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-gray-500'
                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors`}
              >
                {tab.name}
                {tab.id === 'code' && isCodeModified && (
                  <span className="ml-1 w-2 h-2 bg-yellow-400 rounded-full inline-block"></span>
                )}
                {tab.badge && <span className="ml-2 text-xs font-semibold text-blue-400">{tab.badge}</span>}
              </button>
            ))}
          </nav>
        </div>
        
        <div className="mt-8">
          <ErrorBoundary fallback={
            <div className="bg-red-900/30 border border-red-500/50 rounded-lg p-6 text-center">
              <p className="text-red-400">❌ Fehler beim Laden dieses Tabs. Bitte wählen Sie einen anderen Tab oder laden Sie die Seite neu.</p>
            </div>
          }>
            {activeTab === 'predict' && renderPredictTab()}
            {activeTab === 'performance' && renderPerformanceTab()}
            {activeTab === 'data' && renderDataTab()}
            {activeTab === 'code' && renderCodeTab()}
            {activeTab === 'api' && renderApiTab()}
            {activeTab === 'export' && renderExportTab()}
          </ErrorBoundary>
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default ProjectView;