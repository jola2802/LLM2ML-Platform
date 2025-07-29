import React, { useState, useMemo, useCallback } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Project, ProjectStatus, PerformanceInsights as PerformanceInsightsType } from '../types';
import { apiService } from '../services/apiService';
import { ChevronLeftIcon } from './icons/ChevronLeftIcon';
import { Spinner } from './ui/Spinner';
import PerformanceInsights from './PerformanceInsights';

interface ProjectViewProps {
  project: Project;
  onBack: () => void;
  onProjectUpdate?: (updatedProject: Project) => void;
}

type Tab = 'predict' | 'performance' | 'insights' | 'code' | 'api' | 'export';

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

  // Performance Insights State
  const [currentProject, setCurrentProject] = useState<Project>(project);

  const handleInsightsUpdate = useCallback((insights: PerformanceInsightsType) => {
    const updatedProject = { ...currentProject, performanceInsights: insights };
    setCurrentProject(updatedProject);
    
    if (onProjectUpdate) {
      onProjectUpdate(updatedProject);
    }
  }, [currentProject, onProjectUpdate]);
  
  const handleInputChange = (feature: string, value: string) => {
    setPredictionInput(prev => ({ ...prev, [feature]: value }));
  };

  const handlePredict = async () => {
    setIsPredicting(true);
    setPredictionError(null);
    setPredictionResult(null);
    try {
      // Konvertiere Input zu korrekten Datentypen
      const processedInput: { [key: string]: string | number } = {};
      Object.entries(predictionInput).forEach(([key, value]) => {
        // Versuche als Zahl zu parsen, falls möglich
        const numberValue = parseFloat(value);
        processedInput[key] = isNaN(numberValue) ? value : numberValue;
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
      await apiService.updatePythonCode(project.id, pythonCode);
      setIsCodeModified(false);
      setCodeMessage('✅ Code erfolgreich gespeichert');
      
      // Update project in parent component
      if (onProjectUpdate) {
        onProjectUpdate({ ...project, pythonCode });
      }
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
              <label htmlFor={feature} className="block text-sm font-medium text-gray-300 capitalize">{feature.replace(/_/g, ' ')}</label>
              <input
                type="text"
                id={feature}
                value={predictionInput[feature] || ''}
                onChange={(e) => handleInputChange(feature, e.target.value)}
                className="mt-1 block w-full bg-gray-700 border-gray-600 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm text-white p-2"
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
      <div className="bg-gray-800 p-6 rounded-lg flex flex-col items-center justify-center">
        <h3 className="text-xl font-semibold text-white mb-4">Prediction Result</h3>
        <div className="flex-grow flex items-center justify-center w-full">
            {isPredicting && <Spinner size="lg"/>}
            {predictionError && <p className="text-red-400 text-center">{predictionError}</p>}
            {predictionResult && (
                <div className="text-center">
                    <p className="text-sm text-gray-400 mb-2">Predicted {project.targetVariable.replace(/_/g, ' ')}:</p>
                    <p className="text-4xl font-bold text-blue-400 break-all">{predictionResult}</p>
                </div>
            )}
            {!isPredicting && !predictionError && !predictionResult && <p className="text-gray-500">Result will appear here</p>}
        </div>
      </div>
    </div>
  );

  const renderPerformanceTab = () => (
    <div className="space-y-6">
      {/* Quick Insights Summary */}
      {currentProject.performanceInsights && (
        <div className="bg-gradient-to-r from-purple-900/20 to-blue-900/20 border border-purple-500/30 rounded-lg p-4">
          <div className="flex justify-between items-center mb-3">
            <h4 className="text-purple-400 font-medium flex items-center">
              <span className="mr-2">🤖</span>KI-Schnellanalyse
            </h4>
            <div className="flex items-center space-x-3">
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                currentProject.performanceInsights.performanceGrade === 'Excellent' ? 'bg-green-900/50 text-green-400' :
                currentProject.performanceInsights.performanceGrade === 'Good' ? 'bg-blue-900/50 text-blue-400' :
                currentProject.performanceInsights.performanceGrade === 'Fair' ? 'bg-yellow-900/50 text-yellow-400' :
                'bg-red-900/50 text-red-400'
              }`}>
                {currentProject.performanceInsights.performanceGrade}
              </span>
              <span className="text-purple-300 font-semibold">
                {currentProject.performanceInsights.overallScore.toFixed(1)}/10
              </span>
            </div>
          </div>
          <p className="text-purple-100 text-sm mb-3">{currentProject.performanceInsights.summary}</p>
          <button
            onClick={() => setActiveTab('insights')}
            className="text-purple-400 hover:text-purple-300 text-sm font-medium"
          >
            → Vollständige Analyse anzeigen
          </button>
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
          {!currentProject.performanceInsights && (
            <p className="text-gray-500 text-sm">Trainiere das Modell, um automatisch Performance-Metriken und KI-Insights zu erhalten.</p>
          )}
        </div>
      )}
    </div>
  );

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
        </div>
      </div>

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

  const renderInsightsTab = () => (
    <PerformanceInsights 
      project={currentProject} 
      onInsightsUpdate={handleInsightsUpdate}
    />
  );

  const tabs: {id: Tab, name: string, badge?: string}[] = [
      { id: 'predict', name: 'Predict' },
      { id: 'performance', name: 'Performance' },
      { id: 'insights', name: 'KI-Insights' },
      { id: 'code', name: 'Code Editor' },
      { id: 'api', name: 'API Info' },
      { id: 'export', name: 'Export' }
  ];

  return (
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
                project.status === ProjectStatus.Completed ? 'bg-green-900/50 text-green-400 border border-green-500/50' :
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
        {activeTab === 'predict' && renderPredictTab()}
        {activeTab === 'performance' && renderPerformanceTab()}
        {activeTab === 'insights' && renderInsightsTab()}
        {activeTab === 'code' && renderCodeTab()}
        {activeTab === 'api' && renderApiTab()}
        {activeTab === 'export' && renderExportTab()}
      </div>
    </div>
  );
};

export default ProjectView;