import React, { useState } from 'react';
import { PerformanceInsights as PerformanceInsightsType, Project, ImprovementSuggestion } from '../types';
import { apiService } from '../services/apiService';
import { Spinner } from './ui/Spinner';

interface PerformanceInsightsProps {
  project: Project;
  onInsightsUpdate?: (insights: PerformanceInsightsType) => void;
}

const PerformanceInsights: React.FC<PerformanceInsightsProps> = ({ project, onInsightsUpdate }) => {
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);

  const handleEvaluatePerformance = async () => {
    setIsEvaluating(true);
    setEvaluationError(null);
    
    try {
      const response = await apiService.evaluatePerformance(project.id);
      const insights = response.insights;
      
      if (onInsightsUpdate) {
        onInsightsUpdate(insights);
      }
    } catch (error) {
      setEvaluationError(error instanceof Error ? error.message : 'Evaluation fehlgeschlagen');
    } finally {
      setIsEvaluating(false);
    }
  };

  const getGradeColor = (grade: string) => {
    switch (grade) {
      case 'Excellent': return 'text-green-400 border-green-500/50 bg-green-900/30';
      case 'Good': return 'text-blue-400 border-blue-500/50 bg-blue-900/30';
      case 'Fair': return 'text-yellow-400 border-yellow-500/50 bg-yellow-900/30';
      case 'Poor': return 'text-orange-400 border-orange-500/50 bg-orange-900/30';
      case 'Critical': return 'text-red-400 border-red-500/50 bg-red-900/30';
      default: return 'text-gray-400 border-gray-500/50 bg-gray-900/30';
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'High': return 'bg-red-900/50 text-red-400 border border-red-500/50';
      case 'Medium': return 'bg-yellow-900/50 text-yellow-400 border border-yellow-500/50';
      case 'Low': return 'bg-green-900/50 text-green-400 border border-green-500/50';
      default: return 'bg-gray-900/50 text-gray-400 border border-gray-500/50';
    }
  };

  const getReadinessColor = (readiness: string) => {
    switch (readiness) {
      case 'Production Ready': return 'text-green-400';
      case 'Needs Improvement': return 'text-yellow-400';
      case 'Not Ready': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  if (!project.performanceMetrics) {
    return (
      <div className="bg-gray-800/50 rounded-lg p-6 text-center">
        <p className="text-gray-400">Keine Performance-Metriken verfügbar für die Evaluation.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Evaluation Control */}
      <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 border border-purple-500/50 rounded-lg p-6">
        <div className="flex justify-between items-center">
          <div>
            <h3 className="text-lg font-semibold text-white mb-2">🤖 KI-Performance-Analyse</h3>
            <p className="text-purple-200 text-sm">
              Lasse die Performance deines Modells intelligent vom LLM bewerten und erhalte detaillierte Insights.
            </p>
          </div>
          <button
            onClick={handleEvaluatePerformance}
            disabled={isEvaluating}
            className="px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-medium rounded-lg transition-all duration-200 flex items-center space-x-2"
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

      {/* Performance Insights Display */}
      {project.performanceInsights && (
        <div className="space-y-6">
          {/* Overall Score & Grade */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-800/50 rounded-lg p-6 text-center">
              <h4 className="text-gray-400 text-sm font-medium mb-2">Gesamt-Score</h4>
              <div className="text-4xl font-bold text-white mb-2">
                {project.performanceInsights.overallScore.toFixed(1)}/10
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${(project.performanceInsights.overallScore / 10) * 100}%` }}
                ></div>
              </div>
            </div>
            
            <div className="bg-gray-800/50 rounded-lg p-6 text-center">
              <h4 className="text-gray-400 text-sm font-medium mb-2">Performance-Grade</h4>
              <div className={`inline-flex items-center px-4 py-2 rounded-lg border text-lg font-semibold ${getGradeColor(project.performanceInsights.performanceGrade)}`}>
                {project.performanceInsights.performanceGrade}
              </div>
            </div>
          </div>

          {/* Summary */}
          <div className="bg-blue-900/20 border border-blue-500/50 rounded-lg p-6">
            <h4 className="text-blue-400 font-medium mb-3">📊 Zusammenfassung</h4>
            <p className="text-blue-100">{project.performanceInsights.summary}</p>
          </div>

          {/* Detailed Analysis */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="bg-green-900/20 border border-green-500/50 rounded-lg p-6">
              <h4 className="text-green-400 font-medium mb-3 flex items-center">
                <span className="mr-2">✅</span>Stärken
              </h4>
              <ul className="space-y-2">
                {project.performanceInsights.detailedAnalysis.strengths.map((strength, index) => (
                  <li key={index} className="text-green-100 text-sm flex items-start">
                    <span className="text-green-400 mr-2 mt-1">•</span>
                    {strength}
                  </li>
                ))}
              </ul>
            </div>

            <div className="bg-orange-900/20 border border-orange-500/50 rounded-lg p-6">
              <h4 className="text-orange-400 font-medium mb-3 flex items-center">
                <span className="mr-2">⚠️</span>Schwächen
              </h4>
              <ul className="space-y-2">
                {project.performanceInsights.detailedAnalysis.weaknesses.map((weakness, index) => (
                  <li key={index} className="text-orange-100 text-sm flex items-start">
                    <span className="text-orange-400 mr-2 mt-1">•</span>
                    {weakness}
                  </li>
                ))}
              </ul>
            </div>

            <div className="bg-purple-900/20 border border-purple-500/50 rounded-lg p-6">
              <h4 className="text-purple-400 font-medium mb-3 flex items-center">
                <span className="mr-2">🔍</span>Wichtige Erkenntnisse
              </h4>
              <ul className="space-y-2">
                {project.performanceInsights.detailedAnalysis.keyFindings.map((finding, index) => (
                  <li key={index} className="text-purple-100 text-sm flex items-start">
                    <span className="text-purple-400 mr-2 mt-1">•</span>
                    {finding}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Metrics Interpretation */}
          <div className="bg-gray-800/50 rounded-lg p-6">
            <h4 className="text-white font-medium mb-4 flex items-center">
              <span className="mr-2">📈</span>Metriken-Interpretation
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(project.performanceInsights.metricsInterpretation).map(([metric, data]) => (
                <div key={metric} className="bg-gray-700/50 rounded-lg p-4">
                  <div className="flex justify-between items-center mb-2">
                    <h5 className="font-medium text-white capitalize">{metric.replace(/_/g, ' ')}</h5>
                    <span className="text-blue-400 font-mono">{data.value.toFixed(4)}</span>
                  </div>
                  <p className="text-gray-300 text-sm mb-2">{data.interpretation}</p>
                  <p className="text-gray-400 text-xs">{data.benchmarkComparison}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Improvement Suggestions */}
          <div className="bg-gray-800/50 rounded-lg p-6">
            <h4 className="text-white font-medium mb-4 flex items-center">
              <span className="mr-2">💡</span>Verbesserungsvorschläge
            </h4>
            <div className="space-y-4">
              {project.performanceInsights.improvementSuggestions.map((suggestion, index) => (
                <div key={index} className="bg-gray-700/50 rounded-lg p-4">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <span className="text-blue-400 text-sm font-medium">{suggestion.category}</span>
                      <div className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ml-3 ${getImpactColor(suggestion.expectedImpact)}`}>
                        {suggestion.expectedImpact} Impact
                      </div>
                    </div>
                  </div>
                  <p className="text-gray-300 mb-2">{suggestion.suggestion}</p>
                  <details className="cursor-pointer">
                    <summary className="text-gray-400 hover:text-gray-300 text-sm">Umsetzung anzeigen</summary>
                    <p className="text-gray-400 text-sm mt-2 pl-4 border-l-2 border-gray-600">
                      {suggestion.implementation}
                    </p>
                  </details>
                </div>
              ))}
            </div>
          </div>

          {/* Business Impact */}
          <div className="bg-gray-800/50 rounded-lg p-6">
            <h4 className="text-white font-medium mb-4 flex items-center">
              <span className="mr-2">🏢</span>Business Impact
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <h5 className="text-gray-400 text-sm mb-2">Produktionsbereitschaft</h5>
                <span className={`font-medium ${getReadinessColor(project.performanceInsights.businessImpact.readiness)}`}>
                  {project.performanceInsights.businessImpact.readiness}
                </span>
              </div>
              <div className="text-center">
                <h5 className="text-gray-400 text-sm mb-2">Risikobewertung</h5>
                <span className={`font-medium ${project.performanceInsights.businessImpact.riskAssessment === 'Low' ? 'text-green-400' : project.performanceInsights.businessImpact.riskAssessment === 'Medium' ? 'text-yellow-400' : 'text-red-400'}`}>
                  {project.performanceInsights.businessImpact.riskAssessment}
                </span>
              </div>
              <div className="text-center md:col-span-1">
                <h5 className="text-gray-400 text-sm mb-2">Empfehlung</h5>
                <p className="text-gray-300 text-sm">{project.performanceInsights.businessImpact.recommendation}</p>
              </div>
            </div>
          </div>

          {/* Next Steps
          <div className="bg-gray-800/50 rounded-lg p-6">
            <h4 className="text-white font-medium mb-4 flex items-center">
              <span className="mr-2">🚀</span>Nächste Schritte
            </h4>
            <ol className="space-y-2">
              {project.performanceInsights.nextSteps.map((step, index) => (
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
              Evaluiert am {new Date(project.performanceInsights.evaluatedAt).toLocaleString('de-DE')} 
              von {project.performanceInsights.evaluatedBy} (v{project.performanceInsights.version})
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default PerformanceInsights; 