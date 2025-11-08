/**
 * Performance-Analyzer-Worker-Agent
 * 
 * Analysiert Modell-Performance und gibt Verbesserungsvorschläge.
 * Versteht verschiedene Metriken und deren Interpretation.
 */

import { BaseWorker } from './0_base_agent.js';
import { PERFORMANCE_ANALYSIS_PROMPT, PERFORMANCE_ANALYZER_TEST_PROMPT, formatPrompt } from './prompts.js';

export class PerformanceAnalyzerWorker extends BaseWorker {
  constructor() {
    super('PERFORMANCE_ANALYZER');
  }

  async execute(pipelineState) {
    this.log('info', 'Starte Performance-Analyse');
    
    const { project, results } = pipelineState;
    
    // Prüfe, ob Code verfügbar ist (entweder generiert oder reviewed)
    const codeToAnalyze = results.CODE_REVIEWER || results.CODE_GENERATOR;
    if (!codeToAnalyze) {
      throw new Error('Python-Code erforderlich für Performance-Analyse');
    }

    try {
      const performanceAnalysis = await this.analyzePerformance(
        codeToAnalyze,
        project,
        results.DATA_ANALYZER,
        results.HYPERPARAMETER_OPTIMIZER
      );
      
      this.log('success', 'Performance-Analyse erfolgreich abgeschlossen');
      return performanceAnalysis;

    } catch (error) {
      this.log('error', 'Performance-Analyse fehlgeschlagen', error.message);
      throw error;
    }
  }

  async analyzePerformance(pythonCode, project, dataAnalysis, hyperparameterSuggestions) {
    const prompt = formatPrompt(PERFORMANCE_ANALYSIS_PROMPT, {
      pythonCode,
      projectName: project.name,
      algorithm: hyperparameterSuggestions?.primary_algorithm || project.algorithm,
      csvFilePath: project.csvFilePath,
      dataAnalysis: JSON.stringify(dataAnalysis, null, 2),
      hyperparameterSuggestions: JSON.stringify(hyperparameterSuggestions, null, 2)
    });

    const response = await this.callLLM(prompt);
    const analysis = typeof response === 'string' ? response : response?.result || '';
    
    // Strukturiere die Analyse
    const structuredAnalysis = this.structureAnalysis(analysis, project, pythonCode);
    
    return structuredAnalysis;
  }

  structureAnalysis(analysis, project, pythonCode) {
    // Extrahiere Performance-Bewertung
    const performanceScore = this.extractPerformanceScore(analysis);
    
    // Extrahiere identifizierte Probleme
    const problems = this.extractProblems(analysis);
    
    // Extrahiere Verbesserungsvorschläge
    const improvements = this.extractImprovements(analysis);
    
    // Extrahiere erwartete Performance-Steigerung
    const expectedImprovement = this.extractExpectedImprovement(analysis);

    return {
      metadata: {
        timestamp: new Date().toISOString(),
        project: project.name,
        analyzed_by: this.agentKey,
        code_length: pythonCode.length
      },
      performance_score: performanceScore,
      problems_identified: problems,
      improvement_suggestions: improvements,
      expected_improvement: expectedImprovement,
      detailed_analysis: analysis,
      recommendations: this.prioritizeRecommendations(improvements)
    };
  }

  extractPerformanceScore(analysis) {
    // Suche nach Performance-Bewertung (1-10)
    const scoreMatch = analysis.match(/performance.*?(\d+)\/10|bewertung.*?(\d+)\/10|score.*?(\d+)\/10/i);
    if (scoreMatch) {
      return parseInt(scoreMatch[1] || scoreMatch[2] || scoreMatch[3]);
    }
    
    // Fallback basierend auf Schlüsselwörtern
    if (analysis.toLowerCase().includes('excellent') || analysis.toLowerCase().includes('ausgezeichnet')) {
      return 9;
    } else if (analysis.toLowerCase().includes('good') || analysis.toLowerCase().includes('gut')) {
      return 7;
    } else if (analysis.toLowerCase().includes('average') || analysis.toLowerCase().includes('durchschnittlich')) {
      return 5;
    } else if (analysis.toLowerCase().includes('poor') || analysis.toLowerCase().includes('schlecht')) {
      return 3;
    }
    
    return 6; // Default
  }

  extractProblems(analysis) {
    const problems = [];
    
    // Suche nach Problemen
    const problemKeywords = [
      'problem', 'issue', 'fehler', 'bottleneck', 'inefficient', 'ineffizient',
      'slow', 'langsam', 'memory', 'speicher', 'overfitting', 'underfitting'
    ];
    
    const lines = analysis.split('\n');
    for (const line of lines) {
      if (problemKeywords.some(keyword => line.toLowerCase().includes(keyword))) {
        problems.push(line.trim());
      }
    }
    
    return problems.length > 0 ? problems : ['Keine spezifischen Probleme identifiziert'];
  }

  extractImprovements(analysis) {
    const improvements = [];
    
    // Suche nach Verbesserungsvorschlägen
    const improvementKeywords = [
      'improve', 'verbessern', 'optimize', 'optimieren', 'suggest', 'vorschlag',
      'recommend', 'empfehlen', 'better', 'besser', 'enhance', 'erweitern'
    ];
    
    const lines = analysis.split('\n');
    for (const line of lines) {
      if (improvementKeywords.some(keyword => line.toLowerCase().includes(keyword))) {
        improvements.push(line.trim());
      }
    }
    
    return improvements.length > 0 ? improvements : ['Keine spezifischen Verbesserungsvorschläge'];
  }

  extractExpectedImprovement(analysis) {
    // Suche nach erwarteter Verbesserung
    const improvementMatch = analysis.match(/(\d+)%|(\d+)\s*percent|verbesserung.*?(\d+)/i);
    if (improvementMatch) {
      return parseInt(improvementMatch[1] || improvementMatch[2] || improvementMatch[3]);
    }
    
    return 15; // Default 15% Verbesserung
  }

  prioritizeRecommendations(improvements) {
    // Priorisiere Empfehlungen basierend auf Schlüsselwörtern
    const highPriority = improvements.filter(imp => 
      imp.toLowerCase().includes('critical') || 
      imp.toLowerCase().includes('kritisch') ||
      imp.toLowerCase().includes('important') ||
      imp.toLowerCase().includes('wichtig')
    );
    
    const mediumPriority = improvements.filter(imp => 
      imp.toLowerCase().includes('recommend') || 
      imp.toLowerCase().includes('empfehlen') ||
      imp.toLowerCase().includes('suggest') ||
      imp.toLowerCase().includes('vorschlagen')
    );
    
    const lowPriority = improvements.filter(imp => 
      !highPriority.includes(imp) && !mediumPriority.includes(imp)
    );
    
    return {
      high_priority: highPriority,
      medium_priority: mediumPriority,
      low_priority: lowPriority
    };
  }

  async test() {
    const testPrompt = formatPrompt(PERFORMANCE_ANALYZER_TEST_PROMPT, {
      agentName: this.agentKey
    });

    try {
      const response = await this.callLLM(testPrompt, null, 10);
      const result = typeof response === 'string' ? response : response?.result || '';
      return result.toLowerCase().includes('ok');
    } catch (error) {
      this.log('error', 'Test fehlgeschlagen', error.message);
      return false;
    }
  }
}
