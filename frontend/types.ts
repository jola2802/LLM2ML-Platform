
export enum View {
  DASHBOARD,
  WIZARD,
  PROJECT_VIEW,
}

export enum ProjectStatus {
  Training = 'Training',
  Completed = 'Completed',
  Failed = 'Failed',
  Paused = 'Paused',
  'Re-Training' = 'Re-Training',
  'Re-training Failed' = 'Re-training Failed',
}

export enum ModelType {
  Classification = 'Classification',
  Regression = 'Regression',
}

export interface LLMRecommendations {
  targetVariable: string;
  features: string[];
  modelType: ModelType;
  algorithm: string;
  hyperparameters: { [key: string]: any };
  reasoning: string;
  dataSourceName: string;
  confidence?: number;
}

export interface MetricInterpretation {
  value: number;
  interpretation: string;
  benchmarkComparison: string;
}

export interface ImprovementSuggestion {
  category: 'Data Quality' | 'Feature Engineering' | 'Algorithm Tuning' | 'Model Architecture' | 'General';
  suggestion: string;
  expectedImpact: 'Low' | 'Medium' | 'High';
  implementation: string;
}

export interface BusinessImpact {
  readiness: 'Production Ready' | 'Needs Improvement' | 'Not Ready';
  riskAssessment: 'Low' | 'Medium' | 'High';
  recommendation: string;
}

export interface PerformanceInsights {
  overallScore: number;
  performanceGrade: 'Excellent' | 'Good' | 'Fair' | 'Poor' | 'Critical';
  summary: string;
  detailedAnalysis: {
    strengths: string[];
    weaknesses: string[];
    keyFindings: string[];
  };
  metricsInterpretation: { [key: string]: MetricInterpretation };
  improvementSuggestions: ImprovementSuggestion[];
  businessImpact: BusinessImpact;
  nextSteps: string[];
  evaluatedAt: string;
  evaluatedBy: string;
  version: string;
}

export interface Project {
  id: string;
  name: string;
  status: ProjectStatus;
  modelType: ModelType;
  dataSourceName: string;
  targetVariable: string;
  features: string[];
  createdAt: string;
  performanceMetrics?: {
    [key: string]: number;
  };
  performanceInsights?: PerformanceInsights;
  pythonCode?: string;
  originalPythonCode?: string;
  modelArtifact?: string;
  algorithm?: string;
  hyperparameters?: { [key: string]: any };
  recommendations?: LLMRecommendations;
}

export interface ColumnInfo {
    name: string;
}

export interface CsvAnalysisResult {
  fileName: string;
  filePath: string;
  fileType?: string;
  columns: string[];
  rowCount: number;
  dataTypes: { [key: string]: string };
  sampleData: any[][];
  llmAnalysis: string;
  recommendations: LLMRecommendations;
}
