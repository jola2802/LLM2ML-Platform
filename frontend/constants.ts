
import { ProjectStatus, ModelType } from './types';

export const PROJECT_STATUS_COLORS: Record<ProjectStatus, string> = {
  [ProjectStatus.Training]: 'bg-blue-500 text-white',
  [ProjectStatus.Completed]: 'bg-green-500 text-white',
  [ProjectStatus.Failed]: 'bg-red-500 text-white',
  [ProjectStatus.Paused]: 'bg-yellow-500 text-white',
};

export const MODEL_TYPE_DESCRIPTIONS = {
  Classification: 'Predict categories or classes (e.g., spam/not spam, yes/no)',
  Regression: 'Predict continuous numerical values (e.g., price, temperature)'
};

export const ALGORITHMS = {
  Classification: [
    {
      name: 'RandomForestClassifier',
      displayName: 'Random Forest',
      description: 'Robust und interpretierbar, funktioniert gut mit verschiedenen Datentypen',
      complexity: 'Mittel'
    },
    {
      name: 'LogisticRegression', 
      displayName: 'Logistic Regression',
      description: 'Schnell und einfach interpretierbar, gut für binäre Klassifikation',
      complexity: 'Niedrig'
    },
    {
      name: 'SVM',
      displayName: 'Support Vector Machine',
      description: 'Leistungsstark für komplexe Daten, funktioniert gut bei wenigen Samples',
      complexity: 'Hoch'
    },
    {
      name: 'XGBoostClassifier',
      displayName: 'XGBoost',
      description: 'Sehr leistungsstarker Gradient Boosting Algorithmus',
      complexity: 'Hoch'
    }
  ],
  Regression: [
    {
      name: 'RandomForestRegressor',
      displayName: 'Random Forest',
      description: 'Robust und interpretierbar, funktioniert gut mit verschiedenen Datentypen',
      complexity: 'Mittel'
    },
    {
      name: 'LinearRegression',
      displayName: 'Linear Regression', 
      description: 'Schnell und einfach interpretierbar, gut für lineare Zusammenhänge',
      complexity: 'Niedrig'
    },
    {
      name: 'SVR',
      displayName: 'Support Vector Regression',
      description: 'Leistungsstark für nichtlineare Zusammenhänge',
      complexity: 'Hoch'
    },
    {
      name: 'XGBoostRegressor',
      displayName: 'XGBoost',
      description: 'Sehr leistungsstarker Gradient Boosting Algorithmus',
      complexity: 'Hoch'
    }
  ]
};
