import React, { useState, useEffect } from 'react';
import { Spinner } from './ui/Spinner';

interface HyperparameterEditorProps {
  algorithm: string;
  currentHyperparameters: { [key: string]: any };
  onHyperparametersChange: (hyperparameters: { [key: string]: any }) => void;
  isRetraining: boolean;
}

interface HyperparameterConfig {
  name: string;
  displayName: string;
  type: 'number' | 'select';
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
  description: string;
}

const HyperparameterEditor: React.FC<HyperparameterEditorProps> = ({
  algorithm,
  currentHyperparameters,
  onHyperparametersChange,
  isRetraining
}) => {
  const [localHyperparameters, setLocalHyperparameters] = useState<{ [key: string]: any }>(currentHyperparameters);

  // Hyperparameter-Konfigurationen für verschiedene Algorithmen
  const getHyperparameterConfigs = (): HyperparameterConfig[] => {
    switch (algorithm.toLowerCase()) {
      case 'randomforestclassifier':
      case 'randomforest':
        return [
          {
            name: 'n_estimators',
            displayName: 'Anzahl Bäume',
            type: 'number',
            min: 10,
            max: 1000,
            step: 10,
            description: 'Anzahl der Entscheidungsbäume im Wald'
          },
          {
            name: 'max_depth',
            displayName: 'Maximale Tiefe',
            type: 'number',
            min: 1,
            max: 50,
            step: 1,
            description: 'Maximale Tiefe der einzelnen Bäume'
          },
          {
            name: 'min_samples_split',
            displayName: 'Min. Samples zum Teilen',
            type: 'number',
            min: 2,
            max: 20,
            step: 1,
            description: 'Minimale Anzahl Samples zum Teilen eines Knotens'
          },
          {
            name: 'min_samples_leaf',
            displayName: 'Min. Samples pro Blatt',
            type: 'number',
            min: 1,
            max: 10,
            step: 1,
            description: 'Minimale Anzahl Samples pro Blattknoten'
          }
        ];
      
      case 'logisticregression':
        return [
          {
            name: 'C',
            displayName: 'Regularisierung (C)',
            type: 'number',
            min: 0.01,
            max: 100,
            step: 0.01,
            description: 'Inverse Regularisierungsstärke (kleinere Werte = stärkere Regularisierung)'
          },
          {
            name: 'max_iter',
            displayName: 'Max. Iterationen',
            type: 'number',
            min: 100,
            max: 2000,
            step: 100,
            description: 'Maximale Anzahl Optimierungsiterationen'
          }
        ];
      
      case 'svc':
      case 'svm':
        return [
          {
            name: 'C',
            displayName: 'Regularisierung (C)',
            type: 'number',
            min: 0.01,
            max: 100,
            step: 0.01,
            description: 'Regularisierungsparameter'
          },
          {
            name: 'kernel',
            displayName: 'Kernel',
            type: 'select',
            options: [
              { value: 'rbf', label: 'RBF (Standard)' },
              { value: 'linear', label: 'Linear' },
              { value: 'poly', label: 'Polynomial' },
              { value: 'sigmoid', label: 'Sigmoid' }
            ],
            description: 'Kernel-Funktion für SVM'
          }
        ];
      
      case 'gradientboostingclassifier':
      case 'gradientboosting':
        return [
          {
            name: 'n_estimators',
            displayName: 'Anzahl Bäume',
            type: 'number',
            min: 10,
            max: 500,
            step: 10,
            description: 'Anzahl der Boosting-Stufen'
          },
          {
            name: 'learning_rate',
            displayName: 'Lernrate',
            type: 'number',
            min: 0.01,
            max: 1,
            step: 0.01,
            description: 'Lernrate für Boosting'
          },
          {
            name: 'max_depth',
            displayName: 'Maximale Tiefe',
            type: 'number',
            min: 1,
            max: 20,
            step: 1,
            description: 'Maximale Tiefe der einzelnen Bäume'
          }
        ];
      
      case 'kneighborsclassifier':
      case 'knn':
        return [
          {
            name: 'n_neighbors',
            displayName: 'Anzahl Nachbarn (k)',
            type: 'number',
            min: 1,
            max: 50,
            step: 1,
            description: 'Anzahl der nächsten Nachbarn'
          },
          {
            name: 'weights',
            displayName: 'Gewichtung',
            type: 'select',
            options: [
              { value: 'uniform', label: 'Uniform' },
              { value: 'distance', label: 'Nach Distanz' }
            ],
            description: 'Gewichtung der Nachbarn'
          }
        ];
      
      default:
        return [];
    }
  };

  const configs = getHyperparameterConfigs();

  useEffect(() => {
    setLocalHyperparameters(currentHyperparameters);
  }, [currentHyperparameters]);

  const handleParameterChange = (paramName: string, value: any) => {
    // Konvertiere numerische Werte korrekt
    let convertedValue = value;
    if (typeof value === 'string' && !isNaN(Number(value)) && value.trim() !== '') {
      convertedValue = Number(value);
    }
    
    const newHyperparameters = {
      ...localHyperparameters,
      [paramName]: convertedValue
    };
    setLocalHyperparameters(newHyperparameters);
    onHyperparametersChange(newHyperparameters);
  };

  const resetToDefaults = () => {
    const defaultParams: { [key: string]: any } = {};
    configs.forEach(config => {
      if (config.type === 'number') {
        defaultParams[config.name] = config.min || 1;
      } else if (config.type === 'select' && config.options) {
        defaultParams[config.name] = config.options[0].value;
      }
    });
    setLocalHyperparameters(defaultParams);
    onHyperparametersChange(defaultParams);
  };

  if (configs.length === 0) {
    return (
      <div className="bg-yellow-900/20 border border-yellow-500/50 rounded-lg p-4">
        <p className="text-yellow-400 text-sm">
          ⚠️ Hyperparameter-Editor für Algorithmus "{algorithm}" ist noch nicht verfügbar.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h4 className="text-lg font-medium text-white">⚙️ Hyperparameter anpassen</h4>
        <button
          onClick={resetToDefaults}
          disabled={isRetraining}
          className="px-3 py-1 text-xs border border-gray-600 text-gray-300 hover:bg-gray-700 disabled:opacity-50 rounded"
        >
          🔄 Standardwerte
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {configs.map((config) => (
          <div key={config.name} className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              {config.displayName}
            </label>
            
            {config.type === 'number' ? (
              <div className="space-y-2">
                <input
                  type="number"
                  min={config.min}
                  max={config.max}
                  step={config.step}
                  value={localHyperparameters[config.name] || config.min}
                  onChange={(e) => handleParameterChange(config.name, parseFloat(e.target.value))}
                  disabled={isRetraining}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
                />
                <div className="flex justify-between text-xs text-gray-400">
                  <span>Min: {config.min}</span>
                  <span>Max: {config.max}</span>
                </div>
              </div>
            ) : config.type === 'select' && config.options ? (
              <select
                value={localHyperparameters[config.name] || config.options[0].value}
                onChange={(e) => handleParameterChange(config.name, e.target.value)}
                disabled={isRetraining}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
              >
                {config.options.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            ) : null}
            
            <p className="text-xs text-gray-400 mt-2">
              {config.description}
            </p>
          </div>
        ))}
      </div>

      <div className="bg-blue-900/20 border border-blue-500/50 rounded-lg p-3">
        <p className="text-blue-200 text-sm">
          💡 Die Hyperparameter werden automatisch in den Code übernommen, wenn Sie auf "Re-Training" klicken.
        </p>
      </div>
    </div>
  );
};

export default HyperparameterEditor; 