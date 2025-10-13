// Verfügbare Algorithmen definieren
export const ALGORITHMS = {
  // Klassifikation
  'RandomForestClassifier': {
    import: 'from sklearn.ensemble import RandomForestClassifier',
    constructor: 'RandomForestClassifier(n_estimators=100, random_state=42)',
    library: 'sklearn',
    base_hyperparameters: {
      n_estimators: 'int',
      random_state: 'int',
    },
    type: 'Classification'
  },
  'SVM': {
    import: 'from sklearn.svm import SVC',
    constructor: 'SVC(random_state=42, probability=True)',
    library: 'sklearn',
    base_hyperparameters: {
      random_state: 'int',
    },
    type: 'Classification'
  },
  'LogisticRegression': {
    import: 'from sklearn.linear_model import LogisticRegression',
    constructor: 'LogisticRegression(random_state=42, max_iter=1000)',
    library: 'sklearn',
    base_hyperparameters: {
      random_state: 'int',
    },
    type: 'Classification'
  },
  'XGBoostClassifier': {
    import: 'from xgboost import XGBClassifier',
    constructor: 'XGBClassifier(random_state=42, eval_metric="logloss")',
    library: 'xgboost',
    base_hyperparameters: {
      random_state: 'int',
    },
    type: 'Classification'
  },
  
  // Regression
  'RandomForestRegressor': {
    import: 'from sklearn.ensemble import RandomForestRegressor',
    constructor: 'RandomForestRegressor(n_estimators=100, random_state=42)',
    library: 'sklearn',
    base_hyperparameters: {
      random_state: 'int',
    },
    type: 'Regression'
  },
  'SVR': {
    import: 'from sklearn.svm import SVR',
    constructor: 'SVR()',
    library: 'sklearn',
    base_hyperparameters: {
      random_state: 'int',
    },
    type: 'Regression'
  },
  'LinearRegression': {
    import: 'from sklearn.linear_model import LinearRegression',
    constructor: 'LinearRegression()',
    library: 'sklearn',
    base_hyperparameters: {
      random_state: 'int',
    },
    type: 'Regression'
  },
  'XGBoostRegressor': {
    import: 'from xgboost import XGBRegressor',
    constructor: 'XGBRegressor(random_state=42)',
    library: 'xgboost',
    base_hyperparameters: {
      random_state: 'int',
    },
    type: 'Regression'
  },
  'NeuralNetworkClassifier': {
    import: 'from torch.nn import MLPClassifier',
    constructor: 'MLPClassifier(random_state=42, max_iter=1000)',
    library: 'pytorch',
    base_hyperparameters: {
      random_state: 'int',
    },
    type: 'Classification'
  },
  'NeuralNetworkRegressor': {
    import: 'from torch.nn import MLPRegressor',
    constructor: 'MLPRegressor(random_state=42, max_iter=1000)',
    library: 'pytorch',
    base_hyperparameters: {
      random_state: 'int',
    },
    type: 'Regression'
  }
};