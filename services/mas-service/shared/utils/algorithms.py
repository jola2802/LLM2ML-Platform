"""
Algorithmus-Definitionen mit Standard-Hyperparametern
"""

ALGORITHMS = {
    # Scikit-learn Klassifikation
    'RandomForestClassifier': {
        'library': 'sklearn',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt'
        }
    },
    'LogisticRegression': {
        'library': 'sklearn',
        'hyperparameters': {
            'max_iter': 1000,
            'C': 1.0,
            'solver': 'lbfgs'
        }
    },
    'SVC': {
        'library': 'sklearn',
        'hyperparameters': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale'
        }
    },
    'GradientBoostingClassifier': {
        'library': 'sklearn',
        'hyperparameters': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2
        }
    },
    'KNeighborsClassifier': {
        'library': 'sklearn',
        'hyperparameters': {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto'
        }
    },
    'DecisionTreeClassifier': {
        'library': 'sklearn',
        'hyperparameters': {
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
    },
    
    # Scikit-learn Regression
    'RandomForestRegressor': {
        'library': 'sklearn',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt'
        }
    },
    'LinearRegression': {
        'library': 'sklearn',
        'hyperparameters': {}
    },
    'Ridge': {
        'library': 'sklearn',
        'hyperparameters': {
            'alpha': 1.0
        }
    },
    'Lasso': {
        'library': 'sklearn',
        'hyperparameters': {
            'alpha': 1.0
        }
    },
    'GradientBoostingRegressor': {
        'library': 'sklearn',
        'hyperparameters': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2
        }
    },
    'SVR': {
        'library': 'sklearn',
        'hyperparameters': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale'
        }
    },
    
    # XGBoost
    'XGBoostClassifier': {
        'library': 'xgboost',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0
        }
    },
    'XGBoostRegressor': {
        'library': 'xgboost',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0
        }
    },
    
    # PyTorch Neural Networks
    'NeuralNetwork': {
        'library': 'pytorch',
        'hyperparameters': {
            'nn_hidden_layers': [64, 32],
            'learning_rate': 0.001,
            'batch_size': 16,
            'num_epochs': 10
        }
    }
}

