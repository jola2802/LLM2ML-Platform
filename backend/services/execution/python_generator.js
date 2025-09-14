import { runNetworkAgentPipeline } from '../llm/master_agent.js';
import { getCachedDataAnalysis } from '../data/data_exploration.js';

// Python-Code Syntax-Validierung und Bereinigung
function cleanPythonCode(pythonCode) {
  try {
    // Entferne Markdown-Code-Blöcke falls vorhanden
    pythonCode = pythonCode.replace(/```python\s*/g, '');
    pythonCode = pythonCode.replace(/```\s*/g, '');
    
    // Entferne führende und nachfolgende Leerzeichen
    pythonCode = pythonCode.trim();
    
    // Normalisiere Zeilenendezeichen
    pythonCode = pythonCode.replace(/\r\n/g, '\n');
    
    // Teile in Zeilen auf
    let lines = pythonCode.split('\n');
    
    // Bereinige jede Zeile und korrigiere häufige Indentation-Probleme
    lines = lines.map((line, index) => {
      // Entferne Tabs und ersetze sie durch 4 Leerzeichen
      line = line.replace(/\t/g, '    ');
      
      // Korrigiere häufige Indentation-Probleme bei bestimmten Zeilen
      const trimmedLine = line.trim();
      
      // Spezielle Behandlung für train_test_split und ähnliche Zeilen
      if (trimmedLine.includes('train_test_split') && 
          lines[index - 1] && 
          !lines[index - 1].trim().endsWith(':') &&
          !lines[index - 1].trim().endsWith('\\')) {
        // Stelle sicher, dass diese Zeile auf derselben Ebene wie die vorherige ist
        const prevLineIndent = lines[index - 1].length - lines[index - 1].trimStart().length;
        line = ' '.repeat(prevLineIndent) + trimmedLine;
      }
      
      return line;
    });
    
    // Entferne leere Zeilen am Anfang und Ende
    while (lines.length > 0 && lines[0].trim() === '') {
      lines.shift();
    }
    while (lines.length > 0 && lines[lines.length - 1].trim() === '') {
      lines.pop();
    }
    
    return lines.join('\n');
    
  } catch (error) {
    console.error('Fehler bei Python-Code-Bereinigung:', error.message);
    return pythonCode; // Gib ursprünglichen Code zurück bei Fehlern
  }
}

// Verfügbare Algorithmen definieren
const ALGORITHMS = {
  // Klassifikation
  'RandomForestClassifier': {
    import: 'from sklearn.ensemble import RandomForestClassifier',
    constructor: 'RandomForestClassifier(n_estimators=100, random_state=42)',
    type: 'Classification'
  },
  'SVM': {
    import: 'from sklearn.svm import SVC',
    constructor: 'SVC(random_state=42, probability=True)',
    type: 'Classification'
  },
  'LogisticRegression': {
    import: 'from sklearn.linear_model import LogisticRegression',
    constructor: 'LogisticRegression(random_state=42, max_iter=1000)',
    type: 'Classification'
  },
  'XGBoostClassifier': {
    import: 'from xgboost import XGBClassifier',
    constructor: 'XGBClassifier(random_state=42, eval_metric="logloss")',
    type: 'Classification'
  },
  
  // Regression
  'RandomForestRegressor': {
    import: 'from sklearn.ensemble import RandomForestRegressor',
    constructor: 'RandomForestRegressor(n_estimators=100, random_state=42)',
    type: 'Regression'
  },
  'SVR': {
    import: 'from sklearn.svm import SVR',
    constructor: 'SVR()',
    type: 'Regression'
  },
  'LinearRegression': {
    import: 'from sklearn.linear_model import LinearRegression',
    constructor: 'LinearRegression()',
    type: 'Regression'
  },
  'XGBoostRegressor': {
    import: 'from xgboost import XGBRegressor',
    constructor: 'XGBRegressor(random_state=42)',
    type: 'Regression'
  },
  'NeuralNetworkClassifier': {
    import: 'from sklearn.neural_network import MLPClassifier',
    constructor: 'MLPClassifier(random_state=42, max_iter=1000)',
    type: 'Classification'
  },
  'NeuralNetworkRegressor': {
    import: 'from sklearn.neural_network import MLPRegressor',
    constructor: 'MLPRegressor(random_state=42, max_iter=1000)',
    type: 'Regression'
  }
};

// LLM-basierter Python Script Generator 
export async function generatePythonScriptWithLLM(project, maxRetries = 3) {
  let attempt = 0;
  
  while (attempt < maxRetries) {
    try {
      attempt++;
      console.log(`Python Script Generierung - Versuch ${attempt}/${maxRetries}`);
      
      // Automatische Datenexploration durchführen
      const dataAnalysis = await getCachedDataAnalysis(project.csvFilePath);
      
      if (!dataAnalysis.success) {
        console.error('Automatische Datenanalyse fehlgeschlagen, verwende Fallback');
        return generatePythonScriptTemplate(project);
      }

      console.log('🤖 Verwende Netzwerk-Agent-Pipeline...');
      let pythonCode = await runNetworkAgentPipeline(project);
      
      // Validiere den generierten Code
      if (!pythonCode || typeof pythonCode !== 'string') {
        throw new Error('Leerer oder ungültiger Python-Code erhalten');
      }
      
      // Bereinige den Python-Code mit verbesserter Funktion
      pythonCode = cleanPythonCode(pythonCode);
      
      // Zusätzliche Syntax-Validierung für häufige Probleme
      const lines = pythonCode.split('\n');
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        
        // Prüfe auf häufige Indentation-Probleme
        if (line.trim().includes('train_test_split') && 
            i > 0 && 
            !lines[i-1].trim().endsWith(':') && 
            !lines[i-1].trim().endsWith('\\')) {
          
          // Warnung wenn train_test_split falsch eingerückt zu sein scheint
          const currentIndent = line.length - line.trimStart().length;
          const prevIndent = lines[i-1].length - lines[i-1].trimStart().length;
          
          if (Math.abs(currentIndent - prevIndent) > 4) {
            console.warn(`Mögliches Indentation-Problem in Zeile ${i+1}: train_test_split`);
          }
        }
      }
      
      // console.log(`Python Script erfolgreich generiert (Versuch ${attempt})`);
      return pythonCode;
      
    } catch (error) {
      console.error(`Fehler bei LLM Script-Generierung (Versuch ${attempt}):`, error.message);
      
      if (attempt < maxRetries) {
        // Warte kurz vor dem nächsten Versuch
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
        continue;
      } else {
        // Letzter Versuch fehlgeschlagen - Fallback auf Template
        console.error('Alle LLM-Versuche fehlgeschlagen, verwende Template-Fallback');
        return generatePythonScriptTemplate(project);
      }
    }
  }
  
  // Fallback: Sollte nie erreicht werden
  return generatePythonScriptTemplate(project);
}

// Fallback: Template-basierte Generierung (als Backup)
export function generatePythonScriptTemplate(project) {
  const { name, modelType, targetVariable, features, csvFilePath, algorithm, hyperparameters } = project;
  const isClassification = modelType === 'Classification';
  
  // Algorithmus bestimmen (Fallback auf RandomForest)
  let selectedAlgorithm = algorithm || (isClassification ? 'RandomForestClassifier' : 'RandomForestRegressor');
  
  // Überprüfen ob Algorithmus existiert und zum Typ passt
  if (!ALGORITHMS[selectedAlgorithm] || ALGORITHMS[selectedAlgorithm].type !== modelType) {
    selectedAlgorithm = isClassification ? 'RandomForestClassifier' : 'RandomForestRegressor';
  }
  
  const algoConfig = ALGORITHMS[selectedAlgorithm];
  
  // csvFilePath ist bereits ein absoluter Pfad von multer
  if (!csvFilePath) {
    throw new Error('Keine CSV-Datei für das Training verfügbar');
  }
  
  // Hyperparameter als JSON-String für das Template vorbereiten
  const hyperparametersJson = hyperparameters && Object.keys(hyperparameters).length > 0 
    ? JSON.stringify(hyperparameters).replace(/"/g, '\\"')
    : '{}';
  
  return `
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
${algoConfig.import}
${isClassification 
  ? 'from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix'
  : 'from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error'
}
import joblib
import os
import warnings
import json
from collections import Counter
warnings.filterwarnings('ignore')

def main():
    # Hyperparameter definieren
    hyperparameters = "${hyperparametersJson}"
    
    # Dateipfad definieren
    csv_path = r"${csvFilePath}"
    
    # Daten laden (unterstützt CSV/JSON/Excel)
    print("=== Lade Daten-Datei ===")
    print(f"Dateipfad: {csv_path}")

# Überprüfe verschiedene mögliche Pfade
possible_paths = [
    csv_path,
    os.path.join(os.getcwd(), csv_path),
    os.path.join(os.path.dirname(__file__), csv_path),
    os.path.join(os.path.dirname(__file__), '..', csv_path)
]

df = None
actual_path = None
for path_attempt in possible_paths:
    if os.path.exists(path_attempt):
        actual_path = path_attempt
        print(f"Datei gefunden unter: {actual_path}")
        break

if actual_path is None:
    print("FEHLER: Die Datei wurde unter folgenden Pfaden gesucht:")
    for p in possible_paths:
        print(f"  - {p}")
    raise FileNotFoundError(f"Datei unter keinem der möglichen Pfade gefunden")

file_ext = os.path.splitext(actual_path)[1].lower()
df = None
if file_ext == '.csv':
    df = pd.read_csv(actual_path)
elif file_ext == '.json':
    try:
        df = pd.read_json(actual_path)
    except Exception:
        df = pd.read_json(actual_path, lines=True)
elif file_ext in ['.xlsx', '.xls']:
    try:
        df = pd.read_excel(actual_path)
    except Exception:
        df = pd.read_excel(actual_path, engine='openpyxl')
else:
    # Fallback: Versuche als CSV mit automatischer Trennung
    try:
        df = pd.read_csv(actual_path, sep=None, engine='python')
    except Exception as e:
        raise ValueError(f"Dateiformat wird nicht unterstützt: {file_ext} ({e})")

print(f"Datei erfolgreich geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")

print("=== Daten Übersicht ===")
print(df.head())
print(f"Anzahl Zeilen: {len(df)}")
print("Verfügbare Spalten:", df.columns.tolist())
print()

# Überprüfen ob alle benötigten Spalten vorhanden sind
required_columns = ${JSON.stringify([...features, targetVariable])}
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Fehlende Spalten in der CSV-Datei: {missing_columns}")

# Features und Target definieren
X = df[${JSON.stringify(features)}]
y = df['${targetVariable}']

print("=== Erweiterte Datenbereinigung ===")
print(f"Originale Daten: {X.shape[0]} Zeilen, {X.shape[1]} Features")
print(f"Fehlende Werte in Features: {X.isnull().sum().sum()}")
print(f"Fehlende Werte in Target: {y.isnull().sum()}")

# Zeilen mit fehlenden Target-Werten entfernen
initial_rows = len(df)
mask = y.notna()
X = X[mask]
y = y[mask]
print(f"Nach Target-Bereinigung: {len(X)} Zeilen")

# Feature-Engineering: Datentypen identifizieren
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numerische Features: {numeric_features}")
print(f"Kategorische Features: {categorical_features}")

# Preprocessing Pipeline erstellen (mit Imputation)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Target-Variable preprocessing
${isClassification ? `
target_encoder = None
if y.dtype == 'object':
    print("=== Label Encoding für Target Variable ===")
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    print(f"Target-Klassen: {target_encoder.classes_}")
    
    # Target-Encoder speichern für später
    joblib.dump(target_encoder, 'target_encoder.pkl')
` : ''}

print(f"Finale Daten: Features={X.shape}, Target={y.shape}")
print()

    # Train-Test Split mit robuster Stratifikation
    stratify_var = None
    ${isClassification ? `
    # Nur stratify verwenden, wenn mindestens 2 Klassen und jede Klasse >= 2 Samples
    class_counts = Counter(y if not hasattr(y, 'to_numpy') else y.to_numpy())
    if len(class_counts) >= 2 and min(class_counts.values()) >= 2:
        stratify_var = y
    ` : ''}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_var
    )
    
    print(f"Training Set: {X_train.shape}")
    print(f"Test Set: {X_test.shape}")
    
    # Feature-Namen für Debugging (nicht für Vorhersagen verwendet)
    feature_names = list(X.columns)
    print(f"Feature-Namen: {feature_names}")

    # Hyperparameter laden und konvertieren
    hyperparams = json.loads(hyperparameters)
    
    # Hyperparameter korrekt konvertieren - erweitert für alle Algorithmen
    converted_hyperparams = {}
    for key, value in hyperparams.items():
        # Integer-Parameter
        if key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 
                   'n_neighbors', 'max_iter', 'hidden_layer_sizes']:
            if isinstance(value, str):
                converted_hyperparams[key] = int(value)
            else:
                converted_hyperparams[key] = value
        # Float-Parameter
        elif key in ['learning_rate', 'C', 'epsilon', 'learning_rate_init']:
            if isinstance(value, str):
                converted_hyperparams[key] = float(value)
            else:
                converted_hyperparams[key] = value
        # Boolean-Parameter
        elif key in ['fit_intercept']:
            if isinstance(value, str):
                converted_hyperparams[key] = value.lower() == 'true'
            else:
                converted_hyperparams[key] = bool(value)
        # String-Parameter (wie kernel, weights)
        else:
            converted_hyperparams[key] = value
    
    print(f"Verwendete Hyperparameter: {converted_hyperparams}")
    
    # Algorithmus mit Hyperparametern initialisieren
    model = ${algoConfig.constructor.replace(/\(/g, '(').replace(/\)/g, ')')}
    
    # Hyperparameter setzen
    for key, value in converted_hyperparams.items():
        if hasattr(model, key):
            setattr(model, key, value)

    # Komplette Pipeline mit Preprocessing und Model
    step_name = 'classifier' if ${isClassification ? 'True' : 'False'} else 'regressor'
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        (step_name, model)
    ])
    
    print("=== Training ===")
    print(f"Verwendeter Algorithmus: ${selectedAlgorithm}")
    
    # Training ohne Feature-Namen für konsistente Vorhersagen
    X_train_array = X_train.values
    X_test_array = X_test.values
    pipeline.fit(X_train_array, y_train)
    print("Training abgeschlossen")

# Vorhersagen
y_pred = pipeline.predict(X_test_array)
${isClassification ? 'y_pred_proba = pipeline.predict_proba(X_test_array) if hasattr(pipeline.named_steps["classifier"], "predict_proba") else None' : ''}

# Evaluation
print("=== Performance Metriken ===")
${isClassification ? `
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Zusätzliche Metriken je nach Algorithmus
try:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_test)) == 2 and y_pred_proba is not None:  # Binäre Klassifikation
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        print(f"AUC-ROC: {auc:.4f}")
except:
    pass

print("\\nClassification Report:")
print(classification_report(y_test, y_pred))
` : `
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

try:
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"Mean Absolute Percentage Error: {mape:.4f}")
except:
    pass

print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")
`}

# Komplette Pipeline speichern (inkl. Preprocessing)
model_path = 'model.pkl'
joblib.dump(pipeline, model_path)
print(f"Pipeline gespeichert als {model_path}")

# Zusätzlich prüfen ob Datei wirklich erstellt wurde
if os.path.exists(model_path):
    print(f"Model-Datei erfolgreich erstellt: {os.path.abspath(model_path)}")
    print(f"Model-Dateigröße: {os.path.getsize(model_path)} Bytes")
else:
    print("WARNUNG: Model-Datei wurde nicht erstellt!")

# Target-Encoder auch separat speichern (falls vorhanden)
${isClassification ? `
if target_encoder is not None:
    target_encoder_path = 'target_encoder_${project.id}.pkl'
    joblib.dump(target_encoder, target_encoder_path)
    print(f"Target-Encoder gespeichert als {target_encoder_path}")
` : ''}

# Feature-Importances falls verfügbar
try:
    if hasattr(pipeline.named_steps[step_name], 'feature_importances_'):
        print("\\n=== Feature Importances ===")
        # Feature-Namen nach Preprocessing abrufen
        feature_names = (numeric_features + 
                        [f"{cat}_encoded" for cat in categorical_features])
        importances = pipeline.named_steps[step_name].feature_importances_
        for name, importance in zip(feature_names[:len(importances)], importances):
            print(f"{name}: {importance:.4f}")
except Exception as e:
    print(f"Feature Importances nicht verfügbar: {e}")
    
print("\\n=== Training abgeschlossen ===")

if __name__ == "__main__":
    main()
`;
} 