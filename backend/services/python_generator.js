import { callLLMAPI } from './llm.js';
import { analyzeCsvFile } from './llm_api.js';

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
export async function generatePythonScriptWithLLM(project) {
  try {
    // CSV-Datei analysieren für zusätzlichen Kontext
    const csvAnalysis = await analyzeCsvFile(project.csvFilePath);
    
    const prompt = `Du bist ein extrem erfahrener Machine Learning Engineer und Python Programmierer. Generiere ein vollständiges, ausführbares Python-Script für das folgende ML-Projekt.

PROJEKT-DETAILS:
- Name: ${project.name}
- Algorithmus: ${project.algorithm}
- Model-Typ: ${project.modelType}
- Target Variable: ${project.targetVariable}
- Features: ${project.features.join(', ')}
- Hyperparameter: ${JSON.stringify(project.hyperparameters)}

DATEN-ANALYSE:
- LLM-Analyse: ${csvAnalysis.llm_analysis}
- CSV-Pfad: ${project.csvFilePath}
- Anzahl Zeilen: ${csvAnalysis.rowCount}
- Spalten: ${csvAnalysis.columns.join(', ')}
- Datentypen: ${Object.entries(csvAnalysis.dataTypes).map(([col, type]) => `${col}: ${type}`).join(', ')}

BEISPIEL-DATEN (erste 10 Zeilen):
${csvAnalysis.sampleData.slice(0, 10).map((row, i) => 
  `Zeile ${i+1}: ${csvAnalysis.columns.map((col, j) => `${col}=${row[j]}`).join(', ')}`
).join('\n')}

ANFORDERUNGEN:
1. Lade die CSV-Datei und führe intelligente Datenbereinigung durch
2. Implementiere eine vollständige Preprocessing-Pipeline (Skalierung, Encoding, etc.)
3. Verwende den angegebenen Algorithmus mit den Hyperparametern
4. Führe ein ordentliches Train-Test-Split durch
5. Trainiere das Modell und berechne relevante Metriken (mindestens 3 Metriken, gerne so viele wie sinnvoll sind)
6. Speichere das trainierte Modell als 'model.pkl'
7. Speichere Label-Encoder falls nötig als 'target_encoder.pkl'
8. Gib detaillierte Logs und Performance-Metriken aus

ALGORITHMUS-MAPPING:
- RandomForestClassifier: from sklearn.ensemble import RandomForestClassifier
- LogisticRegression: from sklearn.linear_model import LogisticRegression  
- SVM: from sklearn.svm import SVC
- XGBoostClassifier: from xgboost import XGBClassifier
- RandomForestRegressor: from sklearn.ensemble import RandomForestRegressor
- LinearRegression: from sklearn.linear_model import LinearRegression
- SVR: from sklearn.svm import SVR
- XGBoostRegressor: from xgboost import XGBRegressor

WICHTIGE REGELN:
- Verwende IMMER r"${project.csvFilePath}" für den CSV-Pfad
- Gib Performance-Metriken in diesem Format aus: "Accuracy: 0.8524" (für Parsing)
- Behandle fehlende Werte intelligent je nach Datentyp
- Verwende scikit-learn Pipelines für sauberen Code
- Füge ausführliche Kommentare und print-Statements hinzu
- Das Script muss ohne weitere Eingaben ausführbar sein. Füge also keine sonstigen Eingaben hinzu.

Generiere ein vollständiges Python-Script (nur Code, keine Markdown-Formatierung):`;

    // Rufe das LLM API auf
    const response = await callLLMAPI(prompt);
    
    return response;
    
  } catch (error) {
    console.error('Fehler bei LLM Script-Generierung:', error);
    // Fallback auf Template-basierte Generierung
    return generatePythonScriptTemplate(project);
  }
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
  
  // Hyperparameter in Algorithmus-Constructor integrieren
  let algorithmConstructor = algoConfig.constructor;
  if (hyperparameters && Object.keys(hyperparameters).length > 0) {
    // Parse existierende Parameter aus dem Constructor
    const baseParams = algorithmConstructor.match(/\((.*)\)/)?.[1] || '';
    const customParams = Object.entries(hyperparameters)
      .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
      .join(', ');
    
    // Kombiniere base und custom Parameter  
    const allParams = baseParams ? `${baseParams}, ${customParams}` : customParams;
    algorithmConstructor = algorithmConstructor.replace(/\(.*\)/, `(${allParams})`);
  }
  
  return `
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
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
warnings.filterwarnings('ignore')

# Echte CSV-Daten laden
print("=== Lade CSV-Datei ===")
csv_path = r"${csvFilePath}"
print(f"CSV-Pfad: {csv_path}")

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
        print(f"CSV-Datei gefunden unter: {actual_path}")
        break

if actual_path is None:
    print("FEHLER: Die Datei wurde unter folgenden Pfaden gesucht:")
    for p in possible_paths:
        print(f"  - {p}")
    raise FileNotFoundError(f"CSV-Datei unter keinem der möglichen Pfade gefunden")

df = pd.read_csv(actual_path)
print(f"CSV-Datei erfolgreich geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")

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

# Preprocessing Pipeline erstellen
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
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

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=${isClassification ? 'y' : 'None'})

print(f"Training Set: {X_train.shape}")
print(f"Test Set: {X_test.shape}")

# Komplette Pipeline mit Preprocessing und Model
step_name = 'classifier' if ${isClassification ? 'True' : 'False'} else 'regressor'
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    (step_name, ${algorithmConstructor})
])

print("=== Training ===")
print(f"Verwendeter Algorithmus: ${selectedAlgorithm}")
pipeline.fit(X_train, y_train)
print("Training abgeschlossen")

# Vorhersagen
y_pred = pipeline.predict(X_test)
${isClassification ? 'y_pred_proba = pipeline.predict_proba(X_test) if hasattr(pipeline.named_steps["classifier"], "predict_proba") else None' : ''}

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
    joblib.dump(target_encoder, 'target_encoder.pkl')
    print("Target-Encoder gespeichert als target_encoder.pkl")
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
`;
} 