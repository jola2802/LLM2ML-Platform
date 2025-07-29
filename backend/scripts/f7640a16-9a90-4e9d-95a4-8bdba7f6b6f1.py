import pandas as pd
import joblib
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Starte das Machine Learning Projekt: Social Media Nutzungs- und Suchtstudie")
print("-" * 80)

# --- Konfiguration ---
CSV_FILE_PATH = r"C:\Users\jonas\Desktop\Text2ML\ML-Platform\backend\uploads\1753782396947-704141112.csv"
TARGET_VARIABLE = "Addicted_Score"
MODEL_OUTPUT_PATH = "model.pkl"

# Hyperparameter für XGBoostRegressor
XGBOOST_HYPERPARAMETERS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42  # Für Reproduzierbarkeit
}

print(f"Lade Daten von: {CSV_FILE_PATH}")
print(f"Zielvariable: {TARGET_VARIABLE}")
print(f"Modell-Typ: XGBoostRegressor mit Hyperparametern: {XGBOOST_HYPERPARAMETERS}")
print("-" * 80)

# --- 1. Daten laden und intelligente Datenbereinigung ---
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print("Daten erfolgreich geladen.")
    print(f"Anzahl der Zeilen: {len(df)}")
    print("Erste 5 Zeilen der Daten:")
    print(df.head())
    print("\nDaten-Typen der Spalten:")
    print(df.info())
    print("-" * 80)
except FileNotFoundError:
    print(f"Fehler: Die Datei {CSV_FILE_PATH} wurde nicht gefunden.")
    exit()
except Exception as e:
    print(f"Fehler beim Laden der Daten: {e}")
    exit()

# Überprüfung auf fehlende Werte
print("Überprüfe auf fehlende Werte:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

if missing_values.sum() > 0:
    print("\nFehlende Werte gefunden. Führe intelligente Imputation durch.")
    # Imputation: Numerische Spalten mit dem Median, Kategoriale Spalten mit dem Modus
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  - Fehlende Werte in numerischer Spalte '{col}' mit Median ({median_val}) gefüllt.")
            else:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"  - Fehlende Werte in kategorialer Spalte '{col}' mit Modus ('{mode_val}') gefüllt.")
    print("Imputation abgeschlossen.")
    print("Neue Summe der fehlenden Werte:", df.isnull().sum().sum())
else:
    print("Keine fehlenden Werte in den Daten gefunden.")
print("-" * 80)

# Entfernen der 'Student_ID'-Spalte, da sie ein reiner Identifier ist
if 'Student_ID' in df.columns:
    df = df.drop(columns=['Student_ID'])
    print("Spalte 'Student_ID' entfernt.")
else:
    print("Spalte 'Student_ID' nicht gefunden, überspringe das Entfernen.")
print("-" * 80)

# --- 2. Implementierung der vollständigen Preprocessing-Pipeline ---
# Definition der Features und der Zielvariablen
X = df.drop(columns=[TARGET_VARIABLE])
y = df[TARGET_VARIABLE]

# Identifizierung numerischer und kategorialer Features
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"Numerische Features: {numeric_features}")
print(f"Kategoriale Features: {categorical_features}")

# Erstellung der Preprocessing-Pipelines für numerische und kategoriale Features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # 'ignore' verhindert Fehler bei unbekannten Kategorien im Testset
])

# Zusammenfassung der Preprocessing-Schritte mit ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("Preprocessing-Pipeline (StandardScaler für numerische, OneHotEncoder für kategoriale Features) erstellt.")
print("-" * 80)

# --- 3. Train-Test-Split ---
print("Führe Train-Test-Split durch (80% Training, 20% Test)...")
# Beachte: Bei nur 49 Zeilen ist der Test-Split sehr klein (~10 Zeilen).
# Die Performance-Metriken auf einem so kleinen Testset können eine hohe Varianz aufweisen.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Trainingsdaten-Größe: {X_train.shape[0]} Zeilen")
print(f"Testdaten-Größe: {X_test.shape[0]} Zeilen")
print("-" * 80)

# --- 4. Modelltraining ---
print("Initialisiere XGBoostRegressor-Modell...")
model = XGBRegressor(**XGBOOST_HYPERPARAMETERS)

# Erstelle die vollständige Pipeline, die Preprocessing und das Modell kombiniert
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

print("Starte das Modelltraining...")
full_pipeline.fit(X_train, y_train)
print("Modelltraining abgeschlossen.")
print("-" * 80)

# --- 5. Modellbewertung ---
print("Evaluiere das Modell auf den Testdaten...")
y_pred = full_pipeline.predict(X_test)

# Berechne relevante Regression-Metriken
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Performance-Metriken auf dem Testset:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print("-" * 80)

# --- 6. Modell speichern ---
print(f"Speichere das trainierte Modell als '{MODEL_OUTPUT_PATH}'...")
try:
    with open(MODEL_OUTPUT_PATH, 'wb') as file:
        pickle.dump(full_pipeline, file)
    print(f"Modell erfolgreich gespeichert unter '{MODEL_OUTPUT_PATH}'.")
except Exception as e:
    print(f"Fehler beim Speichern des Modells: {e}")
print("-" * 80)

# --- 7. Label-Encoder speichern (falls nötig) ---
# Für dieses Regressionsprojekt ist keine separate Speicherung eines Target-Encoders notwendig,
# da die Zielvariable 'Addicted_Score' bereits numerisch ist und nicht encodiert werden muss.
print("Hinweis: Für ein Regressionsmodell mit numerischer Zielvariable ist kein 'target_encoder.pkl' erforderlich.")
print("-" * 80)

print("Machine Learning Projekt abgeschlossen.")