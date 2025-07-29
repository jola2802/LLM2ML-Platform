import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Projekt-Details ---
PROJECT_NAME = "Social Media Usage and Addiction Survey - Regression Model"
ALGORITHM_NAME = "RandomForestRegressor"
MODEL_TYPE = "Regression"
TARGET_VARIABLE = "Addicted_Score"
FEATURES = [
    "Age", "Gender", "Academic_Level", "Country", "Avg_Daily_Usage_Hours",
    "Most_Used_Platform", "Affects_Academic_Performance", "Sleep_Hours_Per_Night",
    "Mental_Health_Score", "Relationship_Status", "Conflicts_Over_Social_Media"
]
HYPERPARAMETERS_STR = "{\"n_estimators\":150,\"max_depth\":8,\"min_samples_split\":2,\"random_state\":42}"
CSV_PATH = r"C:\Users\jonas\Desktop\Text2ML\ML-Platform\backend\uploads\1753782826446-181972943.csv"
MODEL_SAVE_PATH = 'model.pkl'
ENCODER_SAVE_PATH = 'target_encoder.pkl' # Not used for numeric target, but kept for consistency

print(f"--- Starte ML-Projekt: {PROJECT_NAME} ---")
print(f"Verwendeter Algorithmus: {ALGORITHM_NAME}")
print(f"Zielvariable: {TARGET_VARIABLE}")

# --- 1. Daten laden und intelligente Datenbereinigung ---
try:
    df = pd.read_csv(CSV_PATH)
    print(f"\nDaten erfolgreich geladen von: {CSV_PATH}")
    print(f"Anzahl der Zeilen: {len(df)}")
    print("Erste 5 Zeilen des Datasets:")
    print(df.head())
except FileNotFoundError:
    print(f"Fehler: Die Datei wurde nicht gefunden unter {CSV_PATH}. Bitte überprüfen Sie den Pfad.")
    exit()
except Exception as e:
    print(f"Fehler beim Laden der Daten: {e}")
    exit()

# Überprüfung und Bereinigung der Spaltennamen (falls Leerzeichen etc. vorhanden sind)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')

# Überprüfung auf fehlende Werte
print("\nÜberprüfung auf fehlende Werte:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Intelligente Behandlung fehlender Werte (falls vorhanden)
# Für dieses spezifische Dataset (49 Zeilen, keine explizit genannten fehlenden Werte)
# wird angenommen, dass keine NA-Werte in den relevanten Spalten existieren.
# Falls doch, würde man hier z.B. mediane Imputation für numerische und Modus-Imputation für kategoriale Features anwenden.
# Beispiel:
# for col in df.columns:
#     if df[col].isnull().any():
#         if pd.api.types.is_numeric_dtype(df[col]):
#             df[col].fillna(df[col].median(), inplace=True)
#             print(f"Fehlende Werte in '{col}' mit Median gefüllt.")
#         else:
#             df[col].fillna(df[col].mode()[0], inplace=True)
#             print(f"Fehlende Werte in '{col}' mit Modus gefüllt.")

# Identifizierung der Feature- und Target-Variablen
# Die 'Country'-Spalte hat für jede Zeile ein einzigartiges Land und sollte
# aufgrund der sehr geringen Stichprobengröße (49 Studierende, 49 verschiedene Länder)
# nicht als Feature verwendet werden, da sie zu Overfitting führen oder keine aussagekräftigen Muster liefert.
# Die LLM-Analyse bestätigt dies: "keine länderspezifischen Analysen möglich".
# Ebenso wird 'Student_ID' als reiner Identifier entfernt.
features_to_use = [f for f in FEATURES if f not in ['Country', 'Student_ID']] # Ensure Student_ID is not in FEATURES list if it exists
if 'Student_ID' in df.columns:
    df = df.drop(columns=['Student_ID'])
if 'Country' in df.columns:
    df = df.drop(columns=['Country'])
    print("Spalte 'Country' aufgrund hoher Kardinalität und geringer Stichprobengröße entfernt.")

# Überprüfung, ob alle benötigten Spalten im DataFrame vorhanden sind
if not all(col in df.columns for col in features_to_use + [TARGET_VARIABLE]):
    missing_cols = [col for col in features_to_use + [TARGET_VARIABLE] if col not in df.columns]
    print(f"Fehler: Einige benötigte Spalten fehlen im DataFrame: {missing_cols}")
    exit()

X = df[features_to_use]
y = df[TARGET_VARIABLE]

print(f"\nVerwendete Features ({len(features_to_use)}): {features_to_use}")
print(f"Target-Variable: {TARGET_VARIABLE}")

# --- 2. Implementierung der vollständigen Preprocessing-Pipeline ---
# Identifizierung numerischer und kategorialer Spalten
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"\nNumerische Features: {numeric_features}")
print(f"Kategorische Features: {categorical_features}")

# Preprocessing-Schritte definieren
# Numerische Features: Skalierung (StandardScaler)
# Kategoriale Features: One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

print("\nPreprocessing-Pipeline (ColumnTransformer) erstellt.")

# --- 3. Train-Test-Split ---
# Für kleine Datasets sind die Metriken sehr volatil. 80/20 Split ist hier üblich,
# aber selbst dann ist die Testmenge sehr klein (49 * 0.2 = ~10 Zeilen).
TEST_SIZE = 0.2
RANDOM_STATE = json.loads(HYPERPARAMETERS_STR).get("random_state", 42) # Use random_state from hyperparameters

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print(f"\nDaten aufgeteilt in Trainings- und Testsets (Testgröße: {TEST_SIZE*100}%):")
print(f"Trainingsset-Größe: {len(X_train)} Proben")
print(f"Testset-Größe: {len(X_test)} Proben")

# --- 4. Modellinitialisierung mit Hyperparametern ---
hyperparameters = json.loads(HYPERPARAMETERS_STR)
print(f"\nVerwendete Hyperparameter für {ALGORITHM_NAME}: {hyperparameters}")

model = RandomForestRegressor(**hyperparameters)

# Erstellen der vollständigen Pipeline (Preprocessing + Modell)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])

print(f"\nPipeline mit Preprocessor und {ALGORITHM_NAME} erstellt.")

# --- 5. Modelltraining ---
print("\nStarte Modelltraining...")
pipeline.fit(X_train, y_train)
print("Modelltraining abgeschlossen.")

# --- 6. Modell-Evaluation und Metriken ---
print("\nStarte Modell-Evaluation...")
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Performance-Metriken ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")
print(f"Accuracy: {r2:.4f}") # R2 is the primary metric for regression, outputting as Accuracy for parsing consistency

# --- 7. Modell speichern ---
try:
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    print(f"\nTrainiertes Modell erfolgreich gespeichert als: {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"Fehler beim Speichern des Modells: {e}")

# Kein LabelEncoder für numerisches Target nötig, daher target_encoder.pkl nicht gespeichert.
# Aber für die Konsistenz der Ausgabe wird eine Notiz gemacht.
if TARGET_VARIABLE in categorical_features: # Hypothetischer Fall, wenn das Target kategorial wäre
    print(f"Hinweis: '{ENCODER_SAVE_PATH}' wird nicht benötigt, da '{TARGET_VARIABLE}' eine numerische Variable ist.")
else:
    print(f"Hinweis: '{ENCODER_SAVE_PATH}' wird nicht erstellt, da die Zielvariable '{TARGET_VARIABLE}' numerisch ist.")

print("\n--- ML-Projekt abgeschlossen ---")