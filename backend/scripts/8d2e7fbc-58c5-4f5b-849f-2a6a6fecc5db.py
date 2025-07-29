import pandas as pd
import numpy as np
import joblib
import json
import re # For regex operations in data cleaning

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# --- Konfigurationskonstanten ---
CSV_PATH = r"C:\Users\jonas\Desktop\Text2ML\ML-Platform\backend\uploads\1753801692299-365564900.csv"
MODEL_SAVE_PATH = 'model.pkl'

TARGET_VARIABLE = 'Sale_Amount'
FEATURES = [
    'Clicks', 'Impressions', 'Cost', 'Leads', 'Conversions', 'Conversion Rate',
    'Campaign_Name', 'Ad_Date', 'Location', 'Device', 'Keyword'
]
HYPERPARAMETERS_STR = "{\"n_estimators\":300,\"learning_rate\":0.05,\"max_depth\":5,\"subsample\":0.8,\"colsample_bytree\":0.8}"
ALGORITHM_NAME = 'XGBRegressor' # Für Logging-Zwecke

print(f"[{ALGORITHM_NAME}] Starte ML-Projekt: Ad_Campaign_Performance_Analysis - Regression Model")
print(f"[{ALGORITHM_NAME}] CSV-Pfad: {CSV_PATH}")

# --- 1. Lade die CSV-Datei und führe intelligente Datenbereinigung durch ---
try:
    df = pd.read_csv(CSV_PATH)
    print(f"[{ALGORITHM_NAME}] Daten erfolgreich von {CSV_PATH} geladen. Ursprüngliche Form: {df.shape}")
except FileNotFoundError:
    print(f"[{ALGORITHM_NAME}] Fehler: CSV-Datei nicht gefunden unter {CSV_PATH}. Bitte prüfen Sie den Pfad.")
    exit()
except Exception as e:
    print(f"[{ALGORITHM_NAME}] Fehler beim Laden der CSV: {e}")
    exit()

print(f"[{ALGORITHM_NAME}] Initialer Datenüberblick:")
df.info()

print(f"[{ALGORITHM_NAME}] Führe intelligente Datenbereinigung durch...")

# Bereinigung der 'Cost'-Spalte: Entferne '$' und Kommas, konvertiere zu numerischem Typ
if 'Cost' in df.columns:
    print(f"[{ALGORITHM_NAME}] Bereinige Spalte 'Cost'...")
    # Ersetze '$' und Kommas durch leere Strings, dann konvertiere zu numerischem Typ. Fehler werden zu NaN.
    df['Cost'] = df['Cost'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
    df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
    print(f"[{ALGORITHM_NAME}] Spalte 'Cost' bereinigt. Eingeführte NaNs: {df['Cost'].isnull().sum()}")

# Bereinigung der 'Sale_Amount'-Spalte (Zielvariable): Entferne '$' und Kommas, konvertiere zu numerischem Typ
if TARGET_VARIABLE in df.columns:
    print(f"[{ALGORITHM_NAME}] Bereinige Spalte '{TARGET_VARIABLE}'...")
    # Ersetze '$' und Kommas durch leere Strings, dann konvertiere zu numerischem Typ. Fehler werden zu NaN.
    df[TARGET_VARIABLE] = df[TARGET_VARIABLE].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
    df[TARGET_VARIABLE] = pd.to_numeric(df[TARGET_VARIABLE], errors='coerce')
    print(f"[{ALGORITHM_NAME}] Spalte '{TARGET_VARIABLE}' bereinigt. Eingeführte NaNs: {df[TARGET_VARIABLE].isnull().sum()}")
    # Zeilen löschen, bei denen die Zielvariable NaN ist, da diese für das Training unbrauchbar sind
    initial_rows = df.shape[0]
    df.dropna(subset=[TARGET_VARIABLE], inplace=True)
    rows_dropped = initial_rows - df.shape[0]
    if rows_dropped > 0:
        print(f"[{ALGORITHM_NAME}] {rows_dropped} Zeilen aufgrund von NaN in '{TARGET_VARIABLE}' gelöscht. Aktuelle Form: {df.shape}")
else:
    print(f"[{ALGORITHM_NAME}] Fehler: Zielvariable '{TARGET_VARIABLE}' nicht im Datensatz gefunden.")
    exit()

# Bereinigung der 'Conversion Rate'-Spalte: Leere Strings zu NaN, dann zu numerischem Typ
if 'Conversion Rate' in df.columns:
    print(f"[{ALGORITHM_NAME}] Bereinige Spalte 'Conversion Rate'...")
    # Ersetze leere Strings (oder solche mit nur Whitespace) durch NaN, dann konvertiere zu numerischem Typ
    df['Conversion Rate'] = df['Conversion Rate'].replace(r'^\s*$', np.nan, regex=True)
    df['Conversion Rate'] = pd.to_numeric(df['Conversion Rate'], errors='coerce')
    print(f"[{ALGORITHM_NAME}] Spalte 'Conversion Rate' bereinigt. Eingeführte NaNs: {df['Conversion Rate'].isnull().sum()}")

# Standardisiere kategoriale Features (Kleinbuchstaben und Whitespace entfernen)
categorical_features_to_clean = ['Campaign_Name', 'Location', 'Device', 'Keyword']
for col in categorical_features_to_clean:
    if col in df.columns:
        print(f"[{ALGORITHM_NAME}] Standardisiere Spalte '{col}'...")
        # NaN-Werte mit einem leeren String füllen, bevor str-Methoden angewendet werden
        df[col] = df[col].fillna('').astype(str).str.lower().str.strip()
        print(f"[{ALGORITHM_NAME}] Eindeutige Werte in '{col}' nach der Bereinigung: {df[col].nunique()}")

# 'Ad_Date' als kategoriales Feature behandeln, indem das Format standardisiert wird
if 'Ad_Date' in df.columns:
    print(f"[{ALGORITHM_NAME}] Standardisiere Format der Spalte 'Ad_Date'...")
    # Funktion zum Parsen und Formatieren von Datumsstrings
    def parse_and_format_date(date_str):
        if pd.isna(date_str) or str(date_str).strip() == '':
            return np.nan # NaN oder leere Strings als NaN beibehalten
        try:
            # Versuche, das Datum mit verschiedenen gängigen Formaten zu parsen
            # Priorisiere YYYY-MM-DD und DD-MM-YYYY, dann YYYY/MM/DD
            parsed_date = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
            if pd.isna(parsed_date):
                parsed_date = pd.to_datetime(date_str, format='%d-%m-%Y', errors='coerce')
            if pd.isna(parsed_date):
                parsed_date = pd.to_datetime(date_str, format='%Y/%m/%d', errors='coerce')
            if pd.isna(parsed_date):
                # Fallback auf generischen Parser, falls spezifische Formate fehlschlagen
                parsed_date = pd.to_datetime(date_str, errors='coerce')
            
            if pd.isna(parsed_date):
                return np.nan # Wenn auch der generische Parser fehlschlägt
            else:
                return parsed_date.strftime('%Y-%m-%d') # Konsistentes Format YYYY-MM-DD
        except Exception:
            return np.nan # Bei jedem anderen Fehler während des Parsens
    
    df['Ad_Date'] = df['Ad_Date'].apply(parse_and_format_date)
    print(f"[{ALGORITHM_NAME}] Spalte 'Ad_Date' standardisiert. Eingeführte NaNs: {df['Ad_Date'].isnull().sum()}")

# 'Ad_ID' löschen, da es kein Feature ist
if 'Ad_ID' in df.columns:
    print(f"[{ALGORITHM_NAME}] Lösche Spalte 'Ad_ID' (kein Feature).")
    df.drop('Ad_ID', axis=1, inplace=True)

# Trenne Features (X) und Ziel (y)
X = df[FEATURES]
y = df[TARGET_VARIABLE]

print(f"[{ALGORITHM_NAME}] Form der Features (X): {X.shape}")
print(f"[{ALGORITHM_NAME}] Form des Ziels (y): {y.shape}")

# Identifiziere numerische und kategoriale Features für die Preprocessing-Pipeline
numerical_features = [f for f in FEATURES if pd.api.types.is_numeric_dtype(X[f])]
categorical_features = [f for f in FEATURES if not pd.api.types.is_numeric_dtype(X[f])]

print(f"[{ALGORITHM_NAME}] Numerische Features: {numerical_features}")
print(f"[{ALGORITHM_NAME}] Kategoriale Features: {categorical_features}")

# --- 2. Implementiere eine vollständige Preprocessing-Pipeline ---
print(f"[{ALGORITHM_NAME}] Richte Preprocessing-Pipeline ein...")

# Numerische Pipeline: Imputation mit Median, dann Skalierung
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Kategoriale Pipeline: Imputation mit dem häufigsten Wert, dann One-Hot-Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # 'handle_unknown='ignore' verhindert Fehler bei neuen Kategorien im Testset
])

# Erstelle einen Preprocessor mit ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Nicht spezifizierte Spalten werden gelöscht
)

# --- 4. Führe ein ordentliches Train-Test-Split durch ---
print(f"[{ALGORITHM_NAME}] Teile Daten in Trainings- und Testsets (80/20 Split)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"[{ALGORITHM_NAME}] X_train Form: {X_train.shape}, X_test Form: {X_test.shape}")
print(f"[{ALGORITHM_NAME}] y_train Form: {y_train.shape}, y_test Form: {y_test.shape}")

# --- 3. Verwende den angegebenen Algorithmus mit den Hyperparametern ---
print(f"[{ALGORITHM_NAME}] Initialisiere {ALGORITHM_NAME} mit den angegebenen Hyperparametern...")
hyperparameters = json.loads(HYPERPARAMETERS_STR)
model = XGBRegressor(**hyperparameters)

# Erstelle die vollständige Pipeline: Preprocessor + Modell
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', model)])

# --- 5. Trainiere das Modell ---
print(f"[{ALGORITHM_NAME}] Trainiere das Modell...")
full_pipeline.fit(X_train, y_train)
print(f"[{ALGORITHM_NAME}] Modelltraining abgeschlossen.")

# --- 6. Berechne relevante Metriken ---
print(f"[{ALGORITHM_NAME}] Bewerte die Modellleistung...")
y_pred = full_pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"[{ALGORITHM_NAME}] Leistungsmetriken:")
print(f"[{ALGORITHM_NAME}] Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"[{ALGORITHM_NAME}] Mean Absolute Error (MAE): {mae:.4f}")
print(f"[{ALGORITHM_NAME}] R-squared (R2): {r2:.4f}")

# Gib Performance-Metriken im angegebenen Format für das Parsen aus.
# Für Regression ist R-squared das gebräuchlichste "Genauigkeitsmaß".
print(f"Accuracy: {r2:.4f}")

# --- 7. Speichere das trainierte Modell ---
print(f"[{ALGORITHM_NAME}] Speichere das trainierte Modell in '{MODEL_SAVE_PATH}'...")
joblib.dump(full_pipeline, MODEL_SAVE_PATH)
print(f"[{ALGORITHM_NAME}] Modell erfolgreich gespeichert.")

# --- 8. Speichere Label-Encoder falls nötig ---
# Für eine numerische Regressionszielvariable ist kein Ziel-Encoder erforderlich.
# Wenn Sale_Amount eine kategoriale Variable wäre und LabelEncoding erfordern würde,
# würde der Encoder hier gespeichert.
print(f"[{ALGORITHM_NAME}] Kein Ziel-Encoder für numerische Regressionszielvariable erforderlich.")

print(f"[{ALGORITHM_NAME}] ML-Projekt-Ausführung abgeschlossen.")