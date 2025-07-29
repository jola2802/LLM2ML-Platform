import pandas as pd
import joblib
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Projekt-Details und Konfiguration ---
PROJECT_NAME = "Iris Flower Dataset - Classification Model"
CSV_PATH = r"C:\Users\jonas\Desktop\Text2ML\ML-Platform\backend\uploads\1753784774204-748123980.csv"
TARGET_VARIABLE = "species"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
HYPERPARAMETERS_JSON_STR = "{\"n_estimators\":100,\"max_depth\":10,\"random_state\":42,\"min_samples_leaf\":3}"
ALGORITHM_NAME = "RandomForestClassifier"

print(f"--- Starte ML-Projekt: {PROJECT_NAME} ---")
print(f"Verwendeter Algorithmus: {ALGORITHM_NAME}")
print(f"Zielvariable: {TARGET_VARIABLE}")
print(f"Features: {', '.join(FEATURES)}")
print(f"Hyperparameter: {HYPERPARAMETERS_JSON_STR}\n")

# --- 1. Daten laden und intelligente Datenbereinigung ---
try:
    df = pd.read_csv(CSV_PATH)
    print(f"Schritt 1: Daten erfolgreich von '{CSV_PATH}' geladen.")
    print(f"Anzahl der Zeilen: {df.shape[0]}, Anzahl der Spalten: {df.shape[1]}")
    print("\nErste 5 Zeilen des Datasets:")
    print(df.head())
    print("\nDatentypen der Spalten:")
    print(df.info())

    # Intelligente Datenbereinigung: Überprüfung auf fehlende Werte
    print("\nÜberprüfung auf fehlende Werte...")
    missing_values = df.isnull().sum()
    missing_values_features = missing_values[FEATURES]
    missing_values_target = missing_values[TARGET_VARIABLE]

    if missing_values_features.sum() == 0 and missing_values_target == 0:
        print("Es wurden keine fehlenden Werte in Features oder Zielvariable gefunden. Das Dataset ist sehr sauber.")
    else:
        print("Fehlende Werte gefunden. Führe intelligente Imputation durch.")
        # Numerische Features: Imputation mit Median (robuster gegenüber Ausreißern als Mean)
        for col in FEATURES:
            if missing_values[col] > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"  - Fehlende Werte in '{col}' (numerisch) mit Median ({median_val}) imputiert.")
                # Für den Fall, dass ein Feature fälschlicherweise als numerisch erkannt wird, aber kategorisch ist
                elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
                    print(f"  - Fehlende Werte in '{col}' (kategorisch) mit Modus ('{mode_val}') imputiert.")
        # Zielvariable: Imputation mit Modus (für kategorische Variablen)
        if missing_values_target > 0:
            mode_target = df[TARGET_VARIABLE].mode()[0]
            df[TARGET_VARIABLE].fillna(mode_target, inplace=True)
            print(f"  - Fehlende Werte in '{TARGET_VARIABLE}' (Ziel) mit Modus ('{mode_target}') imputiert.")
    
    print("\nDatenbereinigung abgeschlossen. Zustand des Datasets nach Bereinigung:")
    print(df.isnull().sum())

except FileNotFoundError:
    print(f"Fehler: Die Datei '{CSV_PATH}' wurde nicht gefunden. Bitte überprüfen Sie den Pfad.")
    exit()
except Exception as e:
    print(f"Fehler beim Laden oder Bereinigen der Daten: {e}")
    exit()

# --- 2. Implementierung der vollständigen Preprocessing-Pipeline ---
print("\nSchritt 2: Starte Daten-Preprocessing-Pipeline.")

# Separate Features (X) und Zielvariable (y)
X = df[FEATURES]
y = df[TARGET_VARIABLE]

# Label Encoding für die Zielvariable (y)
# Dies muss außerhalb des ColumnTransformers geschehen, da dieser nur auf X operiert.
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print(f"Zielvariable '{TARGET_VARIABLE}' wurde von kategorisch zu numerisch encodiert.")
print(f"Mapping der Zielklassen: {list(target_encoder.classes_)} -> {np.unique(y_encoded)}")

# Definiere numerische und kategorische Spalten für den ColumnTransformer
# Im Iris-Dataset sind alle Features numerisch
numeric_features = FEATURES
categorical_features = [] # Es gibt keine kategorischen Features in X für dieses Projekt

# Erstelle den ColumnTransformer für die Preprocessing-Schritte
# Für numerische Features: Skalierung mittels StandardScaler
# Für kategorische Features: (Keine in diesem Projekt, aber für Vollständigkeit hier gezeigt)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
        # ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # Beispiel für kategorische Features
    ],
    remainder='passthrough' # Behält andere Spalten bei, die nicht transformiert werden sollen
)
print("ColumnTransformer für Skalierung der numerischen Features erstellt.")

# Lade Hyperparameter aus dem JSON-String
try:
    hyperparameters = json.loads(HYPERPARAMETERS_JSON_STR)
    print(f"Hyperparameter erfolgreich geladen: {hyperparameters}")
except json.JSONDecodeError as e:
    print(f"Fehler beim Parsen der Hyperparameter-JSON: {e}")
    exit()

# Modell-Initialisierung
# Der Algorithmus ist RandomForestClassifier, wie in den Projekt-Details angegeben
model = RandomForestClassifier(**hyperparameters)
print(f"Modell '{ALGORITHM_NAME}' mit den angegebenen Hyperparametern initialisiert.")

# Erstelle die vollständige Pipeline
# Die Pipeline kombiniert Preprocessing (ColumnTransformer) und das Modell
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])
print("Vollständige scikit-learn Pipeline (Preprocessing + Modell) erstellt.")

# --- 3. Train-Test-Split ---
print("\nSchritt 3: Führe Train-Test-Split durch.")
# Verwende Stratified K-Fold, um sicherzustellen, dass die Klassenverteilung in Train- und Testsets erhalten bleibt
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"Trainingsdaten-Shape: {X_train.shape}, Testdaten-Shape: {X_test.shape}")
print(f"Trainings-Labels-Shape: {y_train.shape}, Test-Labels-Shape: {y_test.shape}")
print(f"Klassenverteilung im Trainingsset (Counts): {np.bincount(y_train)}")
print(f"Klassenverteilung im Testset (Counts): {np.bincount(y_test)}")


# --- 4. Modelltraining ---
print("\nSchritt 4: Starte das Modelltraining...")
pipeline.fit(X_train, y_train)
print("Modelltraining erfolgreich abgeschlossen.")

# --- 5. Performance-Metriken berechnen ---
print("\nSchritt 5: Berechne Performance-Metriken.")

y_pred = pipeline.predict(X_test)

# Berechne Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}") # Wichtig: Format für Parsing wie gefordert

# Berechne und drucke den vollständigen Klassifizierungsbericht
print("\nDetaillierter Klassifizierungsbericht:")
# target_names müssen die originalen String-Label sein
classification_rep = classification_report(y_test, y_pred, target_names=target_encoder.classes_)
print(classification_rep)

# --- 6. Modell speichern ---
print("\nSchritt 6: Speichere das trainierte Modell und den Label-Encoder.")

# Speichere das gesamte Pipeline-Objekt
model_filename = 'model.pkl'
try:
    with open(model_filename, 'wb') as file:
        pickle.dump(pipeline, file)
    print(f"Modell erfolgreich als '{model_filename}' gespeichert.")
except Exception as e:
    print(f"Fehler beim Speichern des Modells: {e}")

# Speichere den LabelEncoder
target_encoder_filename = 'target_encoder.pkl'
try:
    with open(target_encoder_filename, 'wb') as file:
        pickle.dump(target_encoder, file)
    print(f"LabelEncoder erfolgreich als '{target_encoder_filename}' gespeichert.")
except Exception as e:
    print(f"Fehler beim Speichern des LabelEncoders: {e}")

print("\n--- ML-Projekt abgeschlossen ---")