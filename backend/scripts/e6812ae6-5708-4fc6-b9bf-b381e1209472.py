import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- Konfiguration des Projekts ---
CSV_PATH = r"C:\Users\jonas\Desktop\Text2ML\ML-Platform\LLM2ML-Platform\backend\uploads\1753811079525-832987270.csv"
TARGET_VARIABLE = "species"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
HYPERPARAMETERS_JSON = "{\"n_estimators\":100,\"max_depth\":10,\"random_state\":42}"
MODEL_FILENAME = "model.pkl"
TARGET_ENCODER_FILENAME = "target_encoder.pkl"
RANDOM_STATE = 42 # Für Reproduzierbarkeit von Train-Test-Split und Modellinitialisierung

print("--- ML-Projekt: Iris Blumen Klassifikationsmodell ---")
print(f"Algorithmus: RandomForestClassifier")
print(f"Zielvariable: {TARGET_VARIABLE}")
print(f"Features: {', '.join(FEATURES)}")
print(f"Verwendete Hyperparameter: {HYPERPARAMETERS_JSON}")
print(f"Pfad zur CSV-Datei: {CSV_PATH}")
print("-" * 60)

# --- 1. Daten laden und intelligente Datenbereinigung ---
try:
    df = pd.read_csv(CSV_PATH)
    print("Daten erfolgreich geladen.")
    print(f"Anzahl Zeilen: {df.shape[0]}, Anzahl Spalten: {df.shape[1]}")

    print("\nErste 5 Zeilen des Datensatzes:")
    print(df.head())

    print("\nDateninformationen (df.info()):")
    df.info()

    print("\nÜberprüfung und Behandlung fehlender Werte:")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print("    Keine fehlenden Werte im Datensatz gefunden. Daten sind sauber.")
    else:
        print("    Fehlende Werte gefunden. Intelligente Imputation wird durchgeführt:")
        for col, count in missing_values.items():
            if count > 0:
                print(f"      Spalte '{col}': {count} fehlende Werte.")
                if df[col].dtype in ['int64', 'float64']:
                    # Numerische Spalten mit dem Median imputieren (robuster gegenüber Ausreißern)
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"        Numerische Spalte '{col}' mit Median ({median_val:.2f}) imputiert.")
                elif df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                    # Kategoriale Spalten mit dem Modus imputieren
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
                    print(f"        Kategoriale Spalte '{col}' mit Modus ('{mode_val}') imputiert.")
                else:
                    print(f"        Warnung: Spalte '{col}' hat einen unbekannten Typ ({df[col].dtype}) und wurde nicht imputiert.")
        print("    Datenbereinigung für fehlende Werte abgeschlossen.")

    # Überprüfung und Sicherstellung der korrekten Datentypen für Features und Zielvariable
    print("\nSicherstellung der Datentypen für Features und Zielvariable:")
    for feature in FEATURES:
        if feature not in df.columns:
            raise ValueError(f"Fehler: Feature '{feature}' wurde im Datensatz nicht gefunden.")
        # Konvertierung zu numerischen Typen, falls nötig, mit Fehlerbehandlung
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
        if df[feature].isnull().any():
            raise ValueError(f"Fehler: Feature '{feature}' konnte nicht vollständig in einen numerischen Typ umgewandelt werden und enthält NaNs nach Coercion.")
        print(f"  '{feature}': Erfolgreich als {df[feature].dtype} bestätigt.")

    if TARGET_VARIABLE not in df.columns:
        raise ValueError(f"Fehler: Zielvariable '{TARGET_VARIABLE}' wurde im Datensatz nicht gefunden.")
    print(f"  '{TARGET_VARIABLE}': Erfolgreich als {df[TARGET_VARIABLE].dtype} bestätigt (kategorial erwartet).")


except FileNotFoundError:
    print(f"FEHLER: Die CSV-Datei wurde unter dem Pfad '{CSV_PATH}' nicht gefunden. Bitte überprüfen Sie den Pfad.")
    exit()
except Exception as e:
    print(f"Ein unerwarteter Fehler ist beim Laden oder der initialen Datenbereinigung aufgetreten: {e}")
    exit()

print("-" * 60)

# --- 2. Implementierung der vollständigen Preprocessing-Pipeline ---

# Trennung von Features (X) und Zielvariable (y)
X = df[FEATURES]
y = df[TARGET_VARIABLE]

print("\nSchritt: Kodierung der Zielvariable ('species')...")
# Initialisiere und trainiere den LabelEncoder für die Zielvariable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"    Zielvariable erfolgreich von {list(label_encoder.classes_)} in numerische Werte kodiert.")
print(f"    Beispiel Kodierung: '{y.iloc[0]}' wurde zu {y_encoded[0]}.")

# Speichern des LabelEncoders, um Vorhersagen später in die Original-Labels zurückwandeln zu können
try:
    joblib.dump(label_encoder, TARGET_ENCODER_FILENAME)
    print(f"    LabelEncoder erfolgreich gespeichert als '{TARGET_ENCODER_FILENAME}'.")
except Exception as e:
    print(f"    FEHLER beim Speichern des LabelEncoders: {e}")

# Definiere numerische Features für die Skalierung
numeric_features = FEATURES # Alle Features sind in diesem Projekt numerisch

# Erstelle den ColumnTransformer für das Preprocessing der Features
# StandardScaler wird auf alle numerischen Features angewendet
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='passthrough' # Behält andere (nicht definierte) Spalten unverändert bei (hier nicht relevant)
)
print(f"Preprocessing-Pipeline für numerische Features ({numeric_features}) mit StandardScaler konfiguriert.")
print("-" * 60)

# --- 3. Train-Test-Split ---
print("\nSchritt: Aufteilung des Datensatzes in Trainings- und Testsets (80/20-Verhältnis)...")
# Verwendung von stratify=y_encoded stellt sicher, dass die Klassenverteilung in beiden Sets erhalten bleibt
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)
print(f"    Trainingsset-Größe: {X_train.shape[0]} Beobachtungen")
print(f"    Testset-Größe: {X_test.shape[0]} Beobachtungen")
print("-" * 60)

# --- 4. Modellinitialisierung mit Hyperparametern ---
print("\nSchritt: Initialisierung des RandomForestClassifier-Modells...")
# Parsen der Hyperparameter aus dem JSON-String
hyperparameters = json.loads(HYPERPARAMETERS_JSON)

# Initialisiere den RandomForestClassifier mit den angegebenen Hyperparametern
model = RandomForestClassifier(**hyperparameters)
print(f"    RandomForestClassifier mit folgenden Hyperparametern initialisiert: {hyperparameters}")

# Erstellung der vollständigen Scikit-learn Pipeline: Preprocessing -> Modell
# Diese Pipeline kapselt alle Schritte und kann als eine Einheit trainiert und gespeichert werden
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), # Erster Schritt: Preprocessing der Features
    ('classifier', model)          # Zweiter Schritt: Der Klassifikator
])
print("    Vollständige ML-Pipeline (Preprocessing + Modell) erstellt.")
print("-" * 60)

# --- 5. Modelltraining ---
print("\nSchritt: Modelltraining starten...")
try:
    full_pipeline.fit(X_train, y_train_encoded)
    print("    Modelltraining erfolgreich abgeschlossen.")
except Exception as e:
    print(f"    FEHLER während des Modelltrainings: {e}")
    exit()
print("-" * 60)

# --- 6. Modellbewertung ---
print("\nSchritt: Bewertung des trainierten Modells auf dem Testset...")
# Vorhersagen auf dem Testset generieren
y_pred_encoded = full_pipeline.predict(X_test)

# Inverse Transformation der Testlabels und Vorhersagen, um sie für den Klassifikationsreport lesbar zu machen
y_test_original_labels = label_encoder.inverse_transform(y_test_encoded)
y_pred_original_labels = label_encoder.inverse_transform(y_pred_encoded)

# Berechnung relevanter Metriken
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
precision_macro = precision_score(y_test_encoded, y_pred_encoded, average='macro', zero_division=0)
recall_macro = recall_score(y_test_encoded, y_pred_encoded, average='macro', zero_division=0)
f1_macro = f1_score(y_test_encoded, y_pred_encoded, average='macro', zero_division=0)

print("\n--- Performance Metriken (Testset) ---")
# Ausgabe der Genauigkeit im geforderten Format
print(f"Accuracy: {accuracy:.4f}") 
print(f"Precision (Makro-Durchschnitt): {precision_macro:.4f}")
print(f"Recall (Makro-Durchschnitt): {recall_macro:.4f}")
print(f"F1-Score (Makro-Durchschnitt): {f1_macro:.4f}")

print("\n--- Detaillierter Klassifikationsreport ---")
# Der Klassifikationsreport zeigt Precision, Recall und F1-Score pro Klasse
print(classification_report(y_test_encoded, y_pred_encoded, target_names=label_encoder.classes_, zero_division=0))

print("\n--- Konfusionsmatrix ---")
# Die Konfusionsmatrix visualisiert die Leistung des Klassifikationsmodells
conf_matrix = confusion_matrix(y_test_encoded, y_pred_encoded)
print("Zeilen: Wahre Klassen, Spalten: Vorhergesagte Klassen")
print(pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_))
print("-" * 60)

# --- 7. Speichern des trainierten Modells und des LabelEncoders ---
print("\nSchritt: Speichern der Modelle und Encoder...")
try:
    joblib.dump(full_pipeline, MODEL_FILENAME)
    print(f"    Trainiertes Modell (inkl. Preprocessing-Pipeline) erfolgreich gespeichert als '{MODEL_FILENAME}'.")
except Exception as e:
    print(f"    FEHLER beim Speichern des Modells: {e}")

# Der LabelEncoder wurde bereits weiter oben gespeichert.

print("\n--- ML-Projekt erfolgreich abgeschlossen ---")