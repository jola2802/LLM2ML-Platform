import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("**************************************************")
print("ML Projekt: Iris Flower Dataset - Classification Model")
print("**************************************************\n")

# --- 1. Projekt-Details und Konfiguration ---
# Definition des absoluten Pfades zur CSV-Datei
CSV_PATH = r"C:\Users\jonas\Desktop\Text2ML\ML-Platform\backend\uploads\1753783950135-240843176.csv"
TARGET_VARIABLE = 'species'
FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
ALGORITHM_NAME = 'RandomForestClassifier'
HYPERPARAMETERS_STR = "{\"n_estimators\":100,\"max_depth\":5,\"random_state\":42}"

print(f"Konfiguration:")
print(f"  CSV Pfad: {CSV_PATH}")
print(f"  Zielvariable: '{TARGET_VARIABLE}'")
print(f"  Merkmale: {', '.join(FEATURES)}")
print(f"  Verwendeter Algorithmus: {ALGORITHM_NAME}")
print(f"  Hyperparameter (JSON): {HYPERPARAMETERS_STR}")

# Parsen der Hyperparameter aus dem JSON-String in ein Python-Dictionary
try:
    hyperparameters = json.loads(HYPERPARAMETERS_STR)
    print(f"  Hyperparameter (Python Dict): {hyperparameters}")
except json.JSONDecodeError as e:
    print(f"FEHLER: Ungültiges Hyperparameter-JSON-Format: {e}")
    exit()
print("-" * 50)

# --- 2. Daten laden ---
print("Schritt 2: Lade Daten aus CSV-Datei...")
try:
    df = pd.read_csv(CSV_PATH)
    print("Daten erfolgreich geladen.")
    print(f"  Anzahl der geladenen Zeilen: {len(df)}")
    print(f"  Spalten im Datensatz: {', '.join(df.columns)}")
    print("\n  Erste 5 Zeilen der Daten:")
    print(df.head())
    print("-" * 50)
except FileNotFoundError:
    print(f"FEHLER: Die angegebene CSV-Datei wurde nicht unter '{CSV_PATH}' gefunden.")
    exit()
except Exception as e:
    print(f"FEHLER beim Laden der Daten: {e}")
    exit()

# --- 3. Intelligente Datenbereinigung und Vorbereitung ---
print("Schritt 3: Datenbereinigung und Vorverarbeitung...")

# Überprüfung auf fehlende Werte und intelligente Behandlung
print("  Überprüfe auf fehlende Werte...")
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("  Keine fehlenden Werte im Datensatz gefunden. Daten sind sauber.")
else:
    print("  Fehlende Werte gefunden. Beginne mit Imputation:")
    for column, count in missing_values.items():
        if count > 0:
            if df[column].dtype in ['int64', 'float64']:
                # Numerische Spalten: Imputation mit dem Mittelwert
                mean_val = df[column].mean()
                df[column].fillna(mean_val, inplace=True)
                print(f"    Fehlende Werte in numerischer Spalte '{column}' mit Mittelwert ({mean_val:.2f}) gefüllt.")
            elif df[column].dtype == 'object':
                # Kategoriale Spalten: Imputation mit dem Modus (häufigster Wert)
                mode_val = df[column].mode()[0]
                df[column].fillna(mode_val, inplace=True)
                print(f"    Fehlende Werte in kategorialer Spalte '{column}' mit Modus ('{mode_val}') gefüllt.")
    print("  Datenbereinigung auf fehlende Werte abgeschlossen.")

# Trennung von Features (X) und Zielvariable (y)
X = df[FEATURES]
y = df[TARGET_VARIABLE]
print(f"  X (Features) Shape: {X.shape}")
print(f"  y (Zielvariable) Shape: {y.shape}")

# Kodierung der Zielvariable ('species')
print(f"  Kodierung der Zielvariable '{TARGET_VARIABLE}'...")
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print(f"  Ursprüngliche Klassen: {target_encoder.classes_}")
print(f"  Kodierte Werte (erste 5): {y_encoded[:5]} (entsprechend {y.head().tolist()})")
print("  Zielvariable erfolgreich kodiert.")
print("-" * 50)

# --- 4. Train-Test-Split ---
print("Schritt 4: Teile Daten in Trainings- und Testset auf (80% Training, 20% Test)...")
# Verwendung von stratify=y_encoded, um sicherzustellen, dass die Klassenverteilung
# in Trainings- und Testset gleich ist (wichtig bei unbalancierten Datensätzen, aber gute Praxis hier).
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"  Trainingsset-Größe (X_train): {X_train.shape}")
print(f"  Testset-Größe (X_test): {X_test.shape}")
print("-" * 50)

# --- 5. Implementierung der Preprocessing-Pipeline und Modelltraining ---
print("Schritt 5: Erstelle Preprocessing Pipeline und trainiere das Modell...")

# Preprocessing für numerische Features (Skalierung)
# Da alle Features numerisch sind und skaliert werden sollen, verwenden wir StandardScaler.
# ColumnTransformer ermöglicht die Anwendung verschiedener Transformationen auf unterschiedliche Spalten.
# Hier wenden wir StandardScaler auf alle definierten Features an.
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, FEATURES)
    ])

# Modellinstanziierung basierend auf dem angegebenen Algorithmus und Hyperparametern
if ALGORITHM_NAME == 'RandomForestClassifier':
    model = RandomForestClassifier(**hyperparameters)
else:
    print(f"FEHLER: Der Algorithmus '{ALGORITHM_NAME}' ist nicht im Script implementiert oder falsch geschrieben.")
    exit()

# Erstellen der vollständigen Machine Learning Pipeline
# Die Pipeline führt zuerst die Preprocessing-Schritte durch und trainiert dann das Modell.
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', model)])

print(f"  Beginne mit dem Training des {ALGORITHM_NAME} Modells...")
full_pipeline.fit(X_train, y_train)
print("  Modelltraining abgeschlossen.")
print("-" * 50)

# --- 6. Modell-Evaluierung ---
print("Schritt 6: Evaluiere das trainierte Modell auf dem Testset...")
y_pred = full_pipeline.predict(X_test)

# Berechne Performance-Metriken
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=target_encoder.classes_)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n--- Performance Metriken ---")
# WICHTIG: Ausgabe der Genauigkeit (Accuracy) im spezifischen Format
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report (Präzision, Recall, F1-Score pro Klasse):")
print(class_report)
print("\nConfusion Matrix (Tatsächliche vs. Vorhergesagte Klassen):")
print(conf_matrix)
print("-" * 50)

# --- 7. Modell und Label Encoder speichern ---
print("Schritt 7: Speichere das trainierte Modell und den Label Encoder...")

MODEL_FILE = 'model.pkl'
ENCODER_FILE = 'target_encoder.pkl'

try:
    joblib.dump(full_pipeline, MODEL_FILE)
    print(f"  Modell erfolgreich unter '{MODEL_FILE}' gespeichert.")
except Exception as e:
    print(f"FEHLER beim Speichern des Modells: {e}")

try:
    joblib.dump(target_encoder, ENCODER_FILE)
    print(f"  Label Encoder erfolgreich unter '{ENCODER_FILE}' gespeichert.")
except Exception as e:
    print(f"FEHLER beim Speichern des Label Encoders: {e}")