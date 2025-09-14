import joblib
import numpy as np

# 1. Modell laden
try:
    model = joblib.load('C:/Users/jonas/Desktop/Text2ML/ML-Platform/LLM2ML-Platform/backend/demo-machine-learning/Classification/models/xgb_classifier.pkl')
    print("XGBClassifier-Modell erfolgreich geladen.")
except FileNotFoundError:
    print("Fehler: Modell nicht gefunden. Bitte zuerst das Trainingsskript ausführen.")
    exit()

# Modell-Informationen
num_features = model.n_features_in_
print(f"Das Modell wurde mit {num_features} Merkmalen trainiert.")

# 2. Nutzereingabe verarbeiten
print("\nBitte geben Sie die Werte für die Merkmale ein, getrennt durch Leerzeichen:")
user_input = input(f"({num_features} Werte erwartet): ")
try:
    input_values = np.array([float(x) for x in user_input.split()]).reshape(1, -1)
    if input_values.shape[1] != num_features:
        raise ValueError
except (ValueError, IndexError):
    print(f"Ungültige Eingabe. Es werden genau {num_features} numerische Werte erwartet.")
    exit()

# 3. Vorhersage treffen
prediction = model.predict(input_values)

# 4. Ergebnis anzeigen
print("\n--- Vorhersage-Ergebnis ---")
print(f"Die vorhergesagte Klasse ist: {prediction[0]}")