export const predictionScript = `
# ==============================================================================
# 🎯 Projekt-Einstellungen
# ==============================================================================
input_features = INPUT_FEATURES
model_path = r'MODEL_PATH'
problem_type = ('PROBLEM_TYPE').lower()
project_id = 'PROJECT_ID'

import pandas as pd
import numpy as np
import joblib
import json

# Input-Features verarbeiten
input_data = input_features
# print(f"Input-Features: {input_data}")

# DataFrame erstellen
input_df = pd.DataFrame([input_data])
# print(f"Input DataFrame Shape: {input_df.shape}")

# Direkt numpy array verwenden
input_array = input_df.values
#print(f"Input Array Shape: {input_array.shape}")

# Model laden
try:
    model = joblib.load(model_path)
    #print("Model erfolgreich geladen")

    # Prüfe ob es eine Pipeline ist oder ein direktes Modell
    if hasattr(model, 'named_steps'):
        print("Pipeline-Modell erkannt")
        # Für Pipeline: Input muss durch die Pipeline gehen
        prediction = model.predict(input_array)[0]
    else:
        #print("Direktes Modell erkannt")
        # Für direktes Modell: Input muss skaliert werden falls nötig
        try:
            if problem_type == 'classification':
                target_encoder_path = '../models/' + 'model_'+ project_id + '_encoder.pkl'
                target_encoder = joblib.load(target_encoder_path)
                prediction = model.predict(input_array)[0]
                prediction = target_encoder.inverse_transform([int(prediction)])[0]
                #print(f"Decoded Prediction mit projekt-spezifischem Encoder: {prediction}")
            elif problem_type == 'regression':
                # Versuche Scaler zu laden
                scaler = joblib.load('../models/' + 'model_'+ project_id + '_scaler.pkl')
                input_scaled = scaler.transform(input_array)
                prediction = model.predict(input_scaled)[0]
        except Exception as scaler_error:
            print(f"Scaler/Encoder nicht gefunden oder Fehler: {scaler_error}")
            # Fallback: Direkte Prediction ohne Skalierung
            # Aber zuerst Feature-Namen entfernen falls vorhanden
            if hasattr(input_array, 'columns'):
                input_array = input_array.values
            prediction = model.predict(input_array)[0]
        except Exception as encoder_error:
            print(f"Encoder nicht gefunden oder Fehler: {encoder_error}")
            pass

    # Ergebnis ausgeben (wird vom Node.js-Server geparst)
    print(f"PREDICTION_RESULT: {prediction}")

except Exception as e:
    print(f"Prediction error: {e}")
    exit(1)
`.trim();