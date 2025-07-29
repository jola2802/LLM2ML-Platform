import pandas as pd
import numpy as np
import joblib
import json

# Model laden
try:
    model = joblib.load(r"C:\Users\jonas\Desktop\Text2ML\ML-Platform\LLM2ML-Platform\backend\models\model_851c3090-e7e7-420e-9b10-fa66669236e4.pkl")
    print("Model erfolgreich geladen")
except Exception as e:
    print(f"Fehler beim Laden des Models: {e}")
    exit(1)

# Input-Features verarbeiten
input_data = {"sepal_length":4,"sepal_width":4,"petal_length":4,"petal_width":1}
print(f"Input-Features: {input_data}")

# DataFrame erstellen
input_df = pd.DataFrame([input_data])
print(f"Input DataFrame Shape: {input_df.shape}")
print(f"Input DataFrame Columns: {list(input_df.columns)}")

# Prediction
try:
    prediction = model.predict(input_df)[0]
    print(f"Raw Prediction: {prediction}")
    
    # Falls Label-Encoder existiert (für Klassifikation)
    try:
        target_encoder = joblib.load('target_encoder.pkl')
        prediction = target_encoder.inverse_transform([int(prediction)])[0]
        print(f"Decoded Prediction: {prediction}")
    except:
        pass
    
    # Ergebnis ausgeben (wird vom Node.js-Server geparst)
    print(f"PREDICTION_RESULT: {prediction}")
    
except Exception as e:
    print(f"Prediction error: {e}")
    exit(1)