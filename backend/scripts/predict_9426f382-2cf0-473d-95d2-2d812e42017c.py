import pandas as pd
import numpy as np
import joblib
import json

# Model laden
try:
    model = joblib.load(r"C:\Users\jonas\Desktop\Text2ML\ML-Platform\backend\models\model_9426f382-2cf0-473d-95d2-2d812e42017c.pkl")
    print("Model erfolgreich geladen")
except Exception as e:
    print(f"Fehler beim Laden des Models: {e}")
    exit(1)

# Input-Features verarbeiten
input_data = {"sepal_length":5.7,"sepal_width":5,"petal_length":5,"petal_width":3}
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