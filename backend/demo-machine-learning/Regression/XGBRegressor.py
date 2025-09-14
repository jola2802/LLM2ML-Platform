# Demo-Code für XGBRegressor

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from xgboost import XGBRegressor

# Funktion zum Laden und Aufteilen der Daten
def load_and_split_data(file_path, target_column, problem_type='classification'):
    """Lädt Daten und teilt sie in Trainings- und Testsets auf."""
    data = pd.read_csv(file_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Für Regression muss y in ein Numpy-Array konvertiert werden
    if problem_type == 'regression':
        y = y.values
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Daten laden und aufteilen
X_train, X_test, y_train, y_test = load_and_split_data('dummy_regression_data.csv', 'target', problem_type='regression')

# 2. Modell instanziieren und trainieren
model_xgb_reg = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model_xgb_reg.fit(X_train, y_train)

# 3. Vorhersagen treffen
y_pred_xgb_reg = model_xgb_reg.predict(X_test)

# 4. Modell bewerten
print("--- XGBRegressor Bewertung ---")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_xgb_reg):.4f}")
print(f"R-squared (R²): {r2_score(y_test, y_pred_xgb_reg):.4f}")

# Vorhersagen vs. tatsächliche Werte visualisieren
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_xgb_reg, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Tatsächliche Werte')
plt.ylabel('Vorhergesagte Werte')
plt.title('Vorhersagen vs. Tatsächliche Werte - XGBRegressor')
plt.grid(True)
plt.show()

# 5. Modell speichern
joblib.dump(model_xgb_reg, 'xgb_regressor.pkl')