# Generiere einen vollständigen Python-Code für Machine Learning Training:

# **Code-Muster:** Muss dem bereitgestellten Muster folgen, einschließlich der Schritte Laden/Splitten, Instanziieren/Trainieren, Vorhersagen, Bewerten und Speichern.

# **Projektname:** XGBRegressor_Demo

# **Daten laden und aufteilen (Schritt 1):**
# * **Laden:** Funktion 'load_and_split_data' verwenden. **ACHTUNG:** Der Parameter `problem_type` muss auf `'regression'` gesetzt werden.
# * **Dateipfad:** 'dummy_regression_data.csv'
# * **Features:** Alle Features

# **Hyperparameter (Schritt 2):**
# * **Algorithmus:** XGBRegressor
# * **Hyperparameter:** {
#   "objective": "reg:squarederror",
#   "n_estimators": 100,
#   "random_state": 42
# }
# * **Zielspalte:** target

# **Vorhersagen (Schritt 3):**
# * **Bibliothek:** 'predict' verwenden.

# **Performance-Metriken (Schritt 4):**
# * **Test-Metriken:** Implementiere **alle Standardmetriken** für den definierten **Problemtyp** (Regression).
#     * **Falls Klassifikation:** 'classification_report' und 'confusion_matrix' (Visualisiert mit 'seaborn').
#     * **Falls Regression:** **'mean_squared_error'** ('MSE') und **'r2_score'**.
#     * **Zusätzlich:** Füge die Berechnung des **R² Scores** für das **Trainings-Set** (`model_xgb_reg.score(X_train, y_train)`) hinzu und visualisiere die Vorhersagen gegen die tatsächlichen Werte (Streudiagramm).

# **Speichern (Schritt 5):**
# * **Bibliothek:** 'joblib.dump' verwenden.
# * **Dateiname:** 'xgb_regressor_demo_model.pkl'.

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