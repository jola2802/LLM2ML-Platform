# Demo-Code für XGBClassifier

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from xgboost import XGBClassifier

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
X_train, X_test, y_train, y_test = load_and_split_data('C:/Users/jonas/Desktop/Text2ML/ML-Platform/LLM2ML-Platform/backend/demo-machine-learning/dummy_classification_data.csv', 'target')

# 2. Modell instanziieren und trainieren
model_xgb_clf = XGBClassifier(objective='binary:logistic', n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
model_xgb_clf.fit(X_train, y_train)

# 3. Vorhersagen treffen
y_pred_xgb_clf = model_xgb_clf.predict(X_test)

# 4. Modell bewerten
print("--- XGBClassifier Bewertung ---")
print(classification_report(y_test, y_pred_xgb_clf))

# Konfusionsmatrix visualisieren
cm = confusion_matrix(y_test, y_pred_xgb_clf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Konfusionsmatrix - XGBClassifier')
plt.xlabel('Vorhergesagte Klasse')
plt.ylabel('Tatsächliche Klasse')
plt.show()

# 5. Modell speichern
joblib.dump(model_xgb_clf, 'C:/Users/jonas/Desktop/Text2ML/ML-Platform/LLM2ML-Platform/backend/demo-machine-learning/Classification/models/xgb_classifier.pkl')