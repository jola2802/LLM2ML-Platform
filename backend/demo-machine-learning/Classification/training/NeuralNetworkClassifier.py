# Generiere einen vollständigen Python-Code für Machine Learning Training:

# **Code-Muster:** Muss dem bereitgestellten Muster folgen, einschließlich der Schritte Laden/Splitten, Instanziieren/Trainieren, Vorhersagen, Bewerten und Speichern.

# **Projektname:** PyTorch_NeuralNet_Demo

# **Daten laden und aufteilen (Schritt 1):**
# * **Laden:** Funktion 'load_and_split_data' verwenden. **ACHTUNG:** Die geladenen Daten müssen sofort in PyTorch-Tensoren umgewandelt und ein DataLoader erstellt werden.
# * **Dateipfad:** '../../dummy_classification_data.csv'
# * **Features:** Alle Features

# **Hyperparameter (Schritt 2):**
# * **Algorithmus:** PyTorch Neural Network Classifier
# * **Hyperparameter:** {
#   "input_size": "dynamisch (X_train.shape[1])",
#   "num_classes": 2,
#   "num_epochs": 10,
#   "batch_size": 16,
#   "learning_rate": 0.001
# }
# * **Zielspalte:** target

# **Vorhersagen (Schritt 3):**
# * **Bibliothek:** **PyTorch** Logik (z.B. `torch.max(outputs.data, 1)`) verwenden, Ergebnisse in NumPy konvertieren.

# **Performance-Metriken (Schritt 4):**
# * **Test-Metriken:** Implementiere **alle Standardmetriken** für den definierten **Problemtyp** (BinaryClassification).
#     * **Falls Klassifikation:** **'classification_report'** und **'confusion_matrix'** (Visualisiert mit 'seaborn') mit den NumPy-konvertierten Ergebnissen.
#     * **Falls Regression:** 'mean_squared_error' ('MSE') und 'r2_score'.
#     * **Zusätzlich:** Gib den **Loss** während des Trainings aus.

# **Speichern (Schritt 5):**
# * **Bibliothek:** **'torch.save'** verwenden (Speichern des `state_dict`).
# * **Dateiname:** '../models/neural_net_classifier.pth'.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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

# 1. Daten laden und in Tensoren umwandeln
X_train, X_test, y_train, y_test = load_and_split_data('../../dummy_classification_data.csv', 'target')
X_train_tensor = torch.Tensor(X_train.values)
y_train_tensor = torch.Tensor(y_train.values).long()
X_test_tensor = torch.Tensor(X_test.values)
y_test_tensor = torch.Tensor(y_test.values).long()

# Datenlader erstellen
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# 2. Modell-Architektur definieren und trainieren
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model_nn _clf = NeuralNet(input_size=X_train.shape[1], num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_nn_clf.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for features, labels in train_loader:
        outputs = model_nn_clf(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 3. Vorhersagen treffen und bewerten
with torch.no_grad():
    outputs = model_nn_clf(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    
    print("--- NeuralNetworkClassifier Bewertung ---")
    y_pred_np = predicted.numpy()
    y_test_np = y_test_tensor.numpy()
    print(classification_report(y_test_np, y_pred_np))

    # # Konfusionsmatrix visualisieren
    # cm = confusion_matrix(y_test_np, y_pred_np)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.title('Konfusionsmatrix - NeuralNetworkClassifier')
    # plt.xlabel('Vorhergesagte Klasse')
    # plt.ylabel('Tatsächliche Klasse')
    # plt.show()

# 4. Modell speichern
torch.save(model_nn_clf.state_dict(), '../models/neural_net_classifier.pth')