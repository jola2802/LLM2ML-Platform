# Demo-Code für NeuralNetworkClassifier

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
X_train, X_test, y_train, y_test = load_and_split_data('dummy_classification_data.csv', 'target')
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

model_nn_clf = NeuralNet(input_size=X_train.shape[1], num_classes=2)
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

    # Konfusionsmatrix visualisieren
    cm = confusion_matrix(y_test_np, y_pred_np)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Konfusionsmatrix - NeuralNetworkClassifier')
    plt.xlabel('Vorhergesagte Klasse')
    plt.ylabel('Tatsächliche Klasse')
    plt.show()

# 4. Modell speichern
torch.save(model_nn_clf.state_dict(), 'neural_net_classifier.pth')