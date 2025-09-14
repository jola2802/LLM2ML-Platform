# Demo-Code für NeuralNetworkRegressor

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
X_train, X_test, y_train, y_test = load_and_split_data('dummy_regression_data.csv', 'target', problem_type='regression')
X_train_tensor_reg = torch.Tensor(X_train.values)
y_train_tensor_reg = torch.Tensor(y_train).unsqueeze(1)
X_test_tensor_reg = torch.Tensor(X_test.values)
y_test_tensor_reg = torch.Tensor(y_test).unsqueeze(1)

# Datenlader erstellen
train_dataset_reg = TensorDataset(X_train_tensor_reg, y_train_tensor_reg)
train_loader_reg = DataLoader(dataset=train_dataset_reg, batch_size=16, shuffle=True)

# 2. Modell-Architektur definieren und trainieren
class NeuralNetRegressor(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model_nn_reg = NeuralNetRegressor(input_size=X_train.shape[1])
criterion_reg = nn.MSELoss()
optimizer_reg = optim.Adam(model_nn_reg.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for features, labels in train_loader_reg:
        outputs = model_nn_reg(features)
        loss = criterion_reg(outputs, labels)
        optimizer_reg.zero_grad()
        loss.backward()
        optimizer_reg.step()

# 3. Vorhersagen treffen und bewerten
with torch.no_grad():
    y_pred_nn_reg = model_nn_reg(X_test_tensor_reg)
    
    print("--- NeuralNetworkRegressor Bewertung ---")
    y_pred_np = y_pred_nn_reg.numpy()
    y_test_np = y_test_tensor_reg.numpy()
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test_np, y_pred_np):.4f}")
    print(f"R-squared (R²): {r2_score(y_test_np, y_pred_np):.4f}")
    
    # Vorhersagen vs. tatsächliche Werte visualisieren
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_np, y_pred_np, alpha=0.6)
    plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--', lw=2)
    plt.xlabel('Tatsächliche Werte')
    plt.ylabel('Vorhergesagte Werte')
    plt.title('Vorhersagen vs. Tatsächliche Werte - NeuralNetworkRegressor')
    plt.grid(True)
    plt.show()

# 4. Modell speichern
torch.save(model_nn_reg.state_dict(), 'neural_net_regressor.pth')