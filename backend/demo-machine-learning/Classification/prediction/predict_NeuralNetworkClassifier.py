import torch
import torch.nn as nn
import numpy as np

# 1. Modell-Architektur definieren (muss genau die gleiche sein wie beim Training)
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

# 2. Modell laden (State Dict)
try:
    # Annahme: Das Modell wurde mit 10 Merkmalen und 2 Klassen trainiert
    num_features = 10 
    num_classes = 2
    model = NeuralNet(input_size=num_features, num_classes=num_classes)
    model.load_state_dict(torch.load('../models/neural_net_classifier.pth'))
    model.eval() # Modell in den Evaluierungsmodus setzen
    print("NeuralNetworkClassifier-Modell erfolgreich geladen.")
except FileNotFoundError:
    print("Fehler: Modell nicht gefunden. Bitte zuerst das Trainingsskript ausführen.")
    exit()

# 3. Nutzereingabe verarbeiten
print("\nBitte geben Sie die Werte für die Merkmale ein, getrennt durch Leerzeichen:")
user_input = input(f"({num_features} Werte erwartet): ")
try:
    input_values = np.array([float(x) for x in user_input.split()]).reshape(1, -1)
    if input_values.shape[1] != num_features:
        raise ValueError
    input_tensor = torch.Tensor(input_values)
except (ValueError, IndexError):
    print(f"Ungültige Eingabe. Es werden genau {num_features} numerische Werte erwartet.")
    exit()

# 4. Vorhersage treffen
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs.data, 1)

# 5. Ergebnis anzeigen
print("\n--- Vorhersage-Ergebnis ---")
print(f"Die vorhergesagte Klasse ist: {predicted.item()}")