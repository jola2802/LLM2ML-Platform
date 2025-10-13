# ==============================================================================
# 🎯 KONFIGURATIONSVARIABLEN (ANPASSEN)
# ==============================================================================
# Projekt-Einstellungen
PROJECT_NAME = "My_ML_Project_Demo"
FILE_PATH = "../../data/my_data.csv" # Pfad zur Quelldatei
TARGET_COLUMN = "target"             # Name der Zielspalte (Target)
PROBLEM_TYPE = "classification"      # 'classification' oder 'regression'

# Modell-Einstellungen 
# WÄHLEN SIE HIER IHR MODELL:
# * SKLEARN: 'LogisticRegression', 'RandomForestClassifier', 'SVC', 'GradientBoostingClassifier', etc.
# * XGBOOST: 'XGBClassifier', 'XGBRegressor'
# * PYTORCH: 'NeuralNetwork'
MODEL_TYPE = "NeuralNetwork" # Wichtig: Der Name des Modells
MODEL_LIB = "pytorch"                # 'sklearn', 'xgboost' oder 'pytorch'
MODEL_PARAMS = {
    'num_epochs': 10, 
    'batch_size': 16, 
    'learning_rate': 0.001,
    'random_state': 42 # Für XGBoost/Sklearn relevant
    'nn_hidden_layers': [64, 32]
}

# Pfade speichern
MODEL_SAVE_PATH = f'../../models/{PROJECT_NAME.lower()}_model.pkl'

# ==============================================================================
# 🐍 BASIS IMPORTE
# ==============================================================================
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    mean_squared_error, 
    r2_score,
    accuracy_score
)

# Optional: PyTorch-Imports für den Demo-Fall
if MODEL_LIB == 'pytorch':
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("WARNUNG: PyTorch ist nicht installiert. PyTorch-Modelle können nicht ausgeführt werden.")

# ==============================================================================
# 🛠️ FUNKTIONEN
# ==============================================================================

def load_and_split_data(file_path: str, target_column: str, problem_type: str = 'classification'):
    """Lädt Daten und teilt sie in Trainings- und Testsets auf (Schritt 1)."""
    print(f"Lade Daten von: {file_path}")
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Fehler: Datei nicht gefunden unter {file_path}")
        return None, None, None, None

    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    if problem_type == 'regression':
        y = y.values 
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


def instantiate_and_train_model_dynamic(model_type_str: str, model_lib: str, params: dict, X_train, y_train):
    """Instanziiert und trainiert das Modell dynamisch (Schritt 2)."""
    
    if model_lib == 'pytorch':
        return None 
        
    print(f"Starte Training des Modells: {model_type_str} (Bibliothek: {model_lib})...")
    
    try:
        if model_lib == 'sklearn':
            # Dynamische Imports aus gängigen Scikit-learn Modulen
            from sklearn.ensemble import __dict__ as ensemble_models
            from sklearn.linear_model import __dict__ as linear_models
            from sklearn.svm import __dict__ as svm_models
            
            all_sklearn_models = {**ensemble_models, **linear_models, **svm_models}

            if model_type_str in all_sklearn_models:
                ModelClass = all_sklearn_models[model_type_str]
            else:
                raise ImportError(f"Modellklasse '{model_type_str}' nicht in gängigen sklearn-Modulen gefunden.")
                    
        elif model_lib == 'xgboost':
            from xgboost import XGBClassifier, XGBRegressor
            if model_type_str == 'XGBClassifier':
                ModelClass = XGBClassifier
            elif model_type_str == 'XGBRegressor':
                ModelClass = XGBRegressor
            else:
                raise ImportError(f"Ungültiger XGBoost-Typ: {model_type_str}")
                
        else:
            raise ValueError(f"Unbekannte Bibliothek: {model_lib}")

    except ImportError as e:
        raise ImportError(f"FEHLER beim Importieren von {model_type_str}: {e}. Haben Sie die Bibliothek installiert?")
        
    model = ModelClass(**params)
    model.fit(X_train, y_train)
    print("Training abgeschlossen.")
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, problem_type: str):
    """Bewertet das Modell und gibt Metriken aus (Schritt 4)."""
    
    # Vorhersagen
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    print(f'\n--- {model.__class__.__name__} Bewertung (Test-Set) ---')
    
    if problem_type == 'classification':
        # Klassifikationsmetriken
        print(f"Genauigkeit (Training): {accuracy_score(y_train, y_pred_train):.4f}")
        print("\n--- Klassifikationsbericht (Test-Set) ---")
        print(classification_report(y_test, y_pred_test))
        
        # Konfusionsmatrix visualisieren
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Konfusionsmatrix - {model.__class__.__name__}')
        plt.xlabel('Vorhergesagte Klasse')
        plt.ylabel('Tatsächliche Klasse')
        plt.savefig('confusion_matrix.png')
        
    elif problem_type == 'regression':
        # Regressionsmetriken
        mse = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        # Genauigkeit auf dem Trainingsset für Regression (z.B. R2 Score)
        if hasattr(model, 'score'):
            r2_train = model.score(X_train, y_train)
            print(f"R-squared (R²) Training: {r2_train:.4f}")
        
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R-squared (R²) Test: {r2:.4f}")
        
        # Visualisierung (Vorhersagen vs. Tatsächliche Werte)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.6)
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2) 
        plt.xlabel('Tatsächliche Werte')
        plt.ylabel('Vorhergesagte Werte')
        plt.title(f'Vorhersagen vs. Tatsächliche Werte - {model.__class__.__name__}')
        plt.grid(True)
        plt.savefig('predictions_vs_actual.png')

def save_model(model, save_path: str, model_lib: str):
    """Speichert das trainierte Modell (Schritt 5)."""
    print(f"\nSpeichere Modell unter: {save_path}")
    
    if model_lib in ['sklearn', 'xgboost']:
        joblib.dump(model, save_path)
        print(f"Modell mit joblib gespeichert.")
    elif model_lib == 'pytorch':
        if hasattr(model, 'state_dict'):
             torch.save(model.state_dict(), save_path)
             print(f"Modell-state_dict mit torch.save gespeichert.")
        else:
             print("FEHLER: Das PyTorch-Modell hat keine 'state_dict'.")
    else:
        print("WARNUNG: Speichern für diese Bibliothek nicht implementiert.")
    
    print("Speichern abgeschlossen.")


# ==============================================================================
# 🟢 HAUPT-TRAININGSPROZESS
# ==============================================================================

if __name__ == "__main__":
    print(f"--- Starte ML-Projekt: {PROJECT_NAME} (Typ: {PROBLEM_TYPE.upper()}, Lib: {MODEL_LIB}) ---")
    
    # 1. DATEN LADEN UND AUFTEILEN
    X_train, X_test, y_train, y_test = load_and_split_data(FILE_PATH, TARGET_COLUMN, PROBLEM_TYPE)

    if X_train is None:
        exit()

    print(f"Trainingsdaten-Shape: {X_train.shape}")
    print(f"Testdaten-Shape: {X_test.shape}")

    # ==========================================================================
    # 2. MODELL-TRAINING (SKLEARN/XGBOOST ODER PYTORCH SPEZIFISCHE LOGIK)
    # ==========================================================================

    if model_lib in ['sklearn', 'xgboost']:
        # Scikit-learn/XGBoost (Einfaches Training)
        try:
            model = instantiate_and_train_model_dynamic(model_type, model_lib, model_params, X_train, y_train)
        except (ValueError, ImportError) as e:
            print(f"FEHLER: {e}")
            exit()
            
        # 3., 4. und 5. für sklearn/xgboost
        evaluate_model(model, X_train, y_train, X_test, y_test, problem_type)
        save_model(model, MODEL_SAVE_PATH, model_lib)
        
    elif model_lib == 'pytorch':
        # PyTorch (Komplexer, erfordert eigene Architektur und Loop)
        print("\n--- Starte PyTorch-Training ---")
        
        if 'torch' not in globals():
             print("FEHLER: PyTorch-Modul nicht importiert. Prüfen Sie die Installation.")
             exit()

        # Daten in Tensoren umwandeln und DataLoader erstellen
        X_train_tensor = torch.Tensor(X_train.values)
        y_train_tensor = torch.Tensor(y_train.values).long() if problem_type == 'classification' else torch.Tensor(y_train).unsqueeze(1)
        X_test_tensor = torch.Tensor(X_test.values)
        y_test_tensor = torch.Tensor(y_test.values).long() if problem_type == 'classification' else torch.Tensor(y_test).unsqueeze(1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        batch_size = MODEL_PARAMS.get('batch_size', 16)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        
        # 🚨 DYNAMISCHE DEFINITION DES NEURALEN NETZES BASIEREND AUF NN_HIDDEN_LAYERS
        class DynamicNet(nn.Module):
            def __init__(self, input_size, output_size, hidden_layers):
                super(DynamicNet, self).__init__()
                layers = []
                prev_size = input_size
                
                for hidden_size in hidden_layers:
                    layers.append(nn.Linear(prev_size, hidden_size))
                    layers.append(nn.ReLU())
                    prev_size = hidden_size
                
                # Output layer
                layers.append(nn.Linear(prev_size, output_size))
                
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)
                
        input_size = X_train.shape[1]
        output_size = 2 if PROBLEM_TYPE == 'classification' else 1
        model = DynamicNet(input_size=input_size, output_size=output_size, hidden_layers=model_params.get('nn_hidden_layers', [64, 32]))
        
        # Hyperparameter
        criterion = nn.CrossEntropyLoss() if problem_type == 'classification' else nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=model_params.get('learning_rate', 0.001))
        num_epochs = MODEL_PARAMS.get('num_epochs', 10)
        
        # Training-Loop
        for epoch in range(num_epochs):
            for i, (features, labels) in enumerate(train_loader):
                outputs = model(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # 3. PyTorch Vorhersage und 4. Bewertung
        with torch.no_grad():
            model.eval()
            outputs = model(X_test_tensor)
            
            # --- Bewertung: Logik an Problemtyp anpassen ---
            if PROBLEM_TYPE == 'classification':
                _, predicted = torch.max(outputs.data, 1)
                y_pred_np = predicted.numpy()
                y_test_np = y_test_tensor.numpy().flatten()
                
                print('\n--- Neural Network Klassifikation Bewertung ---')
                print(classification_report(y_test_np, y_pred_np))
                
                cm = confusion_matrix(y_test_np, y_pred_np)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Konfusionsmatrix - Neural Network')
                plt.xlabel('Vorhergesagte Klasse')
                plt.ylabel('Tatsächliche Klasse')
                plt.savefig(f'{PROJECT_NAME}_conf_matrix_nn.png')

            else: # Regression
                y_pred_np = outputs.numpy().flatten()
                y_test_np = y_test_tensor.numpy().flatten()
                
                mse = mean_squared_error(y_test_np, y_pred_np)
                r2 = r2_score(y_test_np, y_pred_np)
                
                print('\n--- Neural Network Regression Bewertung ---')
                print(f"Mean Squared Error (MSE): {mse:.4f}")
                print(f"R-squared (R²): {r2:.4f}")

                plt.figure(figsize=(8, 6))
                plt.scatter(y_test_np, y_pred_np, alpha=0.6)
                min_val = min(y_test_np.min(), y_pred_np.min())
                max_val = max(y_test_np.max(), y_pred_np.max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2) 
                plt.xlabel('Tatsächliche Werte')
                plt.ylabel('Vorhergesagte Werte')
                plt.title('Vorhersagen vs. Tatsächliche Werte - Neural Network')
                plt.grid(True)
                plt.savefig(f'{PROJECT_NAME}_pred_vs_act.png')

        # 5. Speichern für PyTorch
        save_model(model, MODEL_SAVE_PATH, MODEL_LIB)
        
    else:
        print("FEHLER: Unbekannter MODEL_LIB. Bitte auf 'sklearn', 'xgboost' oder 'pytorch' setzen.")

    print(f"\n--- ML-Projekt: {PROJECT_NAME} ABGESCHLOSSEN ---")