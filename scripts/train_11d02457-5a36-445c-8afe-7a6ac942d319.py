# ==============================================================================
# 🎯 Projekt-Einstellungen
# ==============================================================================
project_name = 'survey_lung_cancer-1762875524093-99 - ML Model'
file_path = r'/app/uploads/survey_lung_cancer-1762875524093-99.csv'
target_column = 'GENDER'
problem_type = ('Classification').lower()
model_type = 'RandomForestClassifier'
model_lib = 'sklearn'
model_params = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt'
}
model_save_path = r'/app/models/model_11d02457-5a36-445c-8afe-7a6ac942d319.pkl'

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
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error, 
    r2_score,
    accuracy_score
)

# Optional: PyTorch-Imports
if model_lib == 'pytorch':
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

def preprocess_target_variable(y, problem_type: str):
    """Bereinigt und konvertiert die Zielvariable für verschiedene Modelltypen."""
    from sklearn.preprocessing import LabelEncoder
    
    if problem_type == 'classification':
        # Prüfe ob y bereits numerisch ist
        if pd.api.types.is_numeric_dtype(y):
            print("Zielvariable ist bereits numerisch.")
            return y.values, None
        else:
            # Kategorische Labels zu numerischen Werten konvertieren
            print(f"Konvertiere kategorische Labels zu numerischen Werten...")
            print(f"Originale Labels: {y.unique()}")
            
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            print(f"Konvertierte Labels: {label_encoder.classes_}")
            print(f"Numerische Zuordnung: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
            
            return y_encoded, label_encoder
    else:  # regression
        # Für Regression: versuche zu konvertieren, falls nötig
        if not pd.api.types.is_numeric_dtype(y):
            try:
                y_numeric = pd.to_numeric(y, errors='coerce')
                if y_numeric.isna().any():
                    print("WARNUNG: Einige Werte konnten nicht zu Zahlen konvertiert werden.")
                return y_numeric.values, None
            except:
                print("FEHLER: Zielvariable für Regression muss numerisch sein.")
                return None, None
        return y.values, None

def load_and_split_data(file_path: str, target_column: str, problem_type: str = 'classification', generated_features: list = None):
    """Lädt Daten und teilt sie in Trainings- und Testsets auf (Schritt 1)."""
    # print(f"Lade Daten von: {file_path}")
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Fehler: Datei nicht gefunden unter {file_path}")
        return None, None, None, None, None

    # Datenvalidierung
    if target_column not in data.columns:
        print(f"FEHLER: Zielspalte '{target_column}' nicht in den Daten gefunden.")
        print(f"Verfügbare Spalten: {list(data.columns)}")
        return None, None, None, None, None
    
    # Feature Engineering: Generiere neue Features aus vorhandenen Spalten
    if generated_features and len(generated_features) > 0:
        print(f"Generiere {len(generated_features)} neue Feature(s)...")
        for feature_def in generated_features:
            try:
                feature_name = feature_def.get('name', '')
                feature_formula = feature_def.get('formula', '')
                if feature_name and feature_formula:
                    # Führe die Formel aus, um das neue Feature zu erstellen
                    # Die Formel sollte auf einem DataFrame namens 'data' arbeiten
                    exec(f"data['{feature_name}'] = {feature_formula}")
                    print(f"  ✓ Feature '{feature_name}' erstellt: {feature_def.get('description', '')}")
                except Exception as e:
                    print(f"  ✗ Fehler beim Erstellen von Feature '{feature_name}': {e}")
            except Exception as e:
                print(f"  ✗ Fehler beim Verarbeiten von Feature-Definition: {e}")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Feature-Validierung
    print(f"Feature-Spalten: {list(X.columns)}")
    print(f"Feature-Datentypen: {X.dtypes.to_dict()}")
    
    # Prüfe auf fehlende Werte
    missing_values = X.isnull().sum()
    if missing_values.any():
        print(f"WARNUNG: Fehlende Werte in Features gefunden:")
        print(missing_values[missing_values > 0])
        # Einfache Imputation für numerische Features
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())
        print("Fehlende Werte in numerischen Features durch Mittelwerte ersetzt.")
    
    # Zielvariable bereinigen und konvertieren
    y_processed, label_encoder = preprocess_target_variable(y, problem_type)
    
    if y_processed is None:
        return None, None, None, None, None
    
    return train_test_split(X, y_processed, test_size=0.2, random_state=42), label_encoder


def instantiate_and_train_model_dynamic(model_type_str: str, model_lib: str, params: dict, X_train, y_train, problem_type: str):
    """Instanziiert und trainiert das Modell dynamisch (Schritt 2)."""
    
    if model_lib == 'pytorch':
        return None 
        
    # print(f"Starte Training des Modells: {model_type_str} (Bibliothek: {model_lib})...")
    
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
            if problem_type == 'classification':
                ModelClass = XGBClassifier
            elif problem_type == 'regression':
                ModelClass = XGBRegressor
            else:
                raise ValueError(f"Ungültiger Problemtyp: {problem_type}")
                
        else:
            raise ValueError(f"Unbekannte Bibliothek: {model_lib}")

    except ImportError as e:
        raise ImportError(f"FEHLER beim Importieren von {model_type_str}: {e}. Haben Sie die Bibliothek installiert?")
        
    model = ModelClass(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, problem_type: str, label_encoder=None):
    """Bewertet das Modell und gibt Metriken aus (Schritt 4)."""
    
    # Vorhersagen
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    print(f'--- {model.__class__.__name__} Bewertung (Test-Set) ---')
    
    # Für Klassifikation: Labels zurückkonvertieren für bessere Lesbarkeit
    if problem_type == 'classification' and label_encoder is not None:
        print(f"Originale Klassen: {label_encoder.classes_}")
        print(f"Numerische Vorhersagen: {np.unique(y_pred_test)}")
    
    if problem_type == 'classification':
        # Klassifikationsmetriken berechnen
        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, average='macro')
        recall = recall_score(y_test, y_pred_test, average='macro')
        f1 = f1_score(y_test, y_pred_test, average='macro')

        # Klassifikationsmetriken ausgeben
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("--- Klassifikationsbericht (Test-Set) ---")
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
        # Regressionsmetriken berechnen
        accuracy = accuracy_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_test)

        # Regressionsmetriken ausgeben
        print(f"Accuracy: {accuracy:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")

def save_model(model, save_path: str, model_lib: str, label_encoder=None):
    """Speichert das trainierte Modell und optional den Label-Encoder (Schritt 5)."""
    # print(f"Speichere Modell unter: {save_path}")
    
    if model_lib in ['sklearn', 'xgboost']:
        joblib.dump(model, save_path)
        # print(f"Modell mit joblib gespeichert.")
        
        # Speichere Label-Encoder falls vorhanden
        if label_encoder is not None:
            encoder_path = save_path.replace('.pkl', '_encoder.pkl')
            joblib.dump(label_encoder, encoder_path)
            print(f"Encoder gespeichert unter: {encoder_path}")
            
    elif model_lib == 'pytorch':
        if hasattr(model, 'state_dict'):
             torch.save(model.state_dict(), save_path)
             print(f"Modell-state_dict mit torch.save gespeichert.")
             
             # Speichere Label-Encoder falls vorhanden
             if label_encoder is not None:
                 encoder_path = save_path.replace('.pth', '_encoder.pkl')
                 joblib.dump(label_encoder, encoder_path)
                 print(f"Encoder gespeichert unter: {encoder_path}")
        else:
             print("FEHLER: Das PyTorch-Modell hat keine 'state_dict'.")
    else:
        print("WARNUNG: Speichern für diese Bibliothek nicht implementiert.")
    
    # print("Speichern abgeschlossen.")


# ==============================================================================
# 🟢 HAUPT-TRAININGSPROZESS
# ==============================================================================

if __name__ == "__main__":
    print(f"--- Starte ML-Projekt: {project_name} (Typ: {problem_type.upper()}, Lib: {model_lib}) ---")
    
    # 1. DATEN LADEN UND AUFTEILEN
    generated_features = [
    {
        'description': 'Calculate index of peer pressure based on ANXIETY and PEER_PRESSURE features',
        'formula': 'import pandas as pd; df = pd.DataFrame({\'GENDER\': [0]*309, \'AGE\': [25]*309, \'SMOKING\': [1]*309, \'YELLOW_FINGERS\': [2]*309, \'ANXIETY\': [3]*309, \'PEER_PRESSURE\': [4]*309})
    index = df[\'PEER_PRESSURE\'] / (df[\'ANXIETY\'] + 1)
    return index',
        'name': 'PEER_PRESSURE_index',
        'reasoning': 'Peer pressure can impact gender based on anxiety levels'
    },
    {
        'description': 'Calculate index of fatigue based on FATIGUE and CHRONIC DISEASE features',
        'formula': 'import pandas as pd; df = pd.DataFrame({\'GENDER\': [0]*309, \'AGE\': [25]*309, \'SMOKING\': [1]*309, \'YELLOW_FINGERS\': [2]*309, \'ANXIETY\': [3]*309, \'PEER_PRESSURE\': [4]*309, \'FATIGUE\': [5]*309, \'CHRONIC DISEASE\': [6]*309})
    index = df[\'FATIGUE\'] / (df[\'CHRONIC DISEASE\'] + 1)
    return index',
        'name': 'FATIGUE_index',
        'reasoning': 'Fatigue can impact gender based on chronic disease levels'
    }
]
    result = load_and_split_data(file_path, target_column, problem_type, generated_features)
    
    if result is None or result[0] is None:
        print("FEHLER: Daten konnten nicht geladen werden.")
        exit()
    
    (X_train, X_test, y_train, y_test), label_encoder = result

    print(f"Trainingsdaten-Shape: {X_train.shape}")
    print(f"Testdaten-Shape: {X_test.shape}")
    
    # Zusätzliche Datenvalidierung
    print(f"Anzahl Features: {X_train.shape[1]}")
    print(f"Anzahl Trainingssamples: {X_train.shape[0]}")
    print(f"Anzahl Testsamples: {X_test.shape[0]}")
    
    if problem_type == 'classification':
        unique_classes = np.unique(y_train)
        print(f"Anzahl Klassen: {len(unique_classes)}")
        print(f"Klassen: {unique_classes}")
        
        if label_encoder is not None:
            print(f"Label-Encoder verfügbar für Rückkonvertierung")

    # ==========================================================================
    # 2. MODELL-TRAINING (SKLEARN/XGBOOST ODER PYTORCH SPEZIFISCHE LOGIK)
    # ==========================================================================

    if model_lib in ['sklearn', 'xgboost']:
        # Scikit-learn/XGBoost (Einfaches Training)
        try:
            model = instantiate_and_train_model_dynamic(model_type, model_lib, model_params, X_train, y_train, problem_type)
        except (ValueError, ImportError) as e:
            print(f"FEHLER: {e}")
            exit()
            
        # 3., 4. und 5. für sklearn/xgboost
        evaluate_model(model, X_train, y_train, X_test, y_test, problem_type, label_encoder)
        save_model(model, model_save_path, model_lib, label_encoder)
        
    elif model_lib == 'pytorch':
        # PyTorch (Komplexer, erfordert eigene Architektur und Loop)
        print("--- Starte PyTorch-Training ---")
        
        if 'torch' not in globals():
             print("FEHLER: PyTorch-Modul nicht importiert. Prüfen Sie die Installation.")
             exit()

        # Daten in Tensoren umwandeln und DataLoader erstellen
        X_train_tensor = torch.Tensor(X_train.values)
        y_train_tensor = torch.Tensor(y_train.values).long() if problem_type == 'classification' else torch.Tensor(y_train).unsqueeze(1)
        X_test_tensor = torch.Tensor(X_test.values)
        y_test_tensor = torch.Tensor(y_test.values).long() if problem_type == 'classification' else torch.Tensor(y_test).unsqueeze(1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        batch_size = model_params.get('batch_size', 16)
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
        output_size = 2 if problem_type == 'classification' else 1
        model = DynamicNet(input_size=input_size, output_size=output_size, hidden_layers=model_params.get('nn_hidden_layers', [64, 32]))
        
        # Hyperparameter
        criterion = nn.CrossEntropyLoss() if problem_type == 'classification' else nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=model_params.get('learning_rate', 0.001))
        num_epochs = model_params.get('num_epochs', 10)
        
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
            if problem_type == 'classification':
                _, predicted = torch.max(outputs.data, 1)
                y_pred_np = predicted.cpu().numpy()
                y_test_np = y_test_tensor.cpu().numpy().flatten()

                # Klassifikationsmetriken berechnen (manuell)
                accuracy = accuracy_score(y_test_np, y_pred_np)
                precision = precision_score(y_test_np, y_pred_np, average='macro')
                recall = recall_score(y_test_np, y_pred_np, average='macro')
                f1 = f1_score(y_test_np, y_pred_np, average='macro')

                print('--- Neural Network Klassifikation Bewertung ---')
                print(classification_report(y_test_np, y_pred_np))

                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")
                
                cm = confusion_matrix(y_test_np, y_pred_np)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Konfusionsmatrix - Neural Network')
                plt.xlabel('Vorhergesagte Klasse')
                plt.ylabel('Tatsächliche Klasse')
                plt.savefig(f'{project_name}_conf_matrix_nn.png')

            else: # Regression
                y_pred_np = outputs.cpu().numpy().flatten()
                y_test_np = y_test_tensor.cpu().numpy().flatten()
                
                mse = mean_squared_error(y_test_np, y_pred_np)
                r2 = r2_score(y_test_np, y_pred_np)
                
                print('--- Neural Network Regression Bewertung ---')
                print(f"MSE: {mse:.4f}")
                print(f"R2: {r2:.4f}")

                plt.figure(figsize=(8, 6))
                plt.scatter(y_test_np, y_pred_np, alpha=0.6)
                min_val = min(y_test_np.min(), y_pred_np.min())
                max_val = max(y_test_np.max(), y_pred_np.max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2) 
                plt.xlabel('Tatsächliche Werte')
                plt.ylabel('Vorhergesagte Werte')
                plt.title('Vorhersagen vs. Tatsächliche Werte - Neural Network')
                plt.grid(True)
                plt.savefig(f'{project_name}_pred_vs_act.png')

        # 5. Speichern für PyTorch
        save_model(model, model_save_path, model_lib, label_encoder)
        
    else:
        print("FEHLER: Unbekannter model_lib. Bitte auf 'sklearn', 'xgboost' oder 'pytorch' setzen.")

    print(f"--- ML-Projekt: {project_name} ABGESCHLOSSEN ---")