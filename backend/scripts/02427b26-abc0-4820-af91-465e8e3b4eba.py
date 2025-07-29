import pandas as pd
import joblib
import numpy as np
import json
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer # Used for robust handling of potential missing values
import xgboost as xgb

print("--- ML Project Script Start ---")

# --- Configuration ---
# Important: Use the exact CSV path as specified in the requirements.
CSV_PATH = r"C:\Users\jonas\Desktop\Text2ML\ML-Platform\backend\uploads\1753785251265-282454645.csv"
TARGET_VARIABLE = 'species'
FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
HYPERPARAMETERS_STR = "{\"n_estimators\":100,\"learning_rate\":0.1,\"max_depth\":3,\"objective\":\"multi:softmax\",\"num_class\":3}"
MODEL_SAVE_PATH = 'model.pkl'
ENCODER_SAVE_PATH = 'target_encoder.pkl'
TEST_SIZE = 0.2  # 80% train, 20% test split
RANDOM_STATE = 42 # For reproducibility of train-test split and model training

print(f"Project Name: Iris Flower Dataset - Classification Model")
print(f"Algorithm: XGBoostClassifier")
print(f"Target Variable: {TARGET_VARIABLE}")
print(f"Features: {FEATURES}")
print(f"CSV Path: {CSV_PATH}")

# --- 1. Load Data ---
print("\n--- 1. Loading Data ---")
try:
    df = pd.read_csv(CSV_PATH)
    print(f"Successfully loaded data from {CSV_PATH}")
    print(f"Dataset shape: {df.shape} (rows, columns)")
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nColumn information:")
    df.info()
except FileNotFoundError:
    print(f"Error: CSV file not found at '{CSV_PATH}'. Please ensure the path is correct.")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- 2. Intelligent Data Cleaning ---
# Based on the detailed LLM analysis, the Iris dataset is "extremely clean"
# with "no visible missing values" and "no outliers".
# However, for a robust script, we include general checks for missing values
# and type consistency, even if they result in no changes for this specific dataset.
print("\n--- 2. Intelligent Data Cleaning & Data Type Verification ---")

# Check for expected columns
required_columns = FEATURES + [TARGET_VARIABLE]
if not all(col in df.columns for col in required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    print(f"Error: Missing one or more required columns in the CSV file: {missing_cols}. Please check the dataset columns.")
    exit()

# Handle missing values and ensure correct data types for features
# Use SimpleImputer for numerical features (e.g., median strategy)
numeric_imputer = SimpleImputer(strategy='median')

for col in FEATURES:
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"Warning: Feature '{col}' is not purely numeric. Attempting to coerce to numeric...")
        df[col] = pd.to_numeric(df[col], errors='coerce') # Convert non-numeric to NaN
        if df[col].isnull().any():
            print(f"Found non-numeric values converted to NaN in '{col}'. Imputing with median.")
            df[col] = numeric_imputer.fit_transform(df[[col]]) # Impute NaNs after coercion
            print(f"Imputed missing/coerced values in '{col}'.")
    elif df[col].isnull().any():
        # If numeric but still has NaNs (e.g., from original data)
        missing_count = df[col].isnull().sum()
        print(f"Found {missing_count} missing values in numerical feature '{col}'. Imputing with median.")
        df[col] = numeric_imputer.fit_transform(df[[col]])
        print(f"Imputed missing values in '{col}'.")

# Handle missing values in target variable by dropping rows
if df[TARGET_VARIABLE].isnull().any():
    initial_rows = df.shape[0]
    df.dropna(subset=[TARGET_VARIABLE], inplace=True)
    rows_dropped = initial_rows - df.shape[0]
    print(f"Warning: Target variable '{TARGET_VARIABLE}' had {rows_dropped} missing values. Dropped affected rows.")
    if df.empty:
        print("Error: Dataset became empty after dropping rows with missing target. Exiting.")
        exit()

print("Data cleaning steps completed.")
print(f"Dataset shape after cleaning: {df.shape}")

# --- 3. Prepare Data for ML Pipeline ---
print("\n--- 3. Preparing Data for ML Pipeline ---")
X = df[FEATURES]
y = df[TARGET_VARIABLE]

# Encode target variable: XGBoostClassifier with 'multi:softmax' expects integer labels (0, 1, 2...).
# LabelEncoder is suitable for this.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Target variable '{TARGET_VARIABLE}' encoded.")
print(f"Original labels: {label_encoder.classes_}")
print(f"Encoded labels sample (first 5): {y_encoded[:5]}")

# Save the LabelEncoder for future use (e.g., inverse transforming predictions)
try:
    with open(ENCODER_SAVE_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"LabelEncoder successfully saved to '{ENCODER_SAVE_PATH}'")
except Exception as e:
    print(f"Error saving LabelEncoder: {e}")

# --- 4. Train-Test Split ---
print("\n--- 4. Performing Train-Test Split ---")
# Stratify ensures that the proportion of each class is the same in both training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded)
print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")

# --- 5. Implement Preprocessing Pipeline and Model Training ---
print("\n--- 5. Building and Training ML Pipeline ---")

# Define preprocessing steps for numerical features
# StandardScaler is used for scaling numerical features.
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Create a preprocessor using ColumnTransformer to apply transformations to specific columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, FEATURES) # Apply numerical_transformer to all features
    ])

# Parse hyperparameters from the string format
hyperparameters = json.loads(HYPERPARAMETERS_STR)
print(f"Using XGBoostClassifier with hyperparameters: {hyperparameters}")

# Define the XGBoostClassifier model
# random_state is passed for reproducibility of the model's internal randomness
model = xgb.XGBClassifier(**hyperparameters, random_state=RANDOM_STATE)

# Create the full pipeline: Preprocessing followed by the classifier
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

print("ML Pipeline created. Training model...")
full_pipeline.fit(X_train, y_train)
print("Model training completed.")

# --- 6. Model Evaluation ---
print("\n--- 6. Evaluating Model Performance ---")
y_pred = full_pipeline.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
# Required format for accuracy output
print(f"Accuracy: {accuracy:.4f}")

# Generate Classification Report for detailed metrics
print("\nClassification Report:")
# To make the classification report readable, use the original class names
target_names_original = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))
print(classification_report(y_test, y_pred, target_names=target_names_original))

# Generate Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# --- 7. Save Model ---
print("\n--- 7. Saving Trained Model ---")
try:
    with open(MODEL_SAVE_PATH, 'wb') as file:
        pickle.dump(full_pipeline, file)
    print(f"Trained model successfully saved to '{MODEL_SAVE_PATH}'")
except Exception as e:
    print(f"Error saving model: {e}")

print("\n--- ML Project Script End ---")