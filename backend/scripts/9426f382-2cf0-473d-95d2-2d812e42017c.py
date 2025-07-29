import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os

print("--------------------------------------------------")
print("ML Project: Iris Flower Dataset - Classification Model")
print("--------------------------------------------------")

# --- 1. Configuration ---
CSV_PATH = r"C:\Users\jonas\Desktop\Text2ML\ML-Platform\backend\uploads\1753801923271-795880374.csv"
MODEL_SAVE_PATH = "model.pkl"
TARGET_ENCODER_SAVE_PATH = "target_encoder.pkl"

TARGET_VARIABLE = "species"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Algorithm and Hyperparameters
ALGORITHM_NAME = "RandomForestClassifier"
HYPERPARAMETERS_JSON = "{\"n_estimators\":100,\"max_depth\":5,\"random_state\":42}"
HYPERPARAMETERS = json.loads(HYPERPARAMETERS_JSON) # Parse JSON string to dictionary

print(f"CSV Path: {CSV_PATH}")
print(f"Target Variable: {TARGET_VARIABLE}")
print(f"Features: {FEATURES}")
print(f"Algorithm: {ALGORITHM_NAME} with Hyperparameters: {HYPERPARAMETERS}")
print("--------------------------------------------------")

# --- 2. Load Data ---
print("Loading data...")
try:
    df = pd.read_csv(CSV_PATH)
    print("Data loaded successfully.")
    print(f"Initial dataset shape: {df.shape}")
    print("First 5 rows of data:")
    print(df.head())
    print("\nColumn information:")
    df.info()
except FileNotFoundError:
    print(f"Error: The file '{CSV_PATH}' was not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)
print("--------------------------------------------------")

# --- 3. Intelligent Data Cleaning ---
print("Starting data cleaning and preprocessing...")

# Check for missing values
print("\nChecking for missing values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
if missing_values.sum() == 0:
    print("No missing values found in the dataset.")
else:
    print("Missing values detected. Imputation will be handled by the pipeline.")

# Check for duplicate rows
print("\nChecking for duplicate rows:")
num_duplicates = df.duplicated().sum()
if num_duplicates > 0:
    print(f"Found {num_duplicates} duplicate rows. Removing them...")
    df.drop_duplicates(inplace=True)
    print(f"Dataset shape after removing duplicates: {df.shape}")
else:
    print("No duplicate rows found.")
print("--------------------------------------------------")

# --- 4. Define Features (X) and Target (y) ---
print("Defining features (X) and target (y)...")
X = df[FEATURES]
y = df[TARGET_VARIABLE]
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("--------------------------------------------------")

# --- 5. Target Variable Encoding ---
# For classification, encode the target variable. LabelEncoder is suitable for the target.
print(f"Encoding target variable '{TARGET_VARIABLE}'...")
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print(f"Original unique classes: {target_encoder.classes_}")
print(f"Encoded classes (first 5): {y_encoded[:5]}")
print("--------------------------------------------------")

# --- 6. Train-Test Split ---
print("Performing Train-Test Split (80% train, 20% test)...")
# Using random_state for reproducibility and stratify for balanced classes in split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=HYPERPARAMETERS['random_state'], stratify=y_encoded
)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print("--------------------------------------------------")

# --- 7. Preprocessing Pipeline for Features ---
print("Setting up preprocessing pipeline for features...")

# Define numerical and categorical features
# In this specific project, all features are numerical based on project details.
numeric_features = FEATURES
# categorical_features = [] # No categorical features in X for this project

# Create preprocessing steps for numerical features
# Impute missing numerical values with the mean, then scale.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Handles potential missing values
    ('scaler', StandardScaler()) # Scales numerical features
])

# Create a preprocessor using ColumnTransformer
# It applies different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
        # Add other transformers here if categorical_features were present:
        # ('cat', categorical_transformer, categorical_features)
    ])
print("Preprocessing pipeline for features configured.")
print("--------------------------------------------------")

# --- 8. Model Training ---
print(f"Initializing and training {ALGORITHM_NAME} model...")

# Get the model class dynamically
if ALGORITHM_NAME == "RandomForestClassifier":
    ML_MODEL = RandomForestClassifier(**HYPERPARAMETERS)
else:
    print(f"Error: Algorithm '{ALGORITHM_NAME}' not recognized or mapped.")
    sys.exit(1)

# Create the full pipeline: preprocessor + model
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', ML_MODEL)])

# Train the model
model_pipeline.fit(X_train, y_train)
print(f"Model training complete using {ALGORITHM_NAME}.")
print("--------------------------------------------------")

# --- 9. Model Evaluation ---
print("Evaluating model performance...")

# Make predictions on the test set
y_pred_encoded = model_pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_encoded)
print(f"Accuracy: {accuracy:.4f}") # Output format for parsing

# Generate classification report
# To make the report more readable, inverse transform y_test and y_pred if needed,
# but for metric calculation, encoded values are fine.
# For display, map back to original labels.
y_test_labels = target_encoder.inverse_transform(y_test)
y_pred_labels = target_encoder.inverse_transform(y_pred_encoded)

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels))

print("\nConfusion Matrix:")
# Get unique sorted labels for display, ensuring all possible labels are covered
labels = target_encoder.classes_
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))
print("--------------------------------------------------")

# --- 10. Save Model and Target Encoder ---
print("Saving trained model and target encoder...")

try:
    joblib.dump(model_pipeline, MODEL_SAVE_PATH)
    print(f"Model saved successfully to '{MODEL_SAVE_PATH}'")
except Exception as e:
    print(f"Error saving model: {e}")

try:
    joblib.dump(target_encoder, TARGET_ENCODER_SAVE_PATH)
    print(f"Target encoder saved successfully to '{TARGET_ENCODER_SAVE_PATH}'")
except Exception as e:
    print(f"Error saving target encoder: {e}")

print("--------------------------------------------------")
print("ML Project execution complete.")
print("--------------------------------------------------")