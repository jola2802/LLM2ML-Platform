import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

print("Starting ML Project: Iris Blumen Klassifikation")

# --- Configuration ---
CSV_PATH = r"C:\Users\jonas\Desktop\Text2ML\ML-Platform\LLM2ML-Platform\backend\uploads\1753810731508-413772562.csv"
MODEL_NAME = "Iris Blumen Klassifikation"
ALGORITHM_NAME = "RandomForestClassifier"
TARGET_VARIABLE = "species"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
HYPERPARAMETERS_STR = "{\"n_estimators\":100,\"max_depth\":10,\"random_state\":42}"
MODEL_FILE = "model.pkl"
ENCODER_FILE = "target_encoder.pkl"

print(f"Project Name: {MODEL_NAME}")
print(f"Algorithm: {ALGORITHM_NAME}")
print(f"Target Variable: {TARGET_VARIABLE}")
print(f"Features: {', '.join(FEATURES)}")
print(f"Hyperparameters: {HYPERPARAMETERS_STR}")

# Parse hyperparameters from string
HYPERPARAMETERS = json.loads(HYPERPARAMETERS_STR)

# --- 1. Load Data and Initial Data Cleaning ---
print(f"\nLoading data from: {CSV_PATH}")
try:
    df = pd.read_csv(CSV_PATH)
    print("Data loaded successfully.")
    print(f"Initial data shape: {df.shape}")
    print("First 5 rows of data:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: The file {CSV_PATH} was not found. Please ensure the path is correct.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# Check for missing values (intelligent data cleaning part)
print("\nChecking for missing values...")
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("No missing values found in the dataset. Data is clean.")
else:
    print("Missing values found per column:")
    print(missing_values[missing_values > 0])
    # Intelligent imputation for robustness will be handled by the pipeline.

# --- 2. Preprocessing Pipeline ---
print("\nSetting up preprocessing pipeline...")

# Separate features (X) and target (y)
X = df[FEATURES]
y = df[TARGET_VARIABLE]

# Identify numerical features for preprocessing (all features are numerical in this case)
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

# Preprocessor for numerical features (imputation with median for robustness)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')) # Median is robust to outliers
])

# Create a column transformer to apply transformations to specific columns.
# Only numerical features require imputation. No scaling needed for RandomForest.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features)
    ],
    remainder='passthrough' # Pass through any columns not explicitly specified (none in this case)
)

print("Preprocessing steps defined: Imputation for numerical features.")

# --- 3. Target Variable Encoding ---
print(f"\nEncoding target variable: '{TARGET_VARIABLE}'")
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print(f"Original target classes: {target_encoder.classes_}")
print(f"Encoded target values sample (first 5): {y_encoded[:5]}")

# Save the target encoder, which maps integer labels back to original class names
try:
    joblib.dump(target_encoder, ENCODER_FILE)
    print(f"Target encoder saved successfully as '{ENCODER_FILE}'.")
except Exception as e:
    print(f"Error saving target encoder: {e}")

# --- 4. Train-Test Split ---
print("\nPerforming Train-Test Split (80% training, 20% testing)...")
# Using stratify to ensure that class proportions are maintained in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("Train-Test Split complete.")

# --- 5. Model Training ---
print(f"\nInitializing {ALGORITHM_NAME} with hyperparameters: {HYPERPARAMETERS}")
model = RandomForestClassifier(**HYPERPARAMETERS)

# Create the full pipeline that includes both preprocessing and the classifier
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

print(f"Training {ALGORITHM_NAME} model...")
full_pipeline.fit(X_train, y_train)
print(f"{ALGORITHM_NAME} training complete.")

# --- 6. Model Evaluation ---
print(f"\nEvaluating {ALGORITHM_NAME} on the test set...")
y_pred = full_pipeline.predict(X_test)

# Decode predictions and true labels back to original string labels for clarity in reports
y_test_decoded = target_encoder.inverse_transform(y_test)
y_pred_decoded = target_encoder.inverse_transform(y_pred)

# Calculate Accuracy (required format)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Display detailed classification report
print("\nClassification Report:")
print(classification_report(y_test_decoded, y_pred_decoded, target_names=target_encoder.classes_))

# Display confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=target_encoder.classes_)
# Convert to DataFrame for better readability with class names
print(pd.DataFrame(cm, index=target_encoder.classes_, columns=target_encoder.classes_))

# --- 7. Save Trained Model ---
print(f"\nSaving trained model to '{MODEL_FILE}'...")
try:
    joblib.dump(full_pipeline, MODEL_FILE)
    print(f"Model saved successfully as '{MODEL_FILE}'.")
except Exception as e:
    print(f"Error saving model: {e}")

print("\nML Project: Iris Blumen Klassifikation - Script finished.")