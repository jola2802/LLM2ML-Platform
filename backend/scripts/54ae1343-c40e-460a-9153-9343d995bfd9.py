import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Starting ML Project: Studenten_Social_Media_Verhalten_und_Wohlbefinden - Regression Model")

# Define file path as specified
CSV_PATH = r"C:\Users\jonas\Desktop\Text2ML\ML-Platform\backend\uploads\1753785666695-273906462.csv"
MODEL_FILENAME = 'model.pkl'
# Note: For regression tasks with numerical targets, a separate LabelEncoder for the target
# is typically not needed, as the target is already numerical.
# Thus, no 'target_encoder.pkl' is generated.

# Project details as provided
TARGET_VARIABLE = 'Addicted_Score'
FEATURES = ['Age', 'Gender', 'Academic_Level', 'Country', 'Avg_Daily_Usage_Hours',
            'Most_Used_Platform', 'Affects_Academic_Performance', 'Sleep_Hours_Per_Night',
            'Mental_Health_Score', 'Relationship_Status', 'Conflicts_Over_Social_Media']
HYPERPARAMETERS_JSON = "{\"n_estimators\":150,\"max_depth\":10,\"min_samples_split\":5,\"min_samples_leaf\":3,\"random_state\":42}"
ALGORITHM_NAME = "RandomForestRegressor"

print(f"Loading data from: {CSV_PATH}")
try:
    df = pd.read_csv(CSV_PATH)
    print("Data loaded successfully.")
    print(f"Initial dataset shape: {df.shape}")
    print("Columns in dataset:", df.columns.tolist())
except FileNotFoundError:
    print(f"Error: The file '{CSV_PATH}' was not found. Please ensure the path is correct.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

print("\n--- Data Cleaning and Preprocessing ---")

# 1. Drop Student_ID column as it's an identifier and not a feature
if 'Student_ID' in df.columns:
    df = df.drop('Student_ID', axis=1)
    print("Dropped 'Student_ID' column.")

# Intelligent Data Cleaning: Handling missing values
print("Checking for missing values...")
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("No missing values found in the dataset.")
else:
    print("Missing values found (before imputation steps in pipeline):")
    print(missing_values[missing_values > 0])
    # While SimpleImputer is used in the pipeline, pre-imputation for
    # initial data analysis or specific transformations can be done here.
    # For robustness, the pipeline itself will handle any remaining or new NaNs.

# Check for duplicates
if df.duplicated().any():
    print(f"Found {df.duplicated().sum()} duplicate rows. Dropping duplicates.")
    df.drop_duplicates(inplace=True)
    print(f"Dataset shape after dropping duplicates: {df.shape}")
else:
    print("No duplicate rows found.")

# Feature Transformation/Mapping for 'Affects_Academic_Performance'
# This is a binary categorical feature, mapping 'Yes' to 1 and 'No' to 0 directly
# makes it numerical and can improve model interpretation and performance.
if 'Affects_Academic_Performance' in df.columns:
    df['Affects_Academic_Performance'] = df['Affects_Academic_Performance'].map({'Yes': 1, 'No': 0})
    # If any other value appears, it will become NaN, which the imputer in pipeline will handle.
    # For this specific dataset, we assume only 'Yes'/'No' exist.
    print("Mapped 'Affects_Academic_Performance' to numerical (Yes: 1, No: 0).")

# Define features (X) and target (y)
X = df[FEATURES]
y = df[TARGET_VARIABLE]

print(f"\nTarget variable: '{TARGET_VARIABLE}'")
print(f"Features used for modeling: {FEATURES}")

# Identify numerical and categorical features for the preprocessor pipeline
# After mapping, 'Affects_Academic_Performance' is numerical.
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"\nNumerical features for scaling: {numerical_features}")
print(f"Categorical features for one-hot encoding: {categorical_features}")

# Create preprocessing pipelines for numerical and categorical features
# Imputation is included in the pipeline for robustness against potential NaNs in test data or future data.
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Imputes missing numerical values with the median
    ('scaler', StandardScaler()) # Scales numerical features to have zero mean and unit variance
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Imputes missing categorical values with the mode
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # Converts categorical features into one-hot encoded vectors.
                                                       # 'handle_unknown='ignore'' prevents errors if unseen categories appear in test set.
])

# Create a preprocessor using ColumnTransformer
# This allows applying different transformations to different columns.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # 'passthrough' keeps columns not specified (e.g., if we added features later)
)

# 4. Perform Train-Test Split
print("\n--- Performing Train-Test Split ---")
# Using random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Parse hyperparameters from JSON string
hyperparameters = json.loads(HYPERPARAMETERS_JSON)
print(f"\nUsing Algorithm: {ALGORITHM_NAME}")
print(f"Hyperparameters: {hyperparameters}")

# 5. Initialize and train the model
# Initialize RandomForestRegressor with the parsed hyperparameters
model = RandomForestRegressor(**hyperparameters)

# Create the full pipeline: first preprocess, then train the regressor
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', model)])

print("\n--- Training the model ---")
full_pipeline.fit(X_train, y_train)
print("Model training completed.")

# 6. Evaluate the model
print("\n--- Evaluating the model ---")
y_pred = full_pipeline.predict(X_test)

# Calculate relevant regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) # Root Mean Squared Error
r2 = r2_score(y_test, y_pred) # R-squared (Coefficient of Determination)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
# Output R2 Score in the requested format, adapted for regression
print(f"R2 Score: {r2:.4f}")

# 7. Save the trained model
print(f"\n--- Saving the trained model to '{MODEL_FILENAME}' ---")
try:
    joblib.dump(full_pipeline, MODEL_FILENAME)
    print(f"Model successfully saved as '{MODEL_FILENAME}'.")
except Exception as e:
    print(f"Error saving the model: {e}")

print("\nML Project execution finished successfully.")