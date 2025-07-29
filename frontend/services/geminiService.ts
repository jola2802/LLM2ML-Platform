
import { GoogleGenAI, Type, GenerateContentResponse } from "@google/genai";
import { ModelType, Project } from "../types";

const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  console.warn("API_KEY environment variable not set. Gemini API calls will fail.");
}

const ai = new GoogleGenAI({ apiKey: API_KEY! });

export const analyzeDataForMLStrategy = async (csvAnalysis: any, targetVariable: string, features: string[]) => {
  try {
    const prompt = `Du bist ein erfahrener Machine Learning und Data Science Experte. Analysiere die folgenden Daten und empfehle die beste ML-Strategie.

DATEN-ANALYSE:
- Anzahl Zeilen: ${csvAnalysis.rowCount}
- Spalten: ${csvAnalysis.columns.length}
- Target Variable: "${targetVariable}"
- Features: ${features.join(', ')}

DATENTYPEN:
${Object.entries(csvAnalysis.dataTypes).map(([col, type]) => `- ${col}: ${type}`).join('\n')}

BEISPIEL-DATEN (erste 3 Zeilen):
${csvAnalysis.sampleData.slice(0, 3).map((row: any[], i: number) => 
  `Zeile ${i+1}: ${csvAnalysis.columns.map((col: string, j: number) => `${col}=${row[j]}`).join(', ')}`
).join('\n')}

AUFGABEN:
1. Bestimme ob es sich um Classification oder Regression handelt
2. Empfehle den BESTEN Algorithmus aus: RandomForestClassifier, LogisticRegression, SVM, XGBoostClassifier (für Classification) oder RandomForestRegressor, LinearRegression, SVR, XGBoostRegressor (für Regression)
3. Empfehle spezifische Hyperparameter für den gewählten Algorithmus
4. Begründe deine Entscheidungen ausführlich

Antworte in folgendem JSON-Format:
{
  "modelType": "Classification" oder "Regression",
  "recommendedAlgorithm": "Algorithmus-Name",
  "hyperparameters": {
    "parameter1": value1,
    "parameter2": value2
  },
  "reasoning": "Ausführliche Begründung warum dieser Algorithmus und diese Parameter optimal sind",
  "confidence": 0.0-1.0,
  "alternativeAlgorithms": ["Algorithmus2", "Algorithmus3"]
}`;

    const responseSchema = {
      type: Type.OBJECT,
      properties: {
        modelType: {
          type: Type.STRING,
          enum: ["Classification", "Regression"],
          description: "The recommended model type"
        },
        recommendedAlgorithm: {
          type: Type.STRING,
          description: "The recommended algorithm name"
        },
        hyperparameters: {
          type: Type.OBJECT,
          description: "Recommended hyperparameters for the algorithm"
        },
        reasoning: {
          type: Type.STRING,
          description: "Detailed reasoning for the recommendations"
        },
        confidence: {
          type: Type.NUMBER,
          description: "Confidence score between 0 and 1"
        },
        alternativeAlgorithms: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
          description: "Alternative algorithm suggestions"
        }
      },
      required: ["modelType", "recommendedAlgorithm", "hyperparameters", "reasoning", "confidence"]
    };

    const response: GenerateContentResponse = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: responseSchema,
      },
    });

    const jsonText = response.text?.trim() || '';
    return JSON.parse(jsonText);

  } catch (error) {
    console.error("Error analyzing data with Gemini:", error);
    throw new Error("Failed to get ML strategy analysis. Please check your API key and network connection.");
  }
};

export const analyzeDataColumns = async (columns: string[]) => {
  try {
    const prompt = `Given the following column names from a dataset, act as an expert data scientist. Your task is to analyze them and suggest a machine learning project configuration. The column names are: ${columns.join(', ')}.

Return a JSON object with the following structure:
- "suggestedModelType": either "Classification" or "Regression".
- "suggestedTargetVariable": the column name you recommend as the prediction target.
- "suggestedFeatures": an array of column names you recommend as input features. Exclude the target variable from this list.
- "reasoning": A brief explanation for your choices.`;

    const responseSchema = {
      type: Type.OBJECT,
      properties: {
        suggestedModelType: {
          type: Type.STRING,
          enum: ["Classification", "Regression"],
          description: "The suggested machine learning model type."
        },
        suggestedTargetVariable: {
          type: Type.STRING,
          description: "The column name recommended as the prediction target."
        },
        suggestedFeatures: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
          description: "An array of column names recommended as input features."
        },
        reasoning: {
          type: Type.STRING,
          description: "A brief explanation for the recommendations."
        }
      },
      required: ["suggestedModelType", "suggestedTargetVariable", "suggestedFeatures", "reasoning"]
    };

    const response: GenerateContentResponse = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: responseSchema,
      },
    });

    const jsonText = response.text?.trim() || '';
    return JSON.parse(jsonText);

  } catch (error) {
    console.error("Error analyzing data with Gemini:", error);
    throw new Error("Failed to get analysis from AI. Please check your API key and network connection.");
  }
};

const executeTrainingScriptAndGetResults = async (script: string, modelType: ModelType) => {
    const isClassification = modelType === ModelType.Classification;
    
    const metricsProperties = isClassification 
      ? {
          accuracy: { type: Type.NUMBER, description: "Accuracy score from the script" },
          precision: { type: Type.NUMBER, description: "Precision score from the script" },
          recall: { type: Type.NUMBER, description: "Recall score from the script" },
          f1: { type: Type.NUMBER, description: "F1 score from the script" }
        }
      : {
          mae: { type: Type.NUMBER, description: "Mean Absolute Error from the script" },
          mse: { type: Type.NUMBER, description: "Mean Squared Error from the script" },
          r2: { type: Type.NUMBER, description: "R-squared score from the script" }
        };

    const prompt = `
    You are a Python execution engine. Given the following scikit-learn training script, simulate its execution and analyze its output.

    Script:
    \`\`\`python
    ${script}
    \`\`\`

    Your tasks:
    1.  Pretend to run this script.
    2.  Extract the final evaluation metrics that the script would print.
    3.  Generate a unique, mock base64 encoded string to represent the saved model artifact (e.g., the .pkl file). This string should be a plausible-looking but fake artifact.

    Return a single JSON object with the following structure:
    - "metrics": An object containing the extracted performance metrics (${Object.keys(metricsProperties).join(', ')}).
    - "modelArtifact": The generated mock base64 string for the model artifact.
    `;

    const responseSchema = {
        type: Type.OBJECT,
        properties: {
            metrics: {
                type: Type.OBJECT,
                properties: metricsProperties,
                required: Object.keys(metricsProperties)
            },
            modelArtifact: {
                type: Type.STRING,
                description: "A mock base64 encoded string representing the trained model artifact."
            }
        },
        required: ["metrics", "modelArtifact"]
    };

    const response: GenerateContentResponse = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: prompt,
        config: {
            responseMimeType: "application/json",
            responseSchema: responseSchema,
        },
    });

    return JSON.parse(response.text?.trim() || '');
};

export const trainModel = async (project: Omit<Project, 'id' | 'status' | 'createdAt'>) => {
    try {
        // Step 1: Generate the Python training script
        const pythonCode = await generatePythonScript(project);

        // Step 2: "Execute" the script to get metrics and a model artifact
        const { metrics, modelArtifact } = await executeTrainingScriptAndGetResults(pythonCode, project.modelType);
        
        return { metrics, pythonCode, modelArtifact };

    } catch (error) {
        console.error("Error in the training process:", error);
        throw new Error("Failed to train the model.");
    }
};

export const makePrediction = async (features: { [key: string]: string | number }, project: Project) => {
    try {
        const featureString = Object.entries(features).map(([key, value]) => `${key}: ${value}`).join(', ');
        const prompt = `You are an ML model inference engine.
        A model has been trained and is represented by the following artifact: "${project.modelArtifact}".
        This model was trained to solve a ${project.modelType} task, predicting the target variable "${project.targetVariable}".
        The model was trained with features like: ${project.features.join(', ')}.

        Now, predict the outcome for the following input data:
        ${featureString}

        Return ONLY the predicted value, without any explanation or extra text.
        For a Classification task, predict a plausible category (e.g., "Yes", "No", "ClassA").
        For a Regression task, predict a plausible number (e.g., 157000, 23.5).
        Your response must be grounded in the context of the provided model artifact and task.
        `;

        const response: GenerateContentResponse = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
            config: {
                thinkingConfig: { thinkingBudget: 0 } 
            }
        });
        
        return response.text?.trim() || '';

    } catch (error) {
        console.error("Error making prediction with Gemini:", error);
        throw new Error("Prediction failed. The AI service may be unavailable.");
    }
};

export const generatePythonScript = async (project: { name: string, modelType: ModelType, targetVariable: string, features: string[] }) => {
    try {
        const { name, modelType, targetVariable, features } = project;
        const isClassification = modelType === ModelType.Classification;
        const modelImport = isClassification 
            ? 'from sklearn.ensemble import RandomForestClassifier' 
            : 'from sklearn.ensemble import RandomForestRegressor';
        const modelInstance = isClassification 
            ? 'RandomForestClassifier(n_estimators=100, random_state=42)' 
            : 'RandomForestRegressor(n_estimators=100, random_state=42)';
        const evaluationMetricsImport = isClassification
            ? 'from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score'
            : 'from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score';
        const evaluationMetricsCode = isClassification
            ? `
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
`
            : `
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared: {r2_score(y_test, y_pred)}")
`;

        const prompt = `
        Generate a complete, executable Python script using scikit-learn for a machine learning project.

        Project Details:
        - Project Name: "${name}"
        - Model Type: ${modelType}
        - Target Variable: "${targetVariable}"
        - Features: ${JSON.stringify(features)}

        The script must include the following sections, clearly marked with comments:
        1.  Import necessary libraries (pandas, scikit-learn, joblib).
        2.  Load a dummy dataset using pandas. The DataFrame should be named 'df' and have columns: ${JSON.stringify([...features, targetVariable])}. Generate 100 rows of realistic-looking sample data.
        3.  Define features (X) and target (y).
        4.  Split the data into training and testing sets (80/20 split, random_state=42).
        5.  Initialize and train the model (${isClassification ? 'RandomForestClassifier' : 'RandomForestRegressor'}).
        6.  Make predictions on the test set.
        7.  Evaluate the model by printing the relevant metrics.
        8.  Save the trained model to a file named 'model.pkl' using joblib.
        9.  Include comments explaining each major step.

        The final script MUST be fully self-contained and executable.
        Return only the Python code. Do not add any text or markdown formatting before or after the code block.
        `;
        
        const response: GenerateContentResponse = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
        });

        let script = response.text?.trim() || '';
        if (script.startsWith('```python')) {
            script = script.substring(9).trim();
        }
        if (script.endsWith('```')) {
            script = script.substring(0, script.length - 3).trim();
        }
        
        const fullScript = `
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
${modelImport}
${evaluationMetricsImport}
import joblib

# 1. Generate Dummy Data
# In a real scenario, you would load your data here, e.g., df = pd.read_csv('your_data.csv')
num_samples = 100
data = {}
for feature in ${JSON.stringify(features)}:
    data[feature] = np.random.rand(num_samples) * 100
if ${isClassification}:
    data['${targetVariable}'] = np.random.randint(0, 2, num_samples)
else:
    data['${targetVariable}'] = np.random.rand(num_samples) * 50000 + 10000
df = pd.DataFrame(data)

print("--- Data Head ---")
print(df.head())
print("\\n")

# 2. Define Features (X) and Target (y)
X = df[${JSON.stringify(features)}]
y = df['${targetVariable}']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train Model
model = ${modelInstance}
print("--- Training Model ---")
model.fit(X_train, y_train)
print("Model training complete.\\n")

# 5. Make Predictions
y_pred = model.predict(X_test)

# 6. Evaluate Model
print("--- Model Evaluation ---")
${evaluationMetricsCode.trim()}
print("\\n")

# 7. Save Model
print("--- Saving Model ---")
joblib.dump(model, 'model.pkl')
print("Model saved to model.pkl")
`;
        
        return fullScript.trim();
    } catch (error) {
        console.error("Error generating Python script with Gemini:", error);
        throw new Error("Failed to generate Python script.");
    }
};
