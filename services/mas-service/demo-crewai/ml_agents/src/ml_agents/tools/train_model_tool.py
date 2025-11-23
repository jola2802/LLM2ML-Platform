from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import subprocess
from ml_agents.src.ml_agents.tools.prediction_template import PREDICTION_TEMPLATE
from ml_agents.src.ml_agents.tools.training_template import TRAINING_TEMPLATE
from datetime import datetime

class train_model_tool_input(BaseModel):
    project_name: str = Field(..., description="The name of the project.")
    file_path: str = Field(..., description="The absolute or relative path to the file.")
    target_column: str = Field(..., description="The name of the target column.")
    problem_type: str = Field(..., description="The problem type. Choose between 'classification' or 'regression'.")
    model_type: str = Field(..., description="The model type. Choose between 'sklearn', 'xgboost' or 'pytorch'.")
    hyperparameter_suggestions: str = Field(..., description="The hyperparameter suggestions in a JSON format.")
    generated_features: str = Field(..., description="The generated and selected features in a JSON format.")

class train_model_tool(BaseTool):
    name: str = "train_model_tool"
    description: str = (
        "Trains the model with the given input and returns the result."
    )
    args_schema: Type[BaseModel] = train_model_tool_input

    def _run(self, project_name: str, file_path: str, target_column: str, problem_type: str, model_type: str, hyperparameter_suggestions: str, generated_features: str) -> str:
        """Trains the model with the given input and returns the result."""
        try:
            base_code = TRAINING_TEMPLATE 
            code = base_code.replace("PROJECT_NAME", project_name)
            code = code.replace("FILE_PATH", file_path)
            code = code.replace("TARGET_COLUMN", target_column)
            code = code.replace("PROBLEM_TYPE", problem_type)
            code = code.replace("MODEL_TYPE", model_type)
            code = code.replace("HYPERPARAMETER_SUGGESTIONS", hyperparameter_suggestions)
            code = code.replace("GENERATED_FEATURES", generated_features)

            # Save the code to a file
            with open("training_ts_" + str(datetime.now().strftime("%Y%m%d_%H%M%S")) + ".py", "w") as f:
                f.write(code)

            # Execute the code
            result = subprocess.run(code, shell=True, capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            return f"FEHLER beim Ausführen des Codes: {str(e)}"
