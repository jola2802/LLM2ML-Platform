"""
Zentrale Sammlung aller Prompts für die LLM-Agents
"""

LLM_RECOMMENDATIONS_PROMPT = """You are an experienced Machine Learning Expert. 
Analyze this automatic data overview and return PRECISE recommendations.

AUTOMATIC DATA OVERVIEW (ONLY ALLOWED FEATURES):
{dataOverview}

USER PREFERENCES (if provided):
{userPreferences}

{hyperparametersInfo}

TASK: Analyze the data and return EXACTLY the following recommendations in JSON format:

{{
  "targetVariable": "[Name of the target variable]",
  "features": ["[List of features]"],
  "generatedFeatures": [
    {{
      "name": "[Feature name]",
      "description": "[Description]",
      "formula": "[Python code]",
      "reasoning": "[Reasoning]"
    }}
  ],
  "modelType": "[Classification or Regression]",
  "algorithm": "[Best algorithm]",
  "hyperparameters": {{
    "[Parameter1]": [Value1]
  }},
  "reasoning": "[Explanation]",
  "dataSourceName": "[Dataset name]"
}}

IMPORTANT: Respond ONLY with the JSON object, no additional explanations."""

FEATURE_ENGINEER_PROMPT = """You are an experienced Machine Learning Expert specializing in Feature Engineering. 
Your task is to analyze the data and suggest NEW features that can be generated from existing columns.

AUTOMATIC DATA OVERVIEW (ONLY ALLOWED FEATURES):
{dataOverview}

USER PREFERENCES (if provided):
{userPreferences}

TASK: Analyze the data and return EXACTLY the following JSON format:

{{
  "generatedFeatures": [
    {{
      "name": "[Feature name]",
      "description": "[Description]",
      "formula": "[Python code]",
      "reasoning": "[Reasoning]"
    }}
  ],
  "reasoning": "[Explanation]"
}}

IMPORTANT: Respond ONLY with the JSON object."""

HYPERPARAMETER_OPTIMIZATION_PROMPT = """You are an experienced Machine Learning Expert specializing in Hyperparameter Optimization.
Your task is to analyze the project context and suggest OPTIMAL hyperparameters for the given algorithm.

PROJECT CONTEXT:
- Algorithm: {algorithm}
- Model Type: {modelType}
- Target Variable: {targetVariable}
- Number of Features: {numFeatures}
- Number of Training Samples: {numSamples}
- Problem Type: {problemType}

DATA CHARACTERISTICS:
{dataCharacteristics}

AVAILABLE HYPERPARAMETERS FOR {algorithm}:
{hyperparameterInfo}

CURRENT HYPERPARAMETERS (if any):
{currentHyperparameters}

TASK: Analyze the context and return EXACTLY the following JSON format with OPTIMIZED hyperparameters:

{{
  "hyperparameters": {{
    "[ParameterName]": [OptimalValue],
    "[ParameterName2]": [OptimalValue2]
  }},
  "reasoning": "Detailed explanation why these hyperparameters are optimal for this specific dataset and problem",
  "expectedPerformance": "Expected performance improvement or characteristics",
  "tuningStrategy": "Brief description of the tuning strategy used"
}}

IMPORTANT GUIDELINES:
1. Consider the dataset size: For small datasets (< 1000 samples), use simpler models with fewer parameters
2. Consider the number of features: For high-dimensional data, consider regularization
3. Consider the problem type: Classification vs Regression have different optimal parameters
4. Use reasonable default values that work well in practice
5. Avoid overfitting: Don't set parameters that are too complex for the dataset size
6. Respond ONLY with the JSON object, no additional explanations."""

PERFORMANCE_EVALUATION_PROMPT = """You are an experienced Machine Learning Expert. Evaluate the performance metrics.

PROJECT CONTEXT:
- Project name: {projectName}
- Algorithm: {algorithm}
- Model type: {modelType}
- Target variable: {targetVariable}
- Features: {features}

PERFORMANCE METRICS:
{performanceMetrics}

TASK: Perform a thorough performance analysis.

Respond in JSON format:
{{
  "overallScore": 0.0-10.0,
  "performanceGrade": "Excellent|Good|Fair|Poor",
  "summary": "Short summary",
  "detailedAnalysis": {{
    "strengths": ["Strength 1"],
    "weaknesses": ["Weakness 1"],
    "keyFindings": ["Finding 1"]
  }},
  "improvementSuggestions": [
    {{
      "category": "Data Quality|Feature Engineering|Algorithm Tuning",
      "suggestion": "Suggestion",
      "expectedImpact": "Low|Medium|High"
    }}
  ]
}}

IMPORTANT: Respond ONLY with the JSON object."""

BASE_WORKER_TEST_PROMPT = """You are {agentName}. 
Respond with "OK" if you receive this message."""

CODE_GENERATION_PROMPT = """Generate Python code for Machine Learning Training.

Project: {projectName}
File path: {csvFilePath}
Target: {targetVariable}
Algorithm: {algorithm}
Hyperparameters: {hyperparameters}

Generate complete Python code following best practices."""

def format_prompt(template: str, variables: dict) -> str:
    """Formatiere Prompt mit Variablen"""
    result = template
    for key, value in variables.items():
        result = result.replace(f'{{{key}}}', str(value))
    return result

def get_hyperparameters_info(algorithms: dict) -> str:
    """Generiere detaillierte Hyperparameter-Info für alle Algorithmen"""
    info_lines = []
    
    for algo_name, algo_config in algorithms.items():
        library = algo_config.get('library', 'sklearn')
        hyperparams = algo_config.get('hyperparameters', {})
        
        info_lines.append(f"\n{algo_name} ({library}):")
        info_lines.append(f"  Available hyperparameters:")
        for param_name, default_value in hyperparams.items():
            param_type = type(default_value).__name__
            if default_value is None:
                info_lines.append(f"    - {param_name}: {param_type} (default: None)")
            else:
                info_lines.append(f"    - {param_name}: {param_type} (default: {default_value})")
    
    return "\n".join(info_lines)

def get_algorithm_hyperparameter_info(algorithm: str, algorithms: dict) -> str:
    """Gibt detaillierte Hyperparameter-Info für einen spezifischen Algorithmus zurück"""
    algo_config = algorithms.get(algorithm, {})
    hyperparams = algo_config.get('hyperparameters', {})
    
    if not hyperparams:
        return f"No hyperparameter information available for {algorithm}"
    
    info_lines = [f"Hyperparameters for {algorithm}:"]
    for param_name, default_value in hyperparams.items():
        param_type = type(default_value).__name__
        if default_value is None:
            info_lines.append(f"  - {param_name}: {param_type} (default: None)")
        else:
            info_lines.append(f"  - {param_name}: {param_type} (default: {default_value})")
    
    return "\n".join(info_lines)
