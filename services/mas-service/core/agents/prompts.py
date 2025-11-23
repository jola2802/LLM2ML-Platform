"""
Zentrale Sammlung aller Prompts für die LLM-Agents
"""

DATA_ANALYSIS_PROMPT = """IMPORTANT: Respond ONLY with JSON, no explanations.

Analyze dataset and return JSON:

{{
  "summary": "2-3 sentence dataset summary",
  "dataQuality": {{
    "missingValues": "Missing values assessment",
    "dataCompleteness": "Completeness percentage/assessment",
    "potentialIssues": ["Issue1", "Issue2"]
  }},
  "dataCharacteristics": {{
    "numericColumns": ["col1", "col2"],
    "categoricalColumns": ["col3"],
    "keyInsights": ["Insight1", "Insight2"]
  }},
  "recommendations": {{
    "preprocessing": ["Step1", "Step2"],
    "featureEngineering": ["Suggestion1"],
    "modelType": "Classification|Regression"
  }},
  "targetVariableSuggestion": "target_var_name",
  "reasoning": "Brief analysis explanation"
}}

DATASET: Rows: {rowCount} | Cols: {columnCount} | Columns: {columns}
TYPES: {dataTypes}
NUMERIC: {numericStats}
CATEGORICAL: {categoricalStats}
SAMPLE: {sampleData}"""

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

FEATURE_ENGINEER_PROMPT = """IMPORTANT: Respond ONLY with JSON (and tool calls if needed).

Suggest new features from existing columns. Return JSON:

{{
  "generatedFeatures": [
    {{
      "name": "feature_name",
      "description": "Description",
      "formula": "Python code",
      "reasoning": "Why this feature helps"
    }}
  ],
  "reasoning": "Brief explanation"
}}

DATA: {dataOverview}
PREFERENCES: {userPreferences}

TOOLS: drop_columns, merge_columns, add_column, filter_rows, rename_columns"""

HYPERPARAMETER_OPTIMIZATION_PROMPT = """IMPORTANT: Respond ONLY with JSON.

Return optimized hyperparameters as JSON:

{{
  "hyperparameters": {{
    "param1": value1,
    "param2": value2
  }},
  "reasoning": "Why these are optimal",
  "expectedPerformance": "Expected improvement",
  "tuningStrategy": "Strategy used"
}}

CONTEXT: Algorithm={algorithm} | Type={modelType} | Target={targetVariable} | Features={numFeatures} | Samples={numSamples} | Problem={problemType}
DATA: {dataCharacteristics}
HYPERPARAMETERS: {hyperparameterInfo}
CURRENT: {currentHyperparameters}

GUIDELINES: Small datasets (<1000) → simpler params | High dimensions → regularization | Match problem type | Avoid overfitting"""

PERFORMANCE_EVALUATION_PROMPT = """IMPORTANT: Respond ONLY with JSON.

Evaluate performance and return JSON:

{{
  "overallScore": 0.0-10.0,
  "performanceGrade": "Excellent|Good|Fair|Poor",
  "summary": "Brief summary",
  "detailedAnalysis": {{
    "strengths": ["Strength1"],
    "weaknesses": ["Weakness1"],
    "keyFindings": ["Finding1"]
  }},
  "improvementSuggestions": [
    {{
      "category": "Data Quality|Feature Engineering|Algorithm Tuning",
      "suggestion": "Suggestion",
      "expectedImpact": "Low|Medium|High"
    }}
  ]
}}

PROJECT: {projectName} | Algorithm: {algorithm} | Type: {modelType} | Target: {targetVariable} | Features: {features}
METRICS: {performanceMetrics}"""

DECISION_PROMPT = """IMPORTANT: Respond ONLY with JSON.

Decide if pipeline should loop again. Return JSON:

{{
  "shouldContinue": true|false,
  "reason": "Why continue or stop",
  "suggestions": ["Suggestion1", "Suggestion2"]
}}

PROJECT: {projectName} | Iteration: {iteration}/{maxIterations}
SCORE: {overallScore}/10.0 | Grade: {performanceGrade} | Summary: {summary}
STRENGTHS: {strengths} | WEAKNESSES: {weaknesses} | SUGGESTIONS: {improvementSuggestions}

CRITERIA: Score >= 7.0 is good | Clear improvements available? | Max iterations reached? | Good enough for use case?"""

BASE_WORKER_TEST_PROMPT = """You are {agentName}. 
Respond with "OK" if you receive this message."""

DATA_CLEANING_PROMPT = """IMPORTANT: Respond ONLY with JSON.

Create cleaning plan. Return JSON:

{{
  "operations": [
    {{
      "type": "dropMissingRows|fillMissing|dropColumn|removeOutliers|encodeCategorial",
      "columns": ["col1", "col2"],
      "method": "mean|median|mode|constant|drop",
      "value": null,
      "threshold": 0.5,
      "reasoning": "Why needed"
    }}
  ],
  "reasoning": "Cleaning strategy",
  "priority": "high|medium|low"
}}

FILE: Rows: {rowCount} | Cols: {columnCount}
QUALITY: {qualitySummary}
MISSING: {missingValues}

OPS: dropMissingRows, fillMissing (mean/median/mode/constant), dropColumn, removeOutliers (iqr/zscore), encodeCategorical (onehot/label)
RULES: Preserve data > remove | Prefer fillMissing | Drop column only if >50% missing"""

CODE_REVIEW_PROMPT = """You are an experienced Software Engineer and ML Expert specializing in Code Review.
Your task is to review generated ML code for correctness, security, and best practices.

PROJECT CONTEXT:
{context}

STATIC ANALYSIS RESULTS:
{staticAnalysis}

CODE TO REVIEW:
```python
{code}
```

TASK: Perform a comprehensive code review and return EXACTLY the following JSON format:

{{
  "issues": [
    {{
      "type": "syntax|logic|security|performance|style",
      "severity": "critical|high|medium|low",
      "message": "Description of the issue",
      "line": 10,
      "suggestion": "How to fix it"
    }}
  ],
  "recommendations": [
    "Recommendation 1: Improve error handling",
    "Recommendation 2: Add input validation"
  ],
  "improvedCode": "```python\\n# Improved version of the code\\n```",
  "securityScore": 8.5,
  "qualityScore": 7.0,
  "reasoning": "Detailed explanation of the review findings"
}}

REVIEW CHECKLIST:
1. **Correctness**: Does the code do what it's supposed to do?
2. **Security**: Are there any security vulnerabilities? (eval, exec, SQL injection, etc.)
3. **Error Handling**: Are exceptions properly handled?
4. **ML Best Practices**: 
   - Is there a train-test split?
   - Are features properly scaled (if needed)?
   - Are metrics appropriate for the problem type?
5. **Code Quality**: Is the code readable, maintainable, and well-structured?
6. **Performance**: Are there any obvious performance issues?

SCORING:
- securityScore: 0-10 (10 = perfectly secure, 0 = critical vulnerabilities)
- qualityScore: 0-10 (10 = excellent code quality, 0 = poor quality)

IMPORTANT: 
- If static analysis found critical issues, address them in your review
- Provide specific, actionable suggestions
- If code is good, say so and give a high score
- Respond ONLY with the JSON object"""

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
