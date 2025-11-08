/**
 * Zentrale Sammlung aller Prompts für die LLM-Agents
 * 
 * Diese Datei enthält alle Prompts, die in der LLM-Pipeline verwendet werden.
 * Alle Prompts sind hier zentral organisiert und können einfach verwaltet werden.
 */

export const LLM_RECOMMENDATIONS_PROMPT = `You are an experienced Machine Learning Expert. 
Analyze this automatic data overview and return PRECISE recommendations.

AUTOMATIC DATA OVERVIEW (ONLY ALLOWED FEATURES):
{dataOverview}

USER PREFERENCES (if provided):
{userPreferences}

TASK: Analyze the data and return EXACTLY the following recommendations in JSON format:

{
  "targetVariable": "[Name of the target variable - the column that should be predicted]", // IMPORTANT: ONLY the column that should be predicted, no other names are allowed; 
  "features": ["[List of features/input variables without the target variable]"],
  "modelType": "[Classification or Regression]",
  "algorithm": "[Best algorithm: RandomForestClassifier, LogisticRegression, SVM, XGBoostClassifier, RandomForestRegressor, LinearRegression, SVR, XGBoostRegressor, MLPClassifier, MLPRegressor]",
  "hyperparameters": {
    "[Parameter1]": "[Value1]",
    "[Parameter2]": "[Value2]"
  },
  "reasoning": "[Short explanation of the decisions made for target variable, features and algorithm]",
  "dataSourceName": "[Descriptive name for the dataset]"
}

 IMPORTANT RULES:
1. Identify the most likely target variable from the available columns unless it was explicitly specified
2. Use ONLY the available columns as features (excluded columns are not available)
3. Exclude meaningless features like "ID", "Name"; Exclude also features that are not related to the task; Exclude features that are not related to the target variable
4. Determine if it is a Classification (categorical target variable) or Regression (numerical target variable)
5. Select the best algorithm based on the available data and the task
6. Think carefully and give the best hyperparameters suitable for the dataset, algorithm and task; The best hyperparameters are the ones that maximize the performance of the model
7. Respond ONLY with the JSON object, no additional explanations outside

 IMPORTANT: Consider the USER PREFERENCES explicitly, unless they contradict the data situation (e.g. a target variable that does not exist should be ignored). Prioritize valid user specifications like desired target variable, preferred model type/algorithm or excluded features.`;

export const PERFORMANCE_EVALUATION_PROMPT = `You are an experienced Machine Learning Expert and Performance Analyst. Evaluate the performance metrics of this ML model comprehensively and professionally.

PROJECT CONTEXT:
- Project name: {projectName}
- Algorithm: {algorithm}
- Model type: {modelType}
- Target variable: {targetVariable}
- Features: {features}
- Data source: {dataSourceName}

PERFORMANCE METRICS:
{performanceMetrics}

ORIGINAL LLM RECOMMENDATIONS:
{recommendations}

TASK: Perform a thorough performance analysis and create a professional evaluation report.

Respond in the following JSON format:
{
  "overallScore": 0.0-10.0,
  "performanceGrade": "Excellent|Good|Fair|Poor|Critical",
  "summary": "Short, concise summary of the model performance in 1-2 sentences",
  "detailedAnalysis": {
    "strengths": ["Strength 1", "Strength 2", "Strength 3"],
    "weaknesses": ["Weakness 1", "Weakness 2"],
    "keyFindings": ["Key finding 1", "Key finding 2"]
  },
  "metricsInterpretation": {metricsInterpretationTemplate},
  "improvementSuggestions": [
    {
      "category": "Data Quality|Feature Engineering|Algorithm Tuning|Model Architecture",
      "suggestion": "Concrete improvement suggestion",
      "expectedImpact": "Low|Medium|High",
      "implementation": "How can this be implemented?"
    }
  ],
  "businessImpact": {
    "readiness": "Production Ready|Needs Improvement|Not Ready",
    "riskAssessment": "Low|Medium|High",
    "recommendation": "Recommendation for business usage"
  },
  "nextSteps": [
    "Next step 1",
    "Next step 2"
  ],
  "confidenceLevel": 0.0-1.0,
  "version": "1.0"
}
IMPORTANT: 
- Interpret ALL available metrics in metricsInterpretation
- Use the exact metric names and values from the performance metrics
- Provide a thorough, data-driven analysis
- Respond ONLY with the JSON object, no additional comments or text
- Respond ONLY with the JSON object, no additional explanations outside
- Responses must be in English`;

// ============================================================================
// TUNING PROMPTS (aus tuning.js)
// ============================================================================

export const AUTO_TUNING_PROMPT = `You are an experienced Machine Learning Expert. Based on this context, suggest an improved configuration (algorithm + hyperparameters) that is expected to increase performance. Respond ONLY with the JSON object.

CONTEXT:
- Model type: {modelType}
- Current algorithm: {algorithm}
- Current hyperparameters: {hyperparameters}
- Features: {features}
- Target variable: {targetVariable}
- Last performance metrics: {performanceMetrics}
- Dataset: {dataSourceName}

RESPONSE FORMAT (JSON):
{
  "algorithm": "Name of the algorithm",
  "hyperparameters": { "param": value },
  "expectedGain": number between 0 and 1,
  "rationale": "short justification"
}`;

// ============================================================================
// CODE GENERATOR PROMPTS (aus code_generator_worker.js)
// ============================================================================

export const CODE_GENERATION_PROMPT = `Generiere einen vollständigen Python-Code für Machine Learning Training:

**Code-Muster:** Muss dem bereitgestellten Muster folgen, einschließlich der Schritte Laden/Splitten, Instanziieren/Trainieren, Vorhersagen, Bewerten und Speichern.

**Projektname:** {projectName}

**Daten laden und aufteilen (Schritt 1):**
* **Laden:** Funktion 'load_and_split_data' verwenden.
* **Dateipfad:** {csvFilePath} 
* **Features:** {features}

**Hyperparameter (Schritt 2):**
* **Algorithmus:** {algorithm}
* **Hyperparameter:** {hyperparameters}
* **Zielspalte:** {targetColumn}

**Vorhersagen (Schritt 3):**
* **Bibliothek:** 'predict' verwenden.

**Performance-Metriken (Schritt 4):**
* **Test-Metriken:** Implementiere **alle Standardmetriken** für den definierten **Problemtyp** ({problemType}).
    * **Falls Klassifikation:** **'classification_report'** und **'confusion_matrix'** (Visualisiert mit 'seaborn').
    * **Falls Regression:** **'mean_squared_error'** ('MSE') und **'r2_score'**.

**Speichern (Schritt 5):**
* **Bibliothek:** 'joblib.dump' verwenden.
* **Dateiname:** '{projectName}_model.pkl'.`;

export const CODE_GENERATOR_TEST_PROMPT = `Du bist der {agentName}. 
Teste die Code-Generierung mit einem einfachen Beispiel:
- Dataset: Iris
- Algorithmus: RandomForestClassifier
- Ziel: Vollständiger Python-Code

Antworte nur mit "OK" wenn du bereit bist.`;

// ============================================================================
// CODE REVIEWER PROMPTS (aus code_reviewer_worker.js)
// ============================================================================

export const CODE_REVIEW_PROMPT = `Überprüfe und optimiere den folgenden Python-Code für Machine Learning:

CODE ZUM REVIEW:
\`\`\`python
{pythonCode}
\`\`\`

REVIEW-KRITERIEN:
1. **Syntax und Fehler**: Prüfe auf Syntax-Fehler und logische Probleme
2. **Performance**: Identifiziere Performance-Bottlenecks und Optimierungsmöglichkeiten
3. **Best Practices**: Überprüfe Python- und ML-Best-Practices
4. **Fehlerbehandlung**: Ergänze robuste Fehlerbehandlung
5. **Memory-Effizienz**: Optimiere Speicherverbrauch

OPTIMIERUNGEN:
- Verbessere Code-Struktur und Modularität
- Ergänze aussagekräftige Kommentare und Docstrings
- Optimiere Imports und Abhängigkeiten
- Verbessere Fehlerbehandlung und Logging
- Optimiere Performance-kritische Bereiche
- Ergänze Validierung und Checks

ANTWORTFORMAT:
Gib den vollständigen, optimierten Python-Code zurück. Der Code sollte:
- Syntaktisch korrekt und ausführbar sein
- Bessere Performance haben
- Robuster und wartbarer sein
- Best Practices befolgen

Falls der Code bereits gut ist, gib ihn mit minimalen Verbesserungen zurück.`;

export const CODE_REVIEWER_TEST_PROMPT = `Du bist der {agentName}. 
Teste die Code-Review-Funktionalität mit einem einfachen Beispiel:

Code zum Review:
\`\`\`python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
\`\`\`

Antworte nur mit "OK" wenn du bereit bist, diesen Code zu reviewen.`;

// ============================================================================
// DATA ANALYZER PROMPTS (aus data_analyzer_worker.js)
// ============================================================================

export const DATA_ANALYSIS_PROMPT = `Führe eine umfassende Datenanalyse für das Dataset durch:

Dataset-Pfad: {csvFilePath}
Projekt: {projectName}
Algorithmus: {algorithm}

Analysiere folgende Aspekte:
1. Dataset-Übersicht (Größe, Spalten, Datentypen)
2. Fehlende Werte und Datenqualität
3. Statistische Beschreibungen
4. Korrelationen zwischen Features
5. Ausreißer-Erkennung
6. Verteilung der Zielvariable (falls vorhanden)
7. Empfehlungen für Preprocessing

Gib eine strukturierte Analyse zurück.`;

export const ENHANCED_DATA_ANALYSIS_PROMPT = `Erweitere die folgende Datenanalyse mit zusätzlichen ML-spezifischen Insights:

Vorhandene Analyse:
{dataAnalysis}

Projekt-Kontext:
- Name: {projectName}
- Algorithmus: {algorithm}
- Features: {featuresCount} ausgewählt

Füge folgende Erkenntnisse hinzu:
1. ML-Algorithmus-Empfehlungen basierend auf Datencharakteristika
2. Feature-Engineering-Vorschläge
3. Preprocessing-Empfehlungen
4. Potentielle Herausforderungen und Lösungsansätze
5. Erwartete Modell-Performance-Indikatoren

Gib eine erweiterte, strukturierte Analyse zurück.`;

// ============================================================================
// HYPERPARAMETER OPTIMIZER PROMPTS (aus hyperparameter_optimizer_worker.js)
// ============================================================================

export const HYPERPARAMETER_OPTIMIZATION_PROMPT = `Basierend auf der folgenden Datenanalyse und deinem Fachwissen, schlage optimale Hyperparameter vor:

DATENANALYSE:
{dataAnalysis}

PROJEKT-KONTEXT:
- Name: {projectName}
- Algorithmus: {algorithm}
- Features: {featuresCount} ausgewählt
- Dataset-Größe: {datasetSize}

AUFGABE:
Schlage für den angegebenen Algorithmus (oder für mehrere geeignete Algorithmen) optimale Hyperparameter vor.

Berücksichtige:
1. Dataset-Größe und -Komplexität
2. Feature-Anzahl und -Typen
3. Problemtyp (Klassifikation/Regression)
4. Die Hyperparameter müssen als Zahlenwerte zurückgegeben werden, keine Strings oder andere Formate.

ANTWORTFORMAT:
Gib eine JSON-Antwort mit EXAKT folgender Struktur zurück:
{
  "primary_algorithm": "Algorithmus-Name",
  "hyperparameters": {
      "param1": "wert1",
      "param2": "wert2",
      ...
  },
  "reasoning": "Erklärung der Hyperparameter-Auswahl",
  "expected_performance": "Erwartete Performance-Indikatoren"
}`;

export const HYPERPARAMETER_OPTIMIZER_TEST_PROMPT = `Du bist der {agentName}. 
Teste die Hyperparameter-Optimierung mit einem einfachen Beispiel:
Dataset: Iris (150 Samples, 4 Features, 3 Klassen)
Algorithmus: RandomForestClassifier

Antworte nur mit "OK" wenn du bereit bist.`;

// ============================================================================
// PERFORMANCE ANALYZER PROMPTS (aus performance_analyzer_worker.js)
// ============================================================================

export const PERFORMANCE_ANALYSIS_PROMPT = `Analysiere die Performance des folgenden ML-Codes und gib Verbesserungsvorschläge:

PYTHON-CODE:
\`\`\`python
{pythonCode}
\`\`\`

PROJEKT-KONTEXT:
- Name: {projectName}
- Algorithmus: {algorithm}
- Dataset: {csvFilePath}

DATENANALYSE:
{dataAnalysis}

HYPERPARAMETER:
{hyperparameterSuggestions}

ANALYSE-BEREICHE:
1. **Code-Performance**: Identifiziere Performance-Bottlenecks im Code
2. **Algorithmus-Auswahl**: Bewerte die Wahl des ML-Algorithmus
3. **Hyperparameter-Optimierung**: Analysiere die Hyperparameter-Auswahl
4. **Feature-Engineering**: Bewerte Feature-Auswahl und -Preprocessing
5. **Modell-Evaluation**: Überprüfe Evaluations-Metriken und -Methoden
6. **Skalierbarkeit**: Bewerte Code für größere Datasets
7. **Robustheit**: Analysiere Fehlerbehandlung und Edge Cases

VERBESSERUNGSVORSCHLÄGE:
- Konkrete Code-Optimierungen
- Alternative Algorithmen
- Bessere Hyperparameter
- Feature-Engineering-Verbesserungen
- Erweiterte Evaluations-Metriken
- Performance-Monitoring

ANTWORTFORMAT:
Gib eine strukturierte Analyse zurück mit:
- Performance-Bewertung (1-10)
- Identifizierte Probleme
- Konkrete Verbesserungsvorschläge
- Erwartete Performance-Steigerung
- Priorisierte Empfehlungen`;

export const PERFORMANCE_ANALYZER_TEST_PROMPT = `Du bist der {agentName}. 
Teste die Performance-Analyse mit einem einfachen Beispiel:

Code zum Analysieren:
\`\`\`python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('data.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)
\`\`\`

Antworte nur mit "OK" wenn du bereit bist, die Performance zu analysieren.`;

// ============================================================================
// BASE WORKER TEST PROMPTS (aus base_worker.js)
// ============================================================================

export const BASE_WORKER_TEST_PROMPT = `Du bist der {agentName}. Antworte nur mit "OK" wenn du diese Nachricht erhältst.`;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Ersetzt Platzhalter in Prompts mit tatsächlichen Werten
 * @param {string} prompt - Der Prompt mit Platzhaltern
 * @param {object} variables - Objekt mit Variablen für die Ersetzung
 * @returns {string} - Der Prompt mit ersetzten Werten
 */
export function formatPrompt(prompt, variables = {}) {
  let formattedPrompt = prompt;

  // Ersetze alle Platzhalter {variable} mit den entsprechenden Werten
  for (const [key, value] of Object.entries(variables)) {
    const placeholder = `{${key}}`;
    const replacement = typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
    formattedPrompt = formattedPrompt.replace(new RegExp(placeholder, 'g'), replacement);
  }

  return formattedPrompt;
}

/**
 * Validiert, ob alle erforderlichen Variablen für einen Prompt vorhanden sind
 * @param {string} prompt - Der Prompt mit Platzhaltern
 * @param {object} variables - Objekt mit verfügbaren Variablen
 * @returns {object} - {valid: boolean, missing: string[]}
 */
export function validatePromptVariables(prompt, variables = {}) {
  const placeholders = prompt.match(/\{([^}]+)\}/g) || [];
  const requiredVars = placeholders.map(p => p.slice(1, -1));
  const availableVars = Object.keys(variables);
  const missing = requiredVars.filter(varName => !availableVars.includes(varName));

  return {
    valid: missing.length === 0,
    missing
  };
}
