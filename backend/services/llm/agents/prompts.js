/**
 * Zentrale Sammlung aller Prompts für die LLM-Agents
 * 
 * Diese Datei enthält alle Prompts, die in der LLM-Pipeline verwendet werden.
 * Alle Prompts sind hier zentral organisiert und können einfach verwaltet werden.
 */

// ============================================================================
// LLM-API PROMPTS (aus llm_api.js)
// ============================================================================

export const LLM_RECOMMENDATIONS_PROMPT = `Du bist ein erfahrener Machine Learning Experte. Analysiere diese automatische Datenübersicht und gib PRÄZISE Empfehlungen zurück.

AUTOMATISCHE DATENÜBERSICHT (NUR ERLAUBTE FEATURES):
{dataOverview}

NUTZERWÜNSCHE (falls vorhanden):
{userPreferences}

AUFGABE: Analysiere die Daten und gib EXAKT die folgenden Empfehlungen zurück im JSON-Format:

{
  "targetVariable": "[Name der Zielvariable - die Spalte die vorhergesagt werden soll]", // WICHTIG: NUR die Spalte die vorhergesagt werden soll, keine sonstigen Namen sind erlaubt; 
  "features": ["[Liste der Features/Eingangsvariablen ohne die Zielvariable - NUR aus den verfügbaren Spalten]"],
  "modelType": "[Classification oder Regression]",
  "algorithm": "[Bester Algorithmus: RandomForestClassifier, LogisticRegression, SVM, XGBoostClassifier, RandomForestRegressor, LinearRegression, SVR, XGBoostRegressor, MLPClassifier, MLPRegressor]",
  "hyperparameters": {
    "[Parameter1]": "[Wert1]",
    "[Parameter2]": "[Wert2]"
  },
  "reasoning": "[Kurze Begründung der Entscheidungen]",
  "dataSourceName": "[Aussagekräftiger Name für das Dataset]"
}

 WICHTIGE REGELN:
1. Identifiziere die wahrscheinlichste Zielvariable aus den verfügbaren Spalten
2. Verwende NUR die verfügbaren Spalten als Features (ausgeschlossene Spalten sind nicht verfügbar)
3. IMPORTANT: Schließe sinnlose Features wie "ID", "Name" aus; Schließe auch Features aus, die nichts mit der Aufgabe zu tun haben
4. Bestimme ob es sich um Classification (kategorische Zielvariable) oder Regression (numerische Zielvariable) handelt
5. Wähle den besten Algorithmus basierend auf den verfügbaren Daten
6. Überlege genau und gebe die wahrscheinlich besten Hyperparameter passend zu dem Datensatz, dem Algorithmus und der Aufgabe an
7. Antworte NUR mit dem JSON-Objekt, keine zusätzlichen Erklärungen außerhalb

 WICHTIG: Berücksichtige ausdrücklich die NUTZERWÜNSCHE, sofern diese nicht im Widerspruch zur Datenlage stehen (z. B. eine Zielvariable, die nicht existiert, darf ignoriert werden). Priorisiere valide Nutzerangaben wie gewünschte Zielvariable, bevorzugter Modelltyp/Algorithmus oder auszuschließende Features.

 WICHTIG: Gib NUR das JSON-Objekt zurück, keine Markdown-Formatierung oder zusätzlichen Text.`;

export const PERFORMANCE_EVALUATION_PROMPT = `Du bist ein erfahrener Machine Learning Experte und Performance-Analyst. Bewerte die Performance-Metriken dieses ML-Modells umfassend und professionell.

PROJEKT-KONTEXT:
- Projektname: {projectName}
- Algorithmus: {algorithm}
- Model-Typ: {modelType}
- Zielvariable: {targetVariable}
- Features: {features}
- Datenquelle: {dataSourceName}

PERFORMANCE-METRIKEN:
{performanceMetrics}

URSPRÜNGLICHE KI-EMPFEHLUNGEN:
{recommendations}

AUFGABE: Führe eine tiefgehende Performance-Analyse durch und erstelle einen professionellen Evaluationsbericht.

Antworte im folgenden JSON-Format:
{
  "overallScore": 0.0-10.0,
  "performanceGrade": "Excellent|Good|Fair|Poor|Critical",
  "summary": "Kurze, prägnante Zusammenfassung der Model-Performance in 1-2 Sätzen",
  "detailedAnalysis": {
    "strengths": ["Stärke 1", "Stärke 2", "Stärke 3"],
    "weaknesses": ["Schwäche 1", "Schwäche 2"],
    "keyFindings": ["Wichtiger Befund 1", "Wichtiger Befund 2"]
  },
  "metricsInterpretation": {metricsInterpretationTemplate},
  "improvementSuggestions": [
    {
      "category": "Data Quality|Feature Engineering|Algorithm Tuning|Model Architecture",
      "suggestion": "Konkrete Verbesserungsempfehlung",
      "expectedImpact": "Low|Medium|High",
      "implementation": "Wie kann das umgesetzt werden?"
    }
  ],
  "businessImpact": {
    "readiness": "Production Ready|Needs Improvement|Not Ready",
    "riskAssessment": "Low|Medium|High",
    "recommendation": "Empfehlung für den Business-Einsatz"
  },
  "nextSteps": [
    "Nächster Schritt 1",
    "Nächster Schritt 2"
  ],
  "confidenceLevel": 0.0-1.0,
  "version": "1.0"
}
WICHTIG: 
- Interpretiere ALLE verfügbaren Metriken in metricsInterpretation
- Verwende die exakten Metrik-Namen und -Werte aus den Performance-Metriken
- Gib eine fundierte, datengetriebene Analyse ab
- Nur gültiges JSON zurückgeben, keine zusätzlichen Kommentare oder Texte
- Antworte NUR mit dem JSON-Objekt, keine zusätzlichen Erklärungen außerhalb
- Antworten müssen in deutscher Sprache sein`;

// ============================================================================
// TUNING PROMPTS (aus tuning.js)
// ============================================================================

export const AUTO_TUNING_PROMPT = `Du bist ein erfahrener Machine-Learning-Experte. Basierend auf diesem Kontext, schlage eine verbesserte Konfiguration (Algorithmus + Hyperparameter) vor, die die Performance voraussichtlich erhöht. Gib NUR JSON zurück.

KONTEXT:
- Modelltyp: {modelType}
- Aktueller Algorithmus: {algorithm}
- Aktuelle Hyperparameter: {hyperparameters}
- Features: {features}
- Zielvariable: {targetVariable}
- Letzte Performance-Metriken: {performanceMetrics}
- Datensatz: {dataSourceName}

ANTWORTFORMAT (JSON):
{
  "algorithm": "Name des Algorithmus",
  "hyperparameters": { "param": Wert },
  "expectedGain": Zahl zwischen 0 und 1,
  "rationale": "kurze Begründung"
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
