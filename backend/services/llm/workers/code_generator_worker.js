/**
 * Code-Generator-Worker-Agent
 * 
 * Generiert optimierten Python-Code für ML-Training.
 * Verwendet moderne ML-Bibliotheken und folgt Best Practices.
 */

import { BaseWorker } from './base_worker.js';

export class CodeGeneratorWorker extends BaseWorker {
  constructor() {
    super('CODE_GENERATOR');
  }

  async execute(pipelineState) {
    this.log('info', 'Starte Code-Generierung');
    
    const { project, results } = pipelineState;
    
    // Prüfe, ob Hyperparameter verfügbar sind
    if (!results.HYPERPARAMETER_OPTIMIZER) {
      throw new Error('Hyperparameter-Vorschläge erforderlich für Code-Generierung');
    }

    try {
      const pythonCode = await this.generatePythonCode(
        project,
        results.DATA_ANALYZER,
        results.HYPERPARAMETER_OPTIMIZER
      );
      
      this.log('success', 'Code-Generierung erfolgreich abgeschlossen');
      return pythonCode;

    } catch (error) {
      this.log('error', 'Code-Generierung fehlgeschlagen', error.message);
      throw error;
    }
  }

  async generatePythonCode(project, dataAnalysis, hyperparameterSuggestions) {
    const prompt = `Generiere einen vollständigen Python-Code für Machine Learning Training:

PROJEKT-INFORMATIONEN:
- Name: ${project.name}
- Algorithmus: ${hyperparameterSuggestions.primary_algorithm || project.algorithm}
- Dataset-Pfad: ${project.csvFilePath}
- Features: ${Array.isArray(project.features) ? project.features.join(', ') : 'Alle Features'}
- Hyperparameter-Empfehlungen: ${JSON.stringify(hyperparameterSuggestions, null, 2)}

Folgende Datenanalyse ist verfügbar:
${JSON.stringify(dataAnalysis, null, 2)}

ANFORDERUNGEN:
1. Verwende moderne Python ML-Bibliotheken (pandas, scikit-learn, numpy, matplotlib)
2. Implementiere vollständige ML-Pipeline: Laden → Preprocessing → Training → Evaluation
3. Verwende die vorgeschlagenen Hyperparameter
4. Füge aussagekräftige Kommentare hinzu
5. Implementiere Fehlerbehandlung
6. Zeige Trainings- und Test-Performance mit aussagekräftigen Metriken
7. Speichere das trainierte Modell

CODE-STRUKTUR:
'start code'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    # 1. Daten laden
    # 2. Preprocessing
    # 3. Train-Test-Split
    # 4. Modell-Training
    # 5. Evaluation
    # 6. Visualisierungen
    # 7. Modell speichern

if __name__ == "__main__":
    main()
'end code'

Generiere den vollständigen, ausführbaren Python-Code.`;

    const response = await this.callLLM(prompt);
    const text = typeof response === 'string' ? response : response?.result || '';
    
    // Extrahiere Code aus der Antwort
    let pythonCode = this.extractCode(text);
    
    // Bereinige und validiere den Code
    pythonCode = this.cleanCode(pythonCode);
    pythonCode = this.validateAndEnhanceCode(pythonCode, project, hyperparameterSuggestions);
    
    return pythonCode;
  }

  validateAndEnhanceCode(code, project, hyperparameterSuggestions) {
    // Validiere Code-Länge
    if (!code || code.length < 500) {
      this.log('warn', 'Generierter Code ist zu kurz, verwende Fallback-Code');
      return this.getFallbackCode(project, hyperparameterSuggestions);
    }

    // Ergänze fehlende Imports falls nötig
    if (!code.includes('import pandas') && !code.includes('import pd')) {
      code = 'import pandas as pd\n' + code;
    }

    if (!code.includes('import sklearn') && !code.includes('from sklearn')) {
      code = 'from sklearn.model_selection import train_test_split\nfrom sklearn.metrics import classification_report, confusion_matrix\n' + code;
    }

    // Füge Header hinzu falls nicht vorhanden
    if (!code.includes('#!/usr/bin/env python3')) {
      code = `#!/usr/bin/env python3
"""
ML-Training Script für ${project.name}
Generiert von: ${this.agentKey}
Datum: ${new Date().toISOString()}
"""

` + code;
    }

    return code;
  }

  getFallbackCode(project, hyperparameterSuggestions) {
    const algorithm = hyperparameterSuggestions.primary_algorithm || project.algorithm || 'RandomForestClassifier';
    const datasetPath = project.csvFilePath || 'data.csv';
    
    return `#!/usr/bin/env python3
"""
ML-Training Script für ${project.name}
Generiert von: ${this.agentKey} (Fallback)
Datum: ${new Date().toISOString()}
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 Starte ML-Training...")
    
    # 1. Daten laden
    try:
        df = pd.read_csv('${datasetPath}')
        print(f"✅ Dataset geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
    except Exception as e:
        print(f"❌ Fehler beim Laden des Datasets: {e}")
        return
    
    # 2. Preprocessing
    # Entferne Zeilen mit fehlenden Werten
    df = df.dropna()
    print(f"📊 Nach Preprocessing: {df.shape[0]} Zeilen")
    
    # 3. Features und Zielvariable vorbereiten
    # Annahme: Letzte Spalte ist die Zielvariable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # 4. Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📈 Train-Set: {X_train.shape[0]} Samples")
    print(f"📈 Test-Set: {X_test.shape[0]} Samples")
    
    # 5. Modell-Training
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    print("🤖 Trainiere Modell...")
    model.fit(X_train, y_train)
    
    # 6. Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\\n📊 ERGEBNISSE:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 7. Modell speichern
    model_path = '${project.name}_model.pkl'
    joblib.dump(model, model_path)
    print(f"💾 Modell gespeichert: {model_path}")
    
    print("✅ Training abgeschlossen!")

if __name__ == "__main__":
    main()`;
  }

  async test() {
    const testPrompt = `Du bist der ${this.config.name}. 
Teste die Code-Generierung mit einem einfachen Beispiel:
- Dataset: Iris
- Algorithmus: RandomForestClassifier
- Ziel: Vollständiger Python-Code

Antworte nur mit "OK" wenn du bereit bist.`;

    try {
      const response = await this.callLLM(testPrompt, null, 10);
      const result = typeof response === 'string' ? response : response?.result || '';
      return result.toLowerCase().includes('ok');
    } catch (error) {
      this.log('error', 'Test fehlgeschlagen', error.message);
      return false;
    }
  }
}
