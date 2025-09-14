/**
 * Code-Reviewer-Worker-Agent
 * 
 * Überprüft und optimiert generierten Python-Code.
 * Fokussiert auf Fehler, Performance-Probleme und Best Practices.
 */

import { BaseWorker } from './base_worker.js';

export class CodeReviewerWorker extends BaseWorker {
  constructor() {
    super('CODE_REVIEWER');
  }

  async execute(pipelineState) {
    this.log('info', 'Starte Code-Review');
    
    const { results } = pipelineState;
    
    // Prüfe, ob Python-Code verfügbar ist
    if (!results.CODE_GENERATOR) {
      throw new Error('Python-Code erforderlich für Code-Review');
    }

    try {
      const reviewedCode = await this.reviewAndOptimizeCode(results.CODE_GENERATOR);
      
      this.log('success', 'Code-Review erfolgreich abgeschlossen');
      return reviewedCode;

    } catch (error) {
      this.log('error', 'Code-Review fehlgeschlagen', error.message);
      throw error;
    }
  }

  async reviewAndOptimizeCode(pythonCode) {
    const prompt = `Überprüfe und optimiere den folgenden Python-Code für Machine Learning:

CODE ZUM REVIEW:
\`\`\`python
${pythonCode}
\`\`\`

REVIEW-KRITERIEN:
1. **Syntax und Fehler**: Prüfe auf Syntax-Fehler und logische Probleme
2. **Performance**: Identifiziere Performance-Bottlenecks und Optimierungsmöglichkeiten
3. **Best Practices**: Überprüfe Python- und ML-Best-Practices
4. **Code-Qualität**: Verbessere Lesbarkeit, Struktur und Dokumentation
5. **Fehlerbehandlung**: Ergänze robuste Fehlerbehandlung
6. **Moderne Bibliotheken**: Verwende aktuelle und effiziente Bibliotheken
7. **Memory-Effizienz**: Optimiere Speicherverbrauch
8. **Skalierbarkeit**: Verbessere Code für größere Datasets

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

    const response = await this.callLLM(prompt);
    const text = typeof response === 'string' ? response : response?.result || '';
    
    // Extrahiere Code aus der Antwort
    let reviewedCode = this.extractCode(text);
    
    // Bereinige und validiere den Code
    reviewedCode = this.cleanCode(reviewedCode);
    reviewedCode = this.validateReviewedCode(reviewedCode, pythonCode);
    
    return reviewedCode;
  }

  validateReviewedCode(reviewedCode, originalCode) {
    // Validiere, dass der Review-Code nicht leer ist
    if (!reviewedCode || reviewedCode.length < 100) {
      this.log('warn', 'Review-Code ist zu kurz, verwende Original-Code mit Verbesserungen');
      return this.enhanceOriginalCode(originalCode);
    }

    // Validiere, dass der Review-Code nicht drastisch kürzer ist als der Original-Code
    if (reviewedCode.length < originalCode.length * 0.5) {
      this.log('warn', 'Review-Code ist deutlich kürzer als Original, verwende Original mit Verbesserungen');
      return this.enhanceOriginalCode(originalCode);
    }

    // Ergänze fehlende Imports falls nötig
    if (!reviewedCode.includes('import pandas') && !reviewedCode.includes('import pd')) {
      reviewedCode = 'import pandas as pd\n' + reviewedCode;
    }

    if (!reviewedCode.includes('import sklearn') && !reviewedCode.includes('from sklearn')) {
      reviewedCode = 'from sklearn.model_selection import train_test_split\nfrom sklearn.metrics import classification_report, confusion_matrix\n' + reviewedCode;
    }

    return reviewedCode;
  }

  enhanceOriginalCode(originalCode) {
    // Füge grundlegende Verbesserungen zum Original-Code hinzu
    let enhancedCode = originalCode;

    // Füge Header hinzu falls nicht vorhanden
    if (!enhancedCode.includes('#!/usr/bin/env python3')) {
      enhancedCode = `#!/usr/bin/env python3
"""
ML-Training Script (Code-Review optimiert)
Generiert von: ${this.agentKey}
Datum: ${new Date().toISOString()}
"""

` + enhancedCode;
    }

    // Ergänze Fehlerbehandlung falls nicht vorhanden
    if (!enhancedCode.includes('try:') && !enhancedCode.includes('except')) {
      enhancedCode = enhancedCode.replace(
        /def main\(\):/,
        `def main():
    try:`
      );
      
      enhancedCode += `
    except Exception as e:
        print(f"❌ Fehler beim Training: {e}")
        return False
    
    return True`;
    }

    // Ergänze Logging falls nicht vorhanden
    if (!enhancedCode.includes('print(') && !enhancedCode.includes('logging')) {
      enhancedCode = enhancedCode.replace(
        /def main\(\):/,
        `def main():
    print("🚀 Starte ML-Training...")`
      );
    }

    return enhancedCode;
  }

  async test() {
    const testPrompt = `Du bist der ${this.agentKey}. 
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
