// Metriken aus Python-Ausgabe extrahieren
export function extractMetricsFromOutput(output, modelType) {
    const metrics = {};
    const foundMetrics = new Set(); // Track bereits gefundene Metriken

    // Erweiterte Metriken-Extraktionsmuster mit Priorität für ausgeschriebene Namen
    const metricPatterns = [
        {
            primaryName: 'mean_absolute_error',
            aliases: ['mae'],
            regexes: [
                /(?:MAE):\s*([\d.]+)/i
            ]
        },
        {
            primaryName: 'mean_squared_error',
            aliases: ['mse'],
            regexes: [
                /(?:MSE):\s*([\d.]+)/i
            ]
        },
        {
            primaryName: 'root_mean_squared_error',
            aliases: ['rmse'],
            regexes: [
                /(?:RMSE):\s*([\d.]+)/i
            ]
        },
        {
            primaryName: 'r_squared',
            aliases: ['r2'],
            regexes: [
                /(?:R2):\s*([\d.]+)/i
            ]
        },
        {
            primaryName: 'accuracy',
            aliases: [],
            regexes: [
                /(?:Accuracy):\s*([\d.]+)/i
            ]
        },
        {
            primaryName: 'precision',
            aliases: [],
            regexes: [
                /(?:Precision)(?:\s*\(Precision\))?:\s*([\d.]+)/i,
                /(?:Precision):\s*([\d.]+)/i
            ]
        },
        {
            primaryName: 'recall',
            aliases: [],
            regexes: [
                /(?:Recall)(?:\s*\(Recall\))?:\s*([\d.]+)/i,
                /(?:Recall):\s*([\d.]+)/i
            ]
        },
        {
            primaryName: 'f1_score',
            aliases: ['f1'],
            regexes: [
                /(?:F1 Score|F1)(?:\s*\(F1\))?:\s*([\d.]+)/i,
                /(?:F1 Score|F1):\s*([\d.]+)/i
            ]
        }
    ];

    // Extrahiere Metriken mit Priorität für ausgeschriebene Namen
    metricPatterns.forEach(metricGroup => {
        // Prüfe ob diese Metrik bereits gefunden wurde
        const allNames = [metricGroup.primaryName, ...metricGroup.aliases];
        const alreadyFound = allNames.some(name => foundMetrics.has(name));

        if (alreadyFound) {
            return; // Überspringe diese Metrik-Gruppe
        }

        for (const regex of metricGroup.regexes) {
            const match = output.match(regex);
            if (match) {
                const value = parseFloat(match[1]);

                // Validiere den Wert
                if (isNaN(value)) {
                    continue; // Überspringe ungültige Werte
                }

                // Speichere nur den primären Namen (ausgeschrieben)
                metrics[metricGroup.primaryName] = value;

                // Markiere alle Aliase als gefunden
                allNames.forEach(name => foundMetrics.add(name));

                break; // Stoppe nach dem ersten erfolgreichen Match
            }
        }
    });

    return metrics;
}

