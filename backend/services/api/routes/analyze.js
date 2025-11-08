import path from 'path';
import { logRESTAPIRequest } from '../../monitoring/log.js';
import {
  analyzeCsvFile,
  analyzeJsonFile,
  analyzeExcelFile,
  analyzeTextFile,
  analyzeGenericFile
} from '../../data/file_analysis.js';
import { masClient } from '../../clients/mas_client.js';

export function setupAnalyzeRoutes(app) {
  // Intelligente LLM-Empfehlungen für manipulierte Daten
  app.post('/api/analyze-data', async (req, res) => {
    try {
      logRESTAPIRequest('analyze-data', req.body);
      const { filePath, excludedColumns, excludedFeatures, selectedColumns, userPreferences } = req.body;

      if (!filePath) {
        return res.status(400).json({ error: 'filePath ist erforderlich' });
      }

      const fileExtension = path.extname(filePath).toLowerCase();

      // Datei basierend auf Typ analysieren (ohne zusätzliche LLM-Analyse; wir nutzen die gecachte Summary)
      let analysis;
      if (fileExtension === '.csv') {
        analysis = await analyzeCsvFile(filePath, false);
        analysis.file_type = 'CSV';
      } else if (fileExtension === '.json') {
        analysis = await analyzeJsonFile(filePath, false);
        analysis.file_type = 'JSON';
      } else if (fileExtension === '.xlsx' || fileExtension === '.xls') {
        analysis = await analyzeExcelFile(filePath, false);
        analysis.file_type = 'Excel';
      } else if (fileExtension === '.txt') {
        analysis = await analyzeTextFile(filePath, false);
        analysis.file_type = 'Text';
      } else {
        analysis = await analyzeGenericFile(filePath, fileExtension, false);
        analysis.file_type = fileExtension.substring(1).toUpperCase();
      }

      // Spalten basierend auf Manipulationen anpassen
      let manipulatedAnalysis = { ...analysis };

      if (excludedColumns && excludedColumns.length > 0) {
        manipulatedAnalysis.columns = analysis.columns.filter(col => !excludedColumns.includes(col));
        manipulatedAnalysis.sampleData = analysis.sampleData.map(row =>
          row.filter((_, index) => !excludedColumns.includes(analysis.columns[index]))
        );
      }

      if (selectedColumns && selectedColumns.length > 0) {
        manipulatedAnalysis.columns = selectedColumns;
        manipulatedAnalysis.sampleData = analysis.sampleData.map(row =>
          selectedColumns.map(col => row[analysis.columns.indexOf(col)])
        );
      }

      if (excludedFeatures && excludedFeatures.length > 0) {
        manipulatedAnalysis.columns = analysis.columns.filter(col => !excludedFeatures.includes(col));
      }

      // LLM-basierte Empfehlungen für manipulierte Daten
      const recommendations = await masClient.getLLMRecommendations(
        manipulatedAnalysis,
        filePath,
        selectedColumns,
        excludedFeatures,
        userPreferences
      );

      // Sicherstellen, dass recommendations.features existiert
      if (!recommendations.features || recommendations.features.length === 0) {
        console.warn('LLM gab keine Features zurück, verwende Fallback');
        recommendations.features = manipulatedAnalysis.columns.filter(col => col !== recommendations.targetVariable);
      }

      res.json({
        analysis: manipulatedAnalysis,
        recommendations: recommendations,
        availableFeatures: recommendations.features
      });

    } catch (error) {
      console.error('Fehler bei der Datenanalyse:', error);
      res.status(500).json({ error: 'Fehler bei der Datenanalyse: ' + error.message });
    }
  });
}


