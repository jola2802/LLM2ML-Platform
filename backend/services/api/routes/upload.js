import path from 'path';
import { logRESTAPIRequest } from '../../monitoring/log.js';
import { 
  analyzeCsvFile, 
  analyzeJsonFile, 
  analyzeExcelFile, 
  analyzeTextFile, 
  analyzeGenericFile
} from '../../data/file_analysis.js';

export function setupUploadRoutes(app, upload) {
  // Datei hochladen und Basis-Analyse (ohne LLM)
  app.post('/api/upload', upload.single('file'), async (req, res) => {
    try {
      logRESTAPIRequest('upload', req.file);
      if (!req.file) {
        return res.status(400).json({ error: 'Keine Datei hochgeladen' });
      }

      const filePath = req.file.path;
      const savedName = path.basename(filePath);
      const fileExtension = path.extname(savedName).toLowerCase();

      console.log(`Datei hochgeladen: ${savedName} (${fileExtension}) -> ${filePath}`);

      // Datei basierend auf Typ analysieren (ohne zusätzliche LLM-Analyse)
      let analysis;
      if (fileExtension === '.csv') {
        analysis = await analyzeCsvFile(filePath, false);
      } else if (fileExtension === '.json') {
        analysis = await analyzeJsonFile(filePath, false);
      } else if (fileExtension === '.xlsx' || fileExtension === '.xls') {
        analysis = await analyzeExcelFile(filePath, false);
      } else if (fileExtension === '.txt') {
        analysis = await analyzeTextFile(filePath, false);
      } else {
        analysis = await analyzeGenericFile(filePath, fileExtension, false);
      }

      // Nur Basis-Analyse zurückgeben (ohne LLM-Empfehlungen)
      res.json({
        fileName: savedName,
        filePath: filePath,
        fileType: fileExtension,
        columns: analysis?.columns || [],
        rowCount: analysis?.rowCount ?? 0,
        dataTypes: analysis?.dataTypes || {},
        sampleData: analysis?.sampleData || [],
        llmAnalysis: analysis?.llm_analysis || null
      });
    } catch (error) {
      console.error('Fehler beim Datei-Upload:', error);
      res.status(500).json({ error: 'Fehler beim Analysieren der Datei: ' + error.message });
    }
  });
}


