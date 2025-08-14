import { logRESTAPIRequest } from '../../monitoring/log.js';
import { cleanupOldPredictScripts } from '../../execution/code_exec.js';
import fs from 'fs/promises';
import path from 'path';

export function setupPredictCacheRoutes(app, scriptDir) {
  // Manuelle Bereinigung alter Predict-Skripte
  app.post('/api/predict-cache/cleanup', async (req, res) => {
    try {
      logRESTAPIRequest('cleanup-predict-cache', req.body);
      
      const { maxAgeHours = 168 } = req.body; // Standard: 7 Tage
      
      if (maxAgeHours < 1 || maxAgeHours > 8760) { // 1 Stunde bis 1 Jahr
        return res.status(400).json({
          success: false,
          error: 'maxAgeHours muss zwischen 1 und 8760 liegen'
        });
      }
      
      const cleaned = await cleanupOldPredictScripts(scriptDir, maxAgeHours);
      
      res.json({
        success: true,
        message: `Bereinigung abgeschlossen: ${cleaned} Skripte entfernt`,
        cleaned,
        maxAgeHours
      });
    } catch (error) {
      console.error('Error cleaning predict cache:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to clean predict cache: ' + error.message
      });
    }
  });

  // Status des Predict-Skript Caches
  app.get('/api/predict-cache/status', async (req, res) => {
    try {
      logRESTAPIRequest('get-predict-cache-status', req.query);
      
      const files = await fs.readdir(scriptDir);
      const predictScripts = files.filter(file => file.startsWith('predict_') && file.endsWith('.py'));
      const metadataFiles = files.filter(file => file.startsWith('predict_') && file.endsWith('_metadata.json'));
      const targetEncoderFiles = files.filter(file => file.startsWith('target_encoder_') && file.endsWith('.pkl'));
      
      // Detaillierte Informationen über gecachte Skripte sammeln
      const cachedScripts = [];
      
      for (const metadataFile of metadataFiles) {
        try {
          const metadataPath = path.join(scriptDir, metadataFile);
          const metadataContent = await fs.readFile(metadataPath, 'utf-8');
          const metadata = JSON.parse(metadataContent);
          
          const projectId = metadata.projectId;
          const scriptPath = path.join(scriptDir, `predict_${projectId}.py`);
          const targetEncoderPath = path.join(scriptDir, `target_encoder_${projectId}.pkl`);
          
          let scriptExists = false;
          let scriptSize = 0;
          let targetEncoderExists = false;
          let targetEncoderSize = 0;
          
          try {
            const stats = await fs.stat(scriptPath);
            scriptExists = true;
            scriptSize = stats.size;
          } catch (statError) {
            // Skript-Datei existiert nicht
          }
          
          try {
            const encoderStats = await fs.stat(targetEncoderPath);
            targetEncoderExists = true;
            targetEncoderSize = encoderStats.size;
          } catch (encoderError) {
            // Target-Encoder existiert nicht
          }
          
          cachedScripts.push({
            projectId,
            createdAt: metadata.createdAt,
            lastUsed: metadata.lastUsed,
            scriptExists,
            scriptSize,
            targetEncoderExists,
            targetEncoderSize,
            totalSize: scriptSize + targetEncoderSize,
            projectHash: metadata.projectHash.substring(0, 8) + '...', // Kurze Version
            ageHours: Math.round((Date.now() - new Date(metadata.lastUsed || metadata.createdAt)) / (1000 * 60 * 60))
          });
        } catch (parseError) {
          console.log(`Fehler beim Parsen der Metadata-Datei ${metadataFile}:`, parseError.message);
        }
      }
      
      // Sortiere nach letzter Verwendung (neueste zuerst)
      cachedScripts.sort((a, b) => new Date(b.lastUsed || b.createdAt) - new Date(a.lastUsed || a.createdAt));
      
      const totalScriptSize = cachedScripts.reduce((sum, script) => sum + script.totalSize, 0);
      
      res.json({
        success: true,
        cache: {
          totalScripts: predictScripts.length,
          totalMetadata: metadataFiles.length,
          totalTargetEncoders: targetEncoderFiles.length,
          totalSizeBytes: totalScriptSize,
          totalSizeMB: Math.round(totalScriptSize / (1024 * 1024) * 100) / 100,
          scripts: cachedScripts
        }
      });
    } catch (error) {
      console.error('Error getting predict cache status:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to get predict cache status: ' + error.message
      });
    }
  });

  // Spezifisches Predict-Skript für ein Projekt löschen
  app.delete('/api/predict-cache/project/:projectId', async (req, res) => {
    try {
      logRESTAPIRequest('delete-predict-cache-project', req.params);
      
      const { projectId } = req.params;
      
      if (!projectId) {
        return res.status(400).json({
          success: false,
          error: 'Project ID ist erforderlich'
        });
      }
      
      const scriptPath = path.join(scriptDir, `predict_${projectId}.py`);
      const metadataPath = path.join(scriptDir, `predict_${projectId}_metadata.json`);
      const targetEncoderPath = path.join(scriptDir, `target_encoder_${projectId}.pkl`);
      
      let deletedFiles = 0;
      const deletedItems = [];
      
      // Lösche Python-Skript
      try {
        await fs.unlink(scriptPath);
        deletedFiles++;
        deletedItems.push('Python-Skript');
      } catch (deleteError) {
        // Datei existiert nicht oder Fehler beim Löschen
      }
      
      // Lösche Metadata
      try {
        await fs.unlink(metadataPath);
        deletedFiles++;
        deletedItems.push('Metadata');
      } catch (deleteError) {
        // Datei existiert nicht oder Fehler beim Löschen
      }
      
      // Lösche projekt-spezifischen Target-Encoder
      try {
        await fs.unlink(targetEncoderPath);
        deletedFiles++;
        deletedItems.push('Target-Encoder');
      } catch (deleteError) {
        // Datei existiert nicht oder Fehler beim Löschen
      }
      
      if (deletedFiles === 0) {
        return res.status(404).json({
          success: false,
          error: `Kein gecachtes Predict-Skript für Projekt ${projectId} gefunden`
        });
      }
      
      res.json({
        success: true,
        message: `Predict-Skript für Projekt ${projectId} erfolgreich gelöscht`,
        projectId,
        deletedFiles,
        deletedItems
      });
    } catch (error) {
      console.error('Error deleting predict cache for project:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to delete predict cache: ' + error.message
      });
    }
  });

  // Alle Predict-Skripte löschen (Cache leeren)
  app.delete('/api/predict-cache/all', async (req, res) => {
    try {
      logRESTAPIRequest('clear-predict-cache', req.body);
      
      const files = await fs.readdir(scriptDir);
      const predictFiles = files.filter(file => 
        file.startsWith('predict_') && (file.endsWith('.py') || file.endsWith('_metadata.json'))
      );
      const targetEncoderFiles = files.filter(file => 
        file.startsWith('target_encoder_') && file.endsWith('.pkl')
      );
      
      let deletedFiles = 0;
      const allFilesToDelete = [...predictFiles, ...targetEncoderFiles];
      
      for (const file of allFilesToDelete) {
        try {
          await fs.unlink(path.join(scriptDir, file));
          deletedFiles++;
        } catch (deleteError) {
          console.log(`Fehler beim Löschen der Datei ${file}:`, deleteError.message);
        }
      }
      
      res.json({
        success: true,
        message: `Predict-Cache erfolgreich geleert: ${deletedFiles} Dateien gelöscht`,
        deletedFiles,
        totalFilesFound: allFilesToDelete.length,
        predictFiles: predictFiles.length,
        targetEncoderFiles: targetEncoderFiles.length
      });
    } catch (error) {
      console.error('Error clearing predict cache:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to clear predict cache: ' + error.message
      });
    }
  });
}
