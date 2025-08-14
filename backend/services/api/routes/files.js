import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { logRESTAPIRequest } from '../../monitoring/log.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export function setupFileRoutes(app) {
  // Basis-Pfade für verschiedene Dateitypen
  const basePaths = {
    scripts: path.resolve(__dirname, '../../../scripts'),
    models: path.resolve(__dirname, '../../../models'),
    uploads: path.resolve(__dirname, '../../../uploads')
  };

  // Hilfsfunktion zum sicheren Pfad-Aufbau
  const getSecurePath = (type, filename = '') => {
    const basePath = basePaths[type];
    if (!basePath) {
      throw new Error('Ungültiger Dateityp');
    }
    
    if (filename) {
      // Verhindere Directory Traversal Angriffe
      const normalizedPath = path.normalize(filename).replace(/^(\.\.[\/\\])+/, '');
      return path.join(basePath, normalizedPath);
    }
    
    return basePath;
  };

  // Dateien eines Typs auflisten
  app.get('/api/files/:type', async (req, res) => {
    try {
      logRESTAPIRequest(`get-files-${req.params.type}`, req.params);
      const { type } = req.params;
      
      if (!['scripts', 'models', 'uploads'].includes(type)) {
        return res.status(400).json({ error: 'Ungültiger Dateityp' });
      }

      const dirPath = getSecurePath(type);
      
      try {
        await fs.access(dirPath);
      } catch {
        // Verzeichnis existiert nicht, erstelle es
        await fs.mkdir(dirPath, { recursive: true });
        return res.json([]);
      }

      const files = await fs.readdir(dirPath);
      const fileInfos = [];

      for (const file of files) {
        try {
          const filePath = path.join(dirPath, file);
          const stats = await fs.stat(filePath);
          
          if (stats.isFile()) {
            fileInfos.push({
              name: file,
              size: stats.size,
              lastModified: stats.mtime.toISOString(),
              type: type.slice(0, -1), // 'scripts' -> 'script'
              path: filePath
            });
          }
        } catch (error) {
          console.error(`Fehler beim Lesen der Datei ${file}:`, error);
          // Überspringe fehlerhafte Dateien
        }
      }

      // Sortiere nach Änderungsdatum (neueste zuerst)
      fileInfos.sort((a, b) => new Date(b.lastModified).getTime() - new Date(a.lastModified).getTime());

      res.json(fileInfos);
    } catch (error) {
      console.error('Fehler beim Auflisten der Dateien:', error);
      res.status(500).json({ error: 'Fehler beim Auflisten der Dateien: ' + error.message });
    }
  });

  // Datei löschen
  app.delete('/api/files/:type', async (req, res) => {
    try {
      logRESTAPIRequest(`delete-file-${req.params.type}`, req.body);
      const { type } = req.params;
      const { filePath } = req.body;
      
      if (!['scripts', 'models', 'uploads'].includes(type)) {
        return res.status(400).json({ error: 'Ungültiger Dateityp' });
      }

      if (!filePath) {
        return res.status(400).json({ error: 'Dateipfad ist erforderlich' });
      }

      // Sicherheitsprüfung: Stelle sicher, dass der Pfad im erlaubten Bereich liegt
      const basePath = getSecurePath(type);
      const normalizedFilePath = path.resolve(filePath);
      
      if (!normalizedFilePath.startsWith(basePath)) {
        return res.status(403).json({ error: 'Zugriff auf diese Datei nicht erlaubt' });
      }

      // Prüfe ob Datei existiert
      try {
        await fs.access(normalizedFilePath);
      } catch {
        return res.status(404).json({ error: 'Datei nicht gefunden' });
      }

      // Sicherheitsabfrage für bestimmte wichtige Dateien
      const fileName = path.basename(normalizedFilePath);
      const protectedFiles = [
        '.gitkeep',
        'requirements.txt',
        'package.json',
        'package-lock.json'
      ];

      if (protectedFiles.includes(fileName)) {
        return res.status(403).json({ error: 'Diese Datei kann nicht gelöscht werden (geschützte Systemdatei)' });
      }

      // Lösche die Datei
      await fs.unlink(normalizedFilePath);
      
      console.log(`✅ Datei gelöscht: ${normalizedFilePath}`);
      
      res.json({ 
        success: true, 
        message: `Datei "${fileName}" wurde erfolgreich gelöscht`,
        deletedFile: fileName
      });

    } catch (error) {
      console.error('Fehler beim Löschen der Datei:', error);
      res.status(500).json({ error: 'Fehler beim Löschen der Datei: ' + error.message });
    }
  });

  // Datei-Informationen abrufen
  app.get('/api/files/:type/:filename', async (req, res) => {
    try {
      logRESTAPIRequest(`get-file-info-${req.params.type}`, req.params);
      const { type, filename } = req.params;
      
      if (!['scripts', 'models', 'uploads'].includes(type)) {
        return res.status(400).json({ error: 'Ungültiger Dateityp' });
      }

      const filePath = getSecurePath(type, filename);
      
      try {
        const stats = await fs.stat(filePath);
        
        res.json({
          name: filename,
          size: stats.size,
          lastModified: stats.mtime.toISOString(),
          created: stats.birthtime.toISOString(),
          type: type.slice(0, -1),
          path: filePath,
          readable: stats.isFile(),
          writable: true // Vereinfacht, könnte erweitert werden
        });
      } catch {
        return res.status(404).json({ error: 'Datei nicht gefunden' });
      }

    } catch (error) {
      console.error('Fehler beim Abrufen der Datei-Informationen:', error);
      res.status(500).json({ error: 'Fehler beim Abrufen der Datei-Informationen: ' + error.message });
    }
  });

  // Speicher-Statistiken abrufen
  app.get('/api/files/storage/stats', async (req, res) => {
    try {
      logRESTAPIRequest('get-storage-stats', {});
      
      const stats = {
        scripts: { count: 0, totalSize: 0 },
        models: { count: 0, totalSize: 0 },
        uploads: { count: 0, totalSize: 0 },
        total: { count: 0, totalSize: 0 }
      };

      for (const type of ['scripts', 'models', 'uploads']) {
        try {
          const dirPath = getSecurePath(type);
          await fs.access(dirPath);
          
          const files = await fs.readdir(dirPath);
          
          for (const file of files) {
            try {
              const filePath = path.join(dirPath, file);
              const fileStat = await fs.stat(filePath);
              
              if (fileStat.isFile()) {
                stats[type].count++;
                stats[type].totalSize += fileStat.size;
              }
            } catch {
              // Überspringe fehlerhafte Dateien
            }
          }
        } catch {
          // Verzeichnis existiert nicht
        }
      }

      // Gesamtstatistiken berechnen
      stats.total.count = stats.scripts.count + stats.models.count + stats.uploads.count;
      stats.total.totalSize = stats.scripts.totalSize + stats.models.totalSize + stats.uploads.totalSize;

      res.json(stats);
    } catch (error) {
      console.error('Fehler beim Abrufen der Speicher-Statistiken:', error);
      res.status(500).json({ error: 'Fehler beim Abrufen der Speicher-Statistiken: ' + error.message });
    }
  });
}
