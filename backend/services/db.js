import sqlite3 from 'sqlite3';
import { v4 as uuidv4 } from 'uuid';

// SQLite Datenbank initialisieren
const db = new sqlite3.Database('projects.db');

// Datenbank-Schema initialisieren
export function initializeDatabase() {
  return new Promise((resolve, reject) => {
    db.serialize(() => {
      db.run(`
        CREATE TABLE IF NOT EXISTS projects (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          status TEXT NOT NULL,
          modelType TEXT NOT NULL,
          dataSourceName TEXT,
          targetVariable TEXT,
          features TEXT, -- JSON string
          createdAt TEXT,
          performanceMetrics TEXT, -- JSON string
          performanceInsights TEXT, -- JSON string mit LLM-Evaluation
          pythonCode TEXT,
          originalPythonCode TEXT, -- Original LLM-generierter Code
          modelPath TEXT,
          csvFilePath TEXT,
          algorithm TEXT,
          hyperparameters TEXT, -- JSON string
          llmRecommendations TEXT -- JSON string mit LLM-Empfehlungen
        )
      `, (err) => {
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      });
    });
  });
}

// Hilfsfunktion: Projekt aus DB abrufen
export function getProject(id) {
  return new Promise((resolve, reject) => {
    db.get('SELECT * FROM projects WHERE id = ?', [id], (err, row) => {
      if (err) {
        reject(err);
      } else if (row) {
        // JSON Strings zurück zu Objekten parsen
        row.features = JSON.parse(row.features || '[]');
        row.performanceMetrics = row.performanceMetrics ? JSON.parse(row.performanceMetrics) : null;
        row.llmRecommendations = row.llmRecommendations ? JSON.parse(row.llmRecommendations) : null;
        row.performanceInsights = row.performanceInsights ? JSON.parse(row.performanceInsights) : null;
        row.hyperparameters = row.hyperparameters ? JSON.parse(row.hyperparameters) : null;
        resolve(row);
      } else {
        resolve(null);
      }
    });
  });
}

// Alle Projekte abrufen
export function getAllProjects() {
  return new Promise((resolve, reject) => {
    db.all('SELECT * FROM projects ORDER BY createdAt DESC', (err, rows) => {
      if (err) {
        reject(err);
      } else {
        // JSON Strings parsen
        const projects = rows.map(row => ({
          ...row,
          features: JSON.parse(row.features || '[]'),
          performanceMetrics: row.performanceMetrics ? JSON.parse(row.performanceMetrics) : null,
          llmRecommendations: row.llmRecommendations ? JSON.parse(row.llmRecommendations) : null,
          performanceInsights: row.performanceInsights ? JSON.parse(row.performanceInsights) : null,
          hyperparameters: row.hyperparameters ? JSON.parse(row.hyperparameters) : null
        }));
        
        resolve(projects);
      }
    });
  });
}

// Neues Projekt erstellen
export function createProject(projectData) {
  return new Promise((resolve, reject) => {
    const { name, modelType, dataSourceName, targetVariable, features, csvFilePath, algorithm, hyperparameters } = projectData;
    const id = uuidv4();
    const createdAt = new Date().toISOString();
    const status = 'Training';
    
    const stmt = db.prepare(`
      INSERT INTO projects (id, name, status, modelType, dataSourceName, targetVariable, features, createdAt, csvFilePath, algorithm, hyperparameters)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    
    // Hyperparameter korrekt konvertieren (numerische Werte als Zahlen, nicht als Strings)
    const convertedHyperparameters = {};
    if (hyperparameters && typeof hyperparameters === 'object') {
      for (const [key, value] of Object.entries(hyperparameters)) {
        if (typeof value === 'string' && !isNaN(Number(value)) && value.trim() !== '') {
          convertedHyperparameters[key] = Number(value);
        } else {
          convertedHyperparameters[key] = value;
        }
      }
    }
    
    stmt.run([
      id, name, status, modelType, dataSourceName, targetVariable, 
      JSON.stringify(features), createdAt, csvFilePath || null, 
      algorithm || 'RandomForest', JSON.stringify(convertedHyperparameters || {})
    ], function(err) {
      if (err) {
        reject(err);
      } else {
        resolve({
          id, name, status, modelType, dataSourceName, targetVariable,
          features, createdAt, performanceMetrics: null, pythonCode: null, modelPath: null, 
          csvFilePath, algorithm: algorithm || 'RandomForest', hyperparameters: convertedHyperparameters || {}
        });
      }
    });
    
    stmt.finalize();
  });
}

// Intelligentes Projekt erstellen (mit LLM-Empfehlungen)
export function createSmartProject(projectData) {
  return new Promise((resolve, reject) => {
    const { name, csvFilePath, recommendations } = projectData;
    const id = uuidv4();
    const createdAt = new Date().toISOString();
    const status = 'Training';
    
    const stmt = db.prepare(`
      INSERT INTO projects (id, name, status, modelType, dataSourceName, targetVariable, features, createdAt, csvFilePath, algorithm, hyperparameters, llmRecommendations)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    
    // Hyperparameter korrekt konvertieren (numerische Werte als Zahlen, nicht als Strings)
    const convertedHyperparameters = {};
    if (recommendations.hyperparameters && typeof recommendations.hyperparameters === 'object') {
      for (const [key, value] of Object.entries(recommendations.hyperparameters)) {
        if (typeof value === 'string' && !isNaN(Number(value)) && value.trim() !== '') {
          convertedHyperparameters[key] = Number(value);
        } else {
          convertedHyperparameters[key] = value;
        }
      }
    }
    
    stmt.run([
      id, name, status, recommendations.modelType, recommendations.dataSourceName, 
      recommendations.targetVariable, JSON.stringify(recommendations.features), 
      createdAt, csvFilePath, recommendations.algorithm, 
      JSON.stringify(convertedHyperparameters), JSON.stringify(recommendations)
    ], function(err) {
      if (err) {
        reject(err);
      } else {
        resolve({
          id, name, status, 
          modelType: recommendations.modelType,
          dataSourceName: recommendations.dataSourceName,
          targetVariable: recommendations.targetVariable,
          features: recommendations.features,
          createdAt, 
          performanceMetrics: null, 
          pythonCode: null, 
          modelPath: null, 
          csvFilePath, 
          algorithm: recommendations.algorithm, 
          hyperparameters: convertedHyperparameters,
          recommendations: recommendations
        });
      }
    });
    
    stmt.finalize();
  });
}

// Projekt-Status und Metriken aktualisieren
export function updateProjectTraining(projectId, updateData) {
  return new Promise((resolve, reject) => {
    const { status, performanceMetrics, pythonCode, originalPythonCode, modelPath, hyperparameters } = updateData;
    
    const stmt = db.prepare(`
      UPDATE projects 
      SET status = ?, performanceMetrics = ?, pythonCode = ?, originalPythonCode = ?, modelPath = ?, hyperparameters = ?
      WHERE id = ?
    `);
    
    stmt.run([
      status,
      performanceMetrics ? JSON.stringify(performanceMetrics) : null,
      pythonCode,
      originalPythonCode,
      modelPath,
      hyperparameters ? JSON.stringify(hyperparameters) : null,
      projectId
    ], function(err) {
      if (err) {
        reject(err);
      } else {
        resolve({ changes: this.changes, lastID: this.lastID });
      }
    });
    
    stmt.finalize();
  });
}

// Projekt-Code aktualisieren
export function updateProjectCode(projectId, pythonCode) {
  return new Promise((resolve, reject) => {
    db.run('UPDATE projects SET pythonCode = ? WHERE id = ?', [pythonCode, projectId], function(err) {
      if (err) {
        reject(err);
      } else {
        resolve({ changes: this.changes, lastID: this.lastID });
      }
    });
  });
}

// Hyperparameter eines Projekts aktualisieren
export function updateProjectHyperparameters(projectId, hyperparameters) {
  return new Promise((resolve, reject) => {
    db.run('UPDATE projects SET hyperparameters = ? WHERE id = ?', [JSON.stringify(hyperparameters), projectId], function(err) {
      if (err) {
        reject(err);
      } else {
        resolve({ changes: this.changes, lastID: this.lastID });
      }
    });
  });
}

// Projekt-Status aktualisieren
export function updateProjectStatus(projectId, status) {
  return new Promise((resolve, reject) => {
    db.run('UPDATE projects SET status = ? WHERE id = ?', [status, projectId], function(err) {
      if (err) {
        reject(err);
      } else {
        resolve({ changes: this.changes, lastID: this.lastID });
      }
    });
  });
}

// Performance-Insights aktualisieren
export function updateProjectInsights(projectId, performanceInsights) {
  return new Promise((resolve, reject) => {
    db.run(
      'UPDATE projects SET performanceInsights = ? WHERE id = ?',
      [JSON.stringify(performanceInsights), projectId],
      function(err) {
        if (err) {
          reject(err);
        } else {
          resolve({ changes: this.changes, lastID: this.lastID });
        }
      }
    );
  });
}

// Projekt löschen
export function deleteProject(projectId) {
  // Hole den Pfad zum Model (falls vorhanden) und zum Script und lösche sie
  const project = getProject(projectId);
  const modelPath = project.modelPath;
  const scriptPath = project.scriptPath;
  if (modelPath) {
    fs.unlink(modelPath, (err) => {
      if (err) {
          console.error('Fehler beim Löschen des Models:', err);
        }
    });
  }
  if (scriptPath) {
    fs.unlink(scriptPath, (err) => {
      if (err) {
        console.error('Fehler beim Löschen des Scripts:', err);
      }
    });
  }

  return new Promise((resolve, reject) => {
    db.run('DELETE FROM projects WHERE id = ?', [projectId], function(err) {
      if (err) {
        reject(err);
      } else {
        resolve({ changes: this.changes, lastID: this.lastID });
      }
    });
  });
}

// Datenbank-Verbindung schließen
export function closeDatabase() {
  return new Promise((resolve, reject) => {
    db.close((err) => {
      if (err) {
        reject(err);
      } else {
        resolve();
      }
    });
  });
}

// Datenbank-Instanz exportieren (für erweiterte Operationen)
export { db };
