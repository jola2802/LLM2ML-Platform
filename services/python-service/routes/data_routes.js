import { performDataExploration, analyzeDataForLLM, getCachedDataAnalysis, clearAnalysisCache, getAnalysisCacheStatus } from '../services/data/data_exploration.js';
import fs from 'fs/promises';
import path from 'path';

export function setupDataRoutes(app, venvDir, uploadsDir) {
    // Data Exploration - Führt performDataExploration aus
    app.post('/api/data/explore', async (req, res) => {
        try {
            const { filePath } = req.body;

            if (!filePath) {
                return res.status(400).json({ error: 'filePath ist erforderlich' });
            }

            // Prüfe ob Datei existiert
            try {
                await fs.access(filePath);
            } catch (accessError) {
                return res.status(404).json({ error: `Datei nicht gefunden: ${filePath}` });
            }

            console.log(`Data Exploration gestartet für: ${filePath}`);

            // Führe Datenexploration durch
            const explorationResult = await performDataExploration(filePath);

            res.json({
                success: true,
                exploration: explorationResult,
                filePath
            });

        } catch (error) {
            console.error('Fehler bei Data Exploration:', error);
            res.status(500).json({
                error: `Fehler bei Data Exploration: ${error.message}`,
                filePath: req.body.filePath
            });
        }
    });

    // Data Analysis - Führt analyzeDataForLLM aus (mit LLM-Zusammenfassung)
    app.post('/api/data/analyze', async (req, res) => {
        try {
            const { filePath, forceRefresh = false } = req.body;

            if (!filePath) {
                return res.status(400).json({ error: 'filePath ist erforderlich' });
            }

            // Prüfe ob Datei existiert
            try {
                await fs.access(filePath);
            } catch (accessError) {
                return res.status(404).json({ error: `Datei nicht gefunden: ${filePath}` });
            }

            console.log(`Data Analysis gestartet für: ${filePath} (forceRefresh: ${forceRefresh})`);

            // Führe Analyse mit Cache durch
            const analysisResult = await getCachedDataAnalysis(filePath, forceRefresh);

            res.json(analysisResult);

        } catch (error) {
            console.error('Fehler bei Data Analysis:', error);
            res.status(500).json({
                error: `Fehler bei Data Analysis: ${error.message}`,
                filePath: req.body.filePath
            });
        }
    });

    // Cache-Status
    app.get('/api/data/cache/status', (req, res) => {
        try {
            const status = getAnalysisCacheStatus();
            res.json(status);
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });

    // Cache leeren
    app.post('/api/data/cache/clear', (req, res) => {
        try {
            clearAnalysisCache();
            res.json({ success: true, message: 'Cache geleert' });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });
}

