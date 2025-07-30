import express from 'express';
import cors from 'cors';
import multer from 'multer';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

// Service-Imports
import { initializeLogging } from '../backend/services/log.js';
import { 
  validatePythonCodeWithLLM,
  validatePythonCode,
  executePythonScript,
  extractMetricsFromOutput
} from '../backend/services/code_exec.js';
import { generatePythonScriptWithLLM } from '../backend/services/python_generator.js';
import { 
  initializeDatabase,
  getProject,
  updateProjectTraining,
  updateProjectStatus,
  updateProjectInsights
} from '../backend/services/db.js';
import { setupAPIEndpoints, setTrainingFunctions, setupPredictionEndpoint } from '../backend/services/api_endpoints.js';
import { evaluatePerformanceWithLLM } from '../backend/services/llm_api.js';

// Environment-Variablen laden
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Vercel serverless function handler
export default function handler(req, res) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  // Route to appropriate handler based on path
  const path = req.url;
  
  if (path.startsWith('/api/projects')) {
    // Handle project routes
    handleProjectRoutes(req, res);
  } else if (path.startsWith('/api/upload')) {
    // Handle upload routes
    handleUploadRoutes(req, res);
  } else if (path.startsWith('/api/predict')) {
    // Handle prediction routes
    handlePredictionRoutes(req, res);
  } else {
    res.status(404).json({ error: 'Not found' });
  }
}

// Placeholder functions - you'll need to implement these
function handleProjectRoutes(req, res) {
  res.json({ message: 'Project routes not yet implemented in Vercel' });
}

function handleUploadRoutes(req, res) {
  res.json({ message: 'Upload routes not yet implemented in Vercel' });
}

function handlePredictionRoutes(req, res) {
  res.json({ message: 'Prediction routes not yet implemented in Vercel' });
} 