import { logRESTAPIRequest } from '../../monitoring/log.js';
import { scalingMonitor } from '../../monitoring/scaling_monitor.js';

export function setupScalingRoutes(app) {
  // Aktuelle Skalierungs-Metriken abrufen
  app.get('/api/scaling/metrics', async (req, res) => {
    try {
      logRESTAPIRequest('get-scaling-metrics', req.query);
      
      const metrics = scalingMonitor.getScalingMetrics();
      res.json({
        success: true,
        metrics
      });
    } catch (error) {
      console.error('Error getting scaling metrics:', error);
      res.status(500).json({ 
        success: false, 
        error: 'Failed to get scaling metrics: ' + error.message 
      });
    }
  });

  // Detaillierter Skalierungs-Report
  app.get('/api/scaling/report', async (req, res) => {
    try {
      logRESTAPIRequest('get-scaling-report', req.query);
      
      const report = scalingMonitor.getDetailedReport();
      res.json({
        success: true,
        report
      });
    } catch (error) {
      console.error('Error getting scaling report:', error);
      res.status(500).json({ 
        success: false, 
        error: 'Failed to get scaling report: ' + error.message 
      });
    }
  });

  // Skalierungs-Historie für bestimmten Zeitraum
  app.get('/api/scaling/history/:type', async (req, res) => {
    try {
      logRESTAPIRequest('get-scaling-history', { ...req.params, ...req.query });
      
      const { type } = req.params;
      const hours = parseInt(req.query.hours) || 24;
      
      if (!['python', 'llm'].includes(type)) {
        return res.status(400).json({ 
          success: false, 
          error: 'Invalid type. Must be "python" or "llm"' 
        });
      }
      
      const history = scalingMonitor.getScalingHistory(type, hours);
      res.json({
        success: true,
        type,
        hours,
        history
      });
    } catch (error) {
      console.error('Error getting scaling history:', error);
      res.status(500).json({ 
        success: false, 
        error: 'Failed to get scaling history: ' + error.message 
      });
    }
  });

  // Worker-Auslastung analysieren
  app.get('/api/scaling/utilization/:type', async (req, res) => {
    try {
      logRESTAPIRequest('get-utilization-analysis', req.params);
      
      const { type } = req.params;
      
      if (!['python', 'llm'].includes(type)) {
        return res.status(400).json({ 
          success: false, 
          error: 'Invalid type. Must be "python" or "llm"' 
        });
      }
      
      const utilization = scalingMonitor.analyzeUtilization(type);
      const efficiency = scalingMonitor.getScalingEfficiency(type);
      
      res.json({
        success: true,
        type,
        utilization,
        efficiency
      });
    } catch (error) {
      console.error('Error analyzing utilization:', error);
      res.status(500).json({ 
        success: false, 
        error: 'Failed to analyze utilization: ' + error.message 
      });
    }
  });

  // Skalierungs-Konfiguration zur Laufzeit aktualisieren
  app.post('/api/scaling/config', async (req, res) => {
    try {
      logRESTAPIRequest('update-scaling-config', req.body);
      
      const { python, llm, global } = req.body;
      
      if (!python && !llm && !global) {
        return res.status(400).json({ 
          success: false, 
          error: 'At least one configuration section (python, llm, global) is required' 
        });
      }
      
      const newConfig = {};
      if (python) newConfig.python = python;
      if (llm) newConfig.llm = llm;
      if (global) newConfig.global = global;
      
      scalingMonitor.updateConfig(newConfig);
      
      res.json({
        success: true,
        message: 'Scaling configuration updated',
        config: scalingMonitor.getScalingMetrics()
      });
    } catch (error) {
      console.error('Error updating scaling config:', error);
      res.status(500).json({ 
        success: false, 
        error: 'Failed to update scaling config: ' + error.message 
      });
    }
  });

  // Live Worker-Status für Dashboard
  app.get('/api/scaling/status', async (req, res) => {
    try {
      logRESTAPIRequest('get-scaling-status', req.query);
      
      const metrics = scalingMonitor.getScalingMetrics();
      const pythonUtil = scalingMonitor.analyzeUtilization('python');
      const llmUtil = scalingMonitor.analyzeUtilization('llm');
      
      // Vereinfachter Status für Dashboard
      const status = {
        timestamp: new Date().toISOString(),
        python: {
          activeWorkers: metrics.python.activeWorkers,
          busyWorkers: metrics.python.busyWorkers,
          queueLength: metrics.python.queueLength,
          utilization: pythonUtil.utilization,
          recommendation: pythonUtil.recommendation,
          lastScaleEvent: metrics.python.lastScaleUp || metrics.python.lastScaleDown
        },
        llm: {
          activeWorkers: metrics.llm.activeWorkers,
          busyWorkers: metrics.llm.busyWorkers,
          queueLength: metrics.llm.queueLength,
          utilization: llmUtil.utilization,
          recommendation: llmUtil.recommendation,
          lastScaleEvent: metrics.llm.lastScaleUp || metrics.llm.lastScaleDown
        },
        systemHealth: scalingMonitor.getSystemStability()
      };
      
      res.json({
        success: true,
        status
      });
    } catch (error) {
      console.error('Error getting scaling status:', error);
      res.status(500).json({ 
        success: false, 
        error: 'Failed to get scaling status: ' + error.message 
      });
    }
  });
}
