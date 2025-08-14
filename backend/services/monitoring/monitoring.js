import { getProject } from '../database/db.js';
import { getCachedDataAnalysis } from '../data/data_exploration.js';

// In-Memory Monitoring Storage (pro Projekt)
// Struktur: { [projectId]: { baseline, live, alerts } }
const monitoringStore = new Map();

function getOrInitProjectState(projectId) {
  if (!monitoringStore.has(projectId)) {
    monitoringStore.set(projectId, {
      baseline: null,
      live: {
        count: 0,
        recent: [], // letzte 500 Einträge: { prediction, truth, timestamp }
        rollingMetrics: {},
        featureStats: {}
      },
      alerts: []
    });
  }
  return monitoringStore.get(projectId);
}

// Einfache numerische Feature-Statistik (Mittelwert, Std)
function updateFeatureStats(stats, features) {
  Object.entries(features || {}).forEach(([key, val]) => {
    const num = typeof val === 'number' ? val : Number(val);
    if (!isFinite(num)) return;
    const s = stats[key] || { n: 0, mean: 0, m2: 0 };
    const n1 = s.n + 1;
    const delta = num - s.mean;
    const mean = s.mean + delta / n1;
    const m2 = s.m2 + delta * (num - mean);
    stats[key] = { n: n1, mean, m2 };
  });
}

function finalizeStd(stats) {
  const result = {};
  Object.entries(stats).forEach(([k, v]) => {
    const variance = v.n > 1 ? v.m2 / (v.n - 1) : 0;
    result[k] = { mean: v.mean, std: Math.sqrt(variance), n: v.n };
  });
  return result;
}

// Population Stability Index (vereinfachte Approximation über Mittelwert/Std-Vergleich)
function computeSimpleDriftScore(baselineStats, liveStats) {
  const keys = Object.keys(baselineStats || {});
  if (keys.length === 0) return 0;
  let total = 0;
  let count = 0;
  keys.forEach((k) => {
    const b = baselineStats[k];
    const l = liveStats[k];
    if (!b || !l || b.std === 0) return;
    const z = Math.abs((l.mean - b.mean) / (b.std || 1e-9));
    total += Math.min(z, 10); // clamp
    count += 1;
  });
  return count > 0 ? total / count : 0;
}

function addAlert(state, message, severity = 'warning', code = 'DRIFT_DETECTED') {
  const alert = { id: Date.now(), timestamp: new Date().toISOString(), severity, message, code };
  state.alerts.push(alert);
  if (state.alerts.length > 1000) state.alerts.shift();
  return alert;
}

export async function initializeMonitoringBaseline(projectId) {
  const state = getOrInitProjectState(projectId);
  if (state.baseline) return state.baseline;

  const project = await getProject(projectId);
  if (!project) throw new Error('Projekt nicht gefunden');
  if (!project.csvFilePath) throw new Error('Kein Trainingsdatensatzpfad vorhanden');

  // Hole kompakte Analyse für Feature-Baseline
  const analysis = await getCachedDataAnalysis(project.csvFilePath, false);
  const pareto = analysis?.exploration?.pareto_analysis || analysis?.pareto_analysis;

  const baseline = {
    performanceMetrics: project.performanceMetrics || {},
    featureStats: {}
  };

  // Aus pareto.key_columns Mittelwert/Std übernehmen falls vorhanden
  const keyCols = pareto?.key_columns || {};
  Object.entries(keyCols).forEach(([col, info]) => {
    if (typeof info.mean === 'number' && typeof info.std === 'number') {
      baseline.featureStats[col] = { mean: info.mean, std: info.std, n: info.n || 0 };
    }
  });

  state.baseline = baseline;
  return baseline;
}

export async function logPredictionEvent(projectId, payload) {
  const state = getOrInitProjectState(projectId);
  await initializeMonitoringBaseline(projectId).catch(() => {});

  const { features = {}, prediction = null, truth = null, timestamp = new Date().toISOString() } = payload || {};
  state.live.count += 1;
  state.live.recent.push({ features, prediction, truth, timestamp });
  if (state.live.recent.length > 500) state.live.recent.shift();

  // Feature-Stats updaten (inkrementell)
  updateFeatureStats(state.live.featureStats, features);

  // Rolling Metrics, wenn Truth vorhanden und Klassifikation (heuristisch)
  if (truth !== null && truth !== undefined && prediction !== null && prediction !== undefined) {
    const isNumericTruth = typeof truth === 'number' && isFinite(truth);
    const isNumericPred = typeof prediction === 'number' && isFinite(prediction);
    if (isNumericTruth && isNumericPred) {
      // Regression: RMSE über Fenster
      const pairs = state.live.recent.filter(r => typeof r.truth === 'number' && typeof r.prediction === 'number');
      const mse = pairs.reduce((acc, r) => acc + Math.pow((r.prediction - r.truth), 2), 0) / Math.max(pairs.length, 1);
      const rmse = Math.sqrt(mse);
      state.live.rollingMetrics.rmse = Number.isFinite(rmse) ? Number(rmse.toFixed(4)) : null;
    } else {
      // Klassifikation: Accuracy über Fenster
      const pairs = state.live.recent.filter(r => r.truth !== null && r.truth !== undefined && r.prediction !== null && r.prediction !== undefined);
      const correct = pairs.reduce((acc, r) => acc + (String(r.prediction) === String(r.truth) ? 1 : 0), 0);
      const acc = pairs.length > 0 ? correct / pairs.length : 0;
      state.live.rollingMetrics.accuracy = Number(acc.toFixed(4));
    }
  }

  // Drift-Bewertung (einfacher z-score Mittelwertvergleich)
  const liveStats = finalizeStd(state.live.featureStats);
  const baselineStats = state.baseline?.featureStats || {};
  const driftScore = computeSimpleDriftScore(baselineStats, liveStats);
  state.live.rollingMetrics.driftScore = Number(driftScore.toFixed(3));

  // Alert auslösen, wenn Schwelle überschritten
  if (driftScore >= 3) {
    addAlert(state, `Möglicher Daten-Drift erkannt (Score=${state.live.rollingMetrics.driftScore})`, 'warning', 'DRIFT_DETECTED');
  }

  return {
    success: true,
    rollingMetrics: state.live.rollingMetrics,
    alerts: state.alerts.slice(-10)
  };
}

export function getMonitoringStatus(projectId) {
  const state = getOrInitProjectState(projectId);
  return {
    success: true,
    baseline: state.baseline,
    rollingMetrics: state.live.rollingMetrics,
    recentCount: state.live.count,
    alerts: state.alerts.slice(-50)
  };
}

export function clearMonitoring(projectId) {
  monitoringStore.delete(projectId);
  return { success: true };
}


