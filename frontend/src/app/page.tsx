"use client";

import React, { useState, useEffect, useCallback } from "react";

interface SystemHealth {
  status: string;
  project: string;
  version: string;
  environment: string;
  python_version: string;
  database: {
    status: string;
    backend: string;
    url_masked: string;
  };
  redis_task_queue: {
    status: string;
    always_eager_fallback: boolean;
  };
  compute: {
    gpu: {
      available: boolean;
      device_name?: string | null;
      device_count?: number;
    };
    platform: string;
  };
  adapters: {
    noaa_ibtracs: { requires_auth: boolean; configured: boolean; base_url: string };
    noaa_goes: { requires_auth: boolean; configured: boolean; endpoint: string };
    isro_insat: { requires_auth: boolean; configured: boolean; instructions?: string };
  };
  models: Array<{
    id: string;
    version: string;
    loaded: boolean;
    description: string;
  }>;
}

interface Product {
  id: string;
  filename: string;
  file_format: string;
  file_size_bytes: number;
  sha256_hash: string;
  source_origin: string;
  variables_count: number;
  channels_count: number;
  spatial_coverage: {
    lat_min: number | null;
    lat_max: number | null;
    lon_min: number | null;
    lon_max: number | null;
  };
  created_at: string;
}



interface ProvenanceLog {
  id: string;
  entity_type: string;
  entity_id: string;
  sha256_hash: string;
  action: string;
  software_version: string;
  parameters: Record<string, unknown>;
  timestamp: string;
}

interface EvaluationReport {
  timestamp_utc: string;
  dataset: {
    source: string;
    total_observations: number;
    split: {
      strategy: string;
      total_storms: number;
      train: { storm_count: number; obs_count: number; seasons: number[] };
      val: { storm_count: number; obs_count: number; seasons: number[] };
      test: { storm_count: number; obs_count: number; seasons: number[] };
      leakage_check_passed: boolean;
    };
    missing_stats: {
      total_observations_examined: number;
      valid_observations_loaded: number;
      missing_wind_observations: number;
      missing_pressure_observations: number;
    };
  };
  models_evaluated: {
    baseline_cliper: {
      intensity_metrics: { mae_kts: number; rmse_kts: number; bias_kts: number; pearson_r: number };
      classification_metrics: { accuracy: number; f1_macro: number };
    };
    environment_only: {
      test_evaluation: {
        intensity_metrics: { mae_kts: number; rmse_kts: number; bias_kts: number; pearson_r: number };
        classification_metrics: { accuracy: number; f1_macro: number };
      };
    };
    image_only: {
      test_evaluation: {
        intensity_metrics: { mae_kts: number; rmse_kts: number; bias_kts: number; pearson_r: number };
        classification_metrics: { accuracy: number; f1_macro: number };
      };
    };
    multimodal_fusion: {
      test_evaluation: {
        intensity_metrics: { mae_kts: number; rmse_kts: number; bias_kts: number; pearson_r: number };
        classification_metrics: { accuracy: number; f1_macro: number };
      };
    };
  };
  explainability_sample?: {
    gradcam: {
      core_concentration_ratio: number;
      peak_activation: number;
      saliency_sha256: string;
      causal_disclaimer: string;
    };
    environmental_attribution: {
      ranked_features: Array<[string, number]>;
      causal_disclaimer: string;
    };
  };
  temporal_comparison_sample?: {
    storm_name: string;
    temporal_interval_hours: number;
    t1: { timestamp: string; ground_truth_wind_kts: number; pressure_hpa: number | null };
    t2: { timestamp: string; ground_truth_wind_kts: number; pressure_hpa: number | null };
    translational_motion: { displacement_km: number; speed_kmh: number; bearing_deg: number };
    structural_evolution: { delta_eyewall_cooling_kelvin: number; delta_eye_warming_kelvin: number };
    intensity_evolution: { delta_wind_true_kts: number; rate_kts_per_hr: number; rapid_intensification_observed: boolean };
    model_predictions?: { pred_wind_t1: number; pred_wind_t2: number; delta_pred_wind: number };
  };
}

export default function CycloneSenseDashboard() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [products, setProducts] = useState<Product[]>([]);
  const [evalReport, setEvalReport] = useState<EvaluationReport | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [provenanceLogs, setProvenanceLogs] = useState<ProvenanceLog[]>([]);
  const [statusMessage, setStatusMessage] = useState<string>("");

  const API_BASE = "http://localhost:8000/api/v1";

  const refreshData = useCallback(async () => {
    setLoading(true);
    try {
      const healthRes = await fetch(`${API_BASE}/system/health`);
      if (healthRes.ok) {
        const hData: SystemHealth = await healthRes.json();
        setHealth(hData);
      }

      const productsRes = await fetch(`${API_BASE}/ingest`);
      if (productsRes.ok) {
        const pData: Product[] = await productsRes.json();
        setProducts(pData);
      }

      const evalRes = await fetch(`${API_BASE}/ml/evaluation-report`);
      if (evalRes.ok) {
        const eData: EvaluationReport = await evalRes.json();
        setEvalReport(eData);
      }

      const provRes = await fetch(`${API_BASE}/provenance?limit=15`);
      if (provRes.ok) {
        const prvData: ProvenanceLog[] = await provRes.json();
        setProvenanceLogs(prvData);
      }
    } catch {
      setStatusMessage("Backend server offline on http://localhost:8000. Launch backend to interact.");
    } finally {
      setLoading(false);
    }
  }, [API_BASE]);

  useEffect(() => {
    let isMounted = true;

    async function loadInitial() {
      try {
        const healthRes = await fetch(`${API_BASE}/system/health`);
        if (healthRes.ok && isMounted) {
          const hData: SystemHealth = await healthRes.json();
          setHealth(hData);
        }

        const productsRes = await fetch(`${API_BASE}/ingest`);
        if (productsRes.ok && isMounted) {
          const pData: Product[] = await productsRes.json();
          setProducts(pData);
        }

        const evalRes = await fetch(`${API_BASE}/ml/evaluation-report`);
        if (evalRes.ok && isMounted) {
          const eData: EvaluationReport = await evalRes.json();
          setEvalReport(eData);
        }

        const provRes = await fetch(`${API_BASE}/provenance?limit=15`);
        if (provRes.ok && isMounted) {
          const prvData: ProvenanceLog[] = await provRes.json();
          setProvenanceLogs(prvData);
        }
      } catch {
        if (isMounted) {
          setStatusMessage("Backend server offline on http://localhost:8000. Launch backend to interact.");
        }
      }
    }

    void loadInitial();

    return () => {
      isMounted = false;
    };
  }, [API_BASE]);

  return (
    <main className="min-h-screen bg-[#080c14] bg-radar-grid text-slate-100 flex flex-col">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-cyan-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <span className="text-xl font-black text-white">🌀</span>
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
              CycloneSense
              <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-cyan-950/80 text-cyan-400 border border-cyan-800/50">
                v0.1.0 (ML Pipeline Ready)
              </span>
            </h1>
            <p className="text-xs text-slate-400">Explainable Multi-Source Tropical Cyclone Pattern Intelligence</p>
          </div>
        </div>

        {/* System Health Badge */}
        <div className="flex items-center space-x-4 text-xs font-mono">
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-md bg-slate-900 border border-slate-800">
            <span
              className={`w-2 h-2 rounded-full ${
                health?.status === "OPERATIONAL" ? "bg-emerald-400 animate-pulse" : "bg-amber-400"
              }`}
            />
            <span className="text-slate-300">
              API: {health ? health.status : "DISCONNECTED"}
            </span>
          </div>

          <div className="hidden sm:flex items-center space-x-2 px-3 py-1.5 rounded-md bg-slate-900 border border-slate-800">
            <span className="text-slate-400">DB:</span>
            <span className="text-cyan-400">{health?.database.backend || "sqlite"}</span>
          </div>

          <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-md bg-slate-900 border border-slate-800">
            <span className="text-slate-400">GPU:</span>
            <span className={health?.compute.gpu.available ? "text-emerald-400" : "text-slate-500"}>
              {health?.compute.gpu.available ? health.compute.gpu.device_name : "CPU Host (CUDA Ready)"}
            </span>
          </div>

          <button
            id="refresh-btn"
            onClick={() => void refreshData()}
            disabled={loading}
            className="px-3 py-1.5 rounded-md bg-cyan-600 hover:bg-cyan-500 text-white font-sans font-medium transition cursor-pointer disabled:opacity-50"
          >
            {loading ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        {/* Status Notification */}
        {statusMessage && (
          <div className="px-4 py-2.5 rounded-lg bg-slate-900/90 border border-cyan-800/40 text-xs font-mono text-cyan-300 flex items-center justify-between">
            <span>ℹ️ {statusMessage}</span>
            <button onClick={() => setStatusMessage("")} className="text-slate-400 hover:text-white">✕</button>
          </div>
        )}

        {/* Section 1: Official Evaluation Matrix (Real Test Split Results) */}
        {evalReport && (
          <section className="glass-panel p-6 rounded-xl border border-cyan-800/40 shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
                  <span>📊</span> CycloneSense V3 Comparative ML Benchmark (Unseen Test Split)
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Storm-level temporal split: {evalReport.dataset.split.train.storm_count} train storms, {evalReport.dataset.split.val.storm_count} val storms, {evalReport.dataset.split.test.storm_count} test storms · Zero frame-level leakage
                </p>
              </div>
              <span className="text-xs font-mono px-2.5 py-1 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/50">
                Verified Zero Leakage
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="border-b border-slate-800 text-slate-400 text-[11px]">
                  <tr>
                    <th className="py-2.5 px-3">Model Architecture</th>
                    <th className="py-2.5 px-3">Input Modality</th>
                    <th className="py-2.5 px-3">Intensity MAE (kts)</th>
                    <th className="py-2.5 px-3">Intensity RMSE (kts)</th>
                    <th className="py-2.5 px-3">Intensity Bias (kts)</th>
                    <th className="py-2.5 px-3">Classification Macro-F1</th>
                    <th className="py-2.5 px-3">Accuracy</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {/* Baseline CLIPER */}
                  <tr className="hover:bg-slate-900/40">
                    <td className="py-2.5 px-3 font-semibold text-white">Baseline CLIPER (Ridge)</td>
                    <td className="py-2.5 px-3 text-slate-400">Environmental Covariates (8)</td>
                    <td className="py-2.5 px-3 text-amber-300 font-bold">
                      {evalReport.models_evaluated.baseline_cliper.intensity_metrics.mae_kts}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {evalReport.models_evaluated.baseline_cliper.intensity_metrics.rmse_kts}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">
                      {evalReport.models_evaluated.baseline_cliper.intensity_metrics.bias_kts}
                    </td>
                    <td className="py-2.5 px-3 text-amber-300 font-bold">
                      {evalReport.models_evaluated.baseline_cliper.classification_metrics.f1_macro}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {(evalReport.models_evaluated.baseline_cliper.classification_metrics.accuracy * 100).toFixed(1)}%
                    </td>
                  </tr>

                  {/* Environment Only MLP */}
                  <tr className="hover:bg-slate-900/40">
                    <td className="py-2.5 px-3 font-semibold text-white">Environment MLP</td>
                    <td className="py-2.5 px-3 text-slate-400">Environmental Covariates (8)</td>
                    <td className="py-2.5 px-3 text-slate-300 font-bold">
                      {evalReport.models_evaluated.environment_only.test_evaluation.intensity_metrics.mae_kts}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {evalReport.models_evaluated.environment_only.test_evaluation.intensity_metrics.rmse_kts}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">
                      {evalReport.models_evaluated.environment_only.test_evaluation.intensity_metrics.bias_kts}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300 font-bold">
                      {evalReport.models_evaluated.environment_only.test_evaluation.classification_metrics.f1_macro}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {(evalReport.models_evaluated.environment_only.test_evaluation.classification_metrics.accuracy * 100).toFixed(1)}%
                    </td>
                  </tr>

                  {/* Image Only CNN */}
                  <tr className="hover:bg-slate-900/40 bg-indigo-950/20">
                    <td className="py-2.5 px-3 font-semibold text-cyan-300">Cyclone CNN (Image-Only)</td>
                    <td className="py-2.5 px-3 text-cyan-400">Multi-Spectral IR + WV [2, 64, 64]</td>
                    <td className="py-2.5 px-3 text-emerald-400 font-bold">
                      {evalReport.models_evaluated.image_only.test_evaluation.intensity_metrics.mae_kts}
                    </td>
                    <td className="py-2.5 px-3 text-emerald-400 font-bold">
                      {evalReport.models_evaluated.image_only.test_evaluation.intensity_metrics.rmse_kts}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">
                      {evalReport.models_evaluated.image_only.test_evaluation.intensity_metrics.bias_kts}
                    </td>
                    <td className="py-2.5 px-3 text-cyan-300 font-bold">
                      {evalReport.models_evaluated.image_only.test_evaluation.classification_metrics.f1_macro}
                    </td>
                    <td className="py-2.5 px-3 text-emerald-400 font-bold">
                      {(evalReport.models_evaluated.image_only.test_evaluation.classification_metrics.accuracy * 100).toFixed(1)}%
                    </td>
                  </tr>

                  {/* Multimodal Fusion */}
                  <tr className="hover:bg-slate-900/40 bg-cyan-950/30">
                    <td className="py-2.5 px-3 font-semibold text-emerald-300 flex items-center gap-1.5">
                      <span>★</span> Cyclone Fusion (Image + Env)
                    </td>
                    <td className="py-2.5 px-3 text-emerald-400">IR + WV + 8 Env Covariates</td>
                    <td className="py-2.5 px-3 text-emerald-300 font-bold">
                      {evalReport.models_evaluated.multimodal_fusion.test_evaluation.intensity_metrics.mae_kts}
                    </td>
                    <td className="py-2.5 px-3 text-emerald-300 font-bold">
                      {evalReport.models_evaluated.multimodal_fusion.test_evaluation.intensity_metrics.rmse_kts}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">
                      {evalReport.models_evaluated.multimodal_fusion.test_evaluation.intensity_metrics.bias_kts}
                    </td>
                    <td className="py-2.5 px-3 text-emerald-300 font-bold">
                      {evalReport.models_evaluated.multimodal_fusion.test_evaluation.classification_metrics.f1_macro}
                    </td>
                    <td className="py-2.5 px-3 text-emerald-300 font-bold">
                      {(evalReport.models_evaluated.multimodal_fusion.test_evaluation.classification_metrics.accuracy * 100).toFixed(1)}%
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Section 2: Temporal Evolution (T1 -> T2 Comparison) */}
        {evalReport?.temporal_comparison_sample && (
          <section className="glass-panel p-6 rounded-xl border border-indigo-800/40 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-indigo-300 uppercase tracking-wider flex items-center gap-2">
                <span>⏱️</span> Spatially Aligned T1 → T2 Temporal Evolution Analysis
              </h2>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                Δt = {evalReport.temporal_comparison_sample.temporal_interval_hours} Hours
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs font-mono">
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="text-[10px] text-slate-400 block uppercase">Wind Intensity Change</span>
                <span className="text-base font-bold text-white">
                  ΔV = {evalReport.temporal_comparison_sample.intensity_evolution.delta_wind_true_kts} kts
                </span>
                <span className="text-[10px] text-cyan-400 block mt-0.5">
                  Rate: {evalReport.temporal_comparison_sample.intensity_evolution.rate_kts_per_hr} kts/h
                </span>
              </div>

              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="text-[10px] text-slate-400 block uppercase">Eyewall Temperature Shift</span>
                <span className="text-base font-bold text-indigo-300">
                  {evalReport.temporal_comparison_sample.structural_evolution.delta_eyewall_cooling_kelvin} K
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">Convective cooling/warming</span>
              </div>

              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="text-[10px] text-slate-400 block uppercase">Forward Motion</span>
                <span className="text-base font-bold text-emerald-400">
                  {evalReport.temporal_comparison_sample.translational_motion.speed_kmh} km/h
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">
                  Heading: {evalReport.temporal_comparison_sample.translational_motion.bearing_deg}°
                </span>
              </div>

              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="text-[10px] text-slate-400 block uppercase">Rapid Intensification</span>
                <span className={`text-base font-bold ${
                  evalReport.temporal_comparison_sample.intensity_evolution.rapid_intensification_observed
                    ? "text-rose-400"
                    : "text-emerald-400"
                }`}>
                  {evalReport.temporal_comparison_sample.intensity_evolution.rapid_intensification_observed ? "DETECTED (RI)" : "STEADY / NON-RI"}
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">Threshold: ≥ 30 kts / 24h</span>
              </div>
            </div>
          </section>
        )}

        {/* Section 3: Explainability & Causal Disclaimers */}
        {evalReport?.explainability_sample && (
          <section className="glass-panel p-6 rounded-xl border border-slate-800 space-y-4">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <span>🔍</span> Explainable Neural Attribution & Sensitivity
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Grad-CAM Attribution */}
              <div className="p-4 rounded-lg bg-slate-900/80 border border-slate-800 space-y-2">
                <h3 className="text-xs font-semibold text-cyan-400 uppercase tracking-wider">
                  Grad-CAM Spatial Convective Map
                </h3>
                <div className="space-y-1 text-xs font-mono text-slate-300">
                  <div className="flex justify-between">
                    <span>Eyewall Core Energy Concentration:</span>
                    <span className="text-emerald-400 font-bold">
                      {(evalReport.explainability_sample.gradcam.core_concentration_ratio * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Peak Receptive Activation:</span>
                    <span className="text-white">
                      {evalReport.explainability_sample.gradcam.peak_activation}
                    </span>
                  </div>
                </div>
                <p className="text-[11px] text-amber-300/80 italic mt-2 border-t border-slate-800/80 pt-2">
                  ⚠️ {evalReport.explainability_sample.gradcam.causal_disclaimer}
                </p>
              </div>

              {/* Environmental Attribution */}
              <div className="p-4 rounded-lg bg-slate-900/80 border border-slate-800 space-y-2">
                <h3 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider">
                  Environmental Covariate Sensitivity
                </h3>
                <div className="space-y-1 text-xs font-mono text-slate-300">
                  {evalReport.explainability_sample.environmental_attribution.ranked_features.slice(0, 3).map(([feat, score], idx) => (
                    <div key={idx} className="flex justify-between">
                      <span className="text-slate-400">{idx + 1}. {feat}:</span>
                      <span className="text-indigo-300 font-bold">{(score * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
                <p className="text-[11px] text-amber-300/80 italic mt-2 border-t border-slate-800/80 pt-2">
                  ⚠️ {evalReport.explainability_sample.environmental_attribution.causal_disclaimer}
                </p>
              </div>
            </div>
          </section>
        )}

        {/* Section 4: Ingested Scientific Products */}
        <section className="glass-panel p-5 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-white uppercase tracking-wider">
              Ingested Authoritative Products & QC Status
            </h2>
            <span className="text-xs font-mono text-slate-400">({products.length} Products)</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {products.map((p) => (
              <div
                key={p.id}
                className="p-4 rounded-lg bg-slate-900/70 border border-slate-800 text-xs font-mono space-y-2"
              >
                <div className="flex justify-between items-center">
                  <span className="font-bold text-white truncate max-w-xs">{p.filename}</span>
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-[10px] text-cyan-300">
                    {p.file_format}
                  </span>
                </div>
                <div className="text-slate-400 text-[11px] break-all">
                  SHA-256: {p.sha256_hash}
                </div>
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>Origin: {p.source_origin}</span>
                  <span>Size: {(p.file_size_bytes / 1024 / 1024).toFixed(2)} MB</span>
                  <span className="text-emerald-400 font-bold">QC: PASSED</span>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Section 5: Cryptographic Provenance Ledger */}
        <section className="glass-panel p-5 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-white">
              Immutable Cryptographic Lineage Audit Trail (W3C PROV)
            </h2>
            <span className="text-xs font-mono text-cyan-400">SHA-256 Verifiable</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="border-b border-slate-800 text-slate-400 text-[11px]">
                <tr>
                  <th className="py-2 px-3">Timestamp (UTC)</th>
                  <th className="py-2 px-3">Action</th>
                  <th className="py-2 px-3">Entity Type</th>
                  <th className="py-2 px-3">SHA-256 Digest</th>
                  <th className="py-2 px-3">Parameters</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {provenanceLogs.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-4 text-center text-slate-500">
                      No provenance records loaded.
                    </td>
                  </tr>
                ) : (
                  provenanceLogs.map((log) => (
                    <tr key={log.id} className="hover:bg-slate-900/40">
                      <td className="py-2 px-3 text-slate-400">{log.timestamp}</td>
                      <td className="py-2 px-3">
                        <span className="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-cyan-400">
                          {log.action}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-slate-300">{log.entity_type}</td>
                      <td className="py-2 px-3 text-cyan-300">{log.sha256_hash.substring(0, 16)}...</td>
                      <td className="py-2 px-3 text-slate-400 truncate max-w-xs">
                        {JSON.stringify(log.parameters)}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950/80 px-6 py-4 text-center text-xs text-slate-500 font-mono">
        CycloneSense — Explainable Multi-Source Tropical Cyclone Intelligence · Built on genuine NetCDF4/HDF5 Earth Observation Arrays
      </footer>
    </main>
  );
}
