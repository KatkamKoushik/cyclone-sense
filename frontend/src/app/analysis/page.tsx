"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  IconPlay,
  IconCpu,
  IconActivity,
  IconAlert,
  IconCheck,
  IconRefresh,
  IconSatellite,
  IconDatabase,
} from "@/components/icons";
import { api, InferencePayload, AnalysisJob, ScientificProduct } from "@/lib/api";

function AnalysisStudioContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Mode: "satellite" for direct real-time ingested satellite granules, "benchmark" for historical storms
  const initialMode = (searchParams.get("mode") as "satellite" | "benchmark") || "satellite";
  const [dataSourceMode, setDataSourceMode] = useState<"satellite" | "benchmark">(initialMode);

  // Satellite products state
  const [products, setProducts] = useState<ScientificProduct[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string>(searchParams.get("product_id") || "");
  const [loadingProducts, setLoadingProducts] = useState(false);

  // Pre-fill from query params if available
  const initialStormId = searchParams.get("storm_id") || "2020136N10088";
  const initialStormName = searchParams.get("storm_name") || "AMPHAN";
  const initialLat = parseFloat(searchParams.get("lat") || "18.5");
  const initialLon = parseFloat(searchParams.get("lon") || "-65.0");
  const initialPres = parseFloat(searchParams.get("pres") || "980.0");
  const initialSpeed = parseFloat(searchParams.get("speed") || "16.0");
  const initialBearing = parseFloat(searchParams.get("bearing") || "310.0");

  const [stormId, setStormId] = useState(initialStormId);
  const [stormName, setStormName] = useState(initialStormName);
  const [centerLat, setCenterLat] = useState(initialLat);
  const [centerLon, setCenterLon] = useState(initialLon);
  const [pressureHpa, setPressureHpa] = useState(initialPres);
  const [forwardSpeedKmh, setForwardSpeedKmh] = useState(initialSpeed);
  const [forwardBearingDeg, setForwardBearingDeg] = useState(initialBearing);

  const [modelType, setModelType] = useState<"fusion" | "image" | "environment" | "baseline">("fusion");

  const [submitting, setSubmitting] = useState(false);
  const [jobResult, setJobResult] = useState<AnalysisJob | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Available sample storms
  const SAMPLE_STORMS = [
    { id: "2020136N10088", name: "AMPHAN", season: 2020, lat: 11.2, lon: 86.1, pres: 982.0, speed: 15.0, bearing: 315.0 },
    { id: "2019117N09888", name: "FANI", season: 2019, lat: 14.5, lon: 85.0, pres: 932.0, speed: 18.0, bearing: 340.0 },
    { id: "2023157N13066", name: "BIPARJOY", season: 2023, lat: 18.2, lon: 67.5, pres: 960.0, speed: 10.0, bearing: 355.0 },
    { id: "2023130N11088", name: "MOCHA", season: 2023, lat: 16.0, lon: 90.5, pres: 938.0, speed: 20.0, bearing: 30.0 },
  ];

  // Fetch available authentic ingested satellite products
  useEffect(() => {
    async function loadProducts() {
      setLoadingProducts(true);
      try {
        const list = await api.listProducts(30);
        setProducts(list);
        if (!selectedProductId && list.length > 0) {
          // Default to the first satellite granule (prefer NOAA GOES or NASA products)
          const preferred = list.find((p) => p.source_origin.includes("GOES") || p.source_origin.includes("NASA")) || list[0];
          setSelectedProductId(preferred.id);
        }
      } catch (e) {
        console.error("Failed to load satellite products:", e);
      } finally {
        setLoadingProducts(false);
      }
    }
    loadProducts();
  }, []);

  const selectedProduct = products.find((p) => p.id === selectedProductId);

  const handleSelectSample = (s: typeof SAMPLE_STORMS[0]) => {
    setStormId(s.id);
    setStormName(s.name);
    setCenterLat(s.lat);
    setCenterLon(s.lon);
    setPressureHpa(s.pres);
    setForwardSpeedKmh(s.speed);
    setForwardBearingDeg(s.bearing);
  };

  const handleExecuteAnalysis = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    setJobResult(null);

    const payload: InferencePayload = {
      model_type: modelType,
      center_latitude: centerLat,
      center_longitude: centerLon,
      pressure_hpa: pressureHpa,
      forward_speed_kmh: forwardSpeedKmh,
      forward_bearing_deg: forwardBearingDeg,
      storm_id: dataSourceMode === "satellite" ? "REALTIME_SAT" : stormId,
      storm_name: dataSourceMode === "satellite" ? (selectedProduct?.filename || "REALTIME_GRANULE") : stormName,
      product_id: dataSourceMode === "satellite" && selectedProductId ? selectedProductId : undefined,
    };

    try {
      const job = await api.submitAnalysisJob(payload);
      setJobResult(job);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Inference execution failed");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="pb-2 border-b border-slate-800/80">
        <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-white flex items-center gap-3">
          <span>Cyclone Analysis Studio</span>
          <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
            Multimodal Neural Inference & Grad-CAM
          </span>
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Execute real-time neural inference on live ingested satellite granules (NOAA GOES-16, NASA Earthdata) or historical cyclone tracks.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-center gap-3">
          <IconAlert className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Input Source Mode Selector */}
      <div className="p-4 rounded-2xl bg-[#0c121e]/80 border border-slate-800/80 shadow-lg space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-300 font-mono uppercase tracking-wider">
            Observation Data Source:
          </span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setDataSourceMode("satellite")}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-mono font-bold transition-all ${
                dataSourceMode === "satellite"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm"
                  : "bg-slate-900/60 hover:bg-slate-800/60 text-slate-400 border border-slate-800"
              }`}
            >
              <IconSatellite className="w-3.5 h-3.5" />
              <span>Direct Satellite Granule (Real-Time Ingested)</span>
            </button>
            <button
              type="button"
              onClick={() => setDataSourceMode("benchmark")}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-mono font-bold transition-all ${
                dataSourceMode === "benchmark"
                  ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 shadow-sm"
                  : "bg-slate-900/60 hover:bg-slate-850/60 text-slate-400 border border-slate-800"
              }`}
            >
              <IconDatabase className="w-3.5 h-3.5" />
              <span>Historical IBTrACS Benchmark Storms</span>
            </button>
          </div>
        </div>

        {/* Satellite Mode: Ingested Granule Selector */}
        {dataSourceMode === "satellite" && (
          <div className="p-4 rounded-xl bg-slate-950/60 border border-cyan-500/20 space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-mono font-semibold text-cyan-300 flex items-center gap-2">
                <IconSatellite className="w-4 h-4 text-cyan-400" />
                <span>Select Ingested Satellite Granule (NetCDF4 / HDF5):</span>
              </label>
              <button
                type="button"
                onClick={() => router.push("/settings")}
                className="text-xs font-mono text-cyan-400 hover:text-cyan-300 underline"
              >
                + Acquire Live Granule from NOAA S3 / NASA CMR →
              </button>
            </div>

            {loadingProducts ? (
              <div className="py-2 text-xs font-mono text-slate-400 flex items-center gap-2">
                <IconRefresh className="w-3.5 h-3.5 animate-spin" />
                <span>Loading available scientific products from database...</span>
              </div>
            ) : products.length === 0 ? (
              <div className="text-xs font-mono text-amber-300">
                No satellite products ingested yet. Go to Settings &gt; Real-Time Satellite Acquisition Console to pull live granules from NOAA GOES or NASA CMR.
              </div>
            ) : (
              <select
                value={selectedProductId}
                onChange={(e) => setSelectedProductId(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-white font-mono text-xs focus:outline-none focus:border-cyan-500"
              >
                {products.map((p) => (
                  <option key={p.id} value={p.id}>
                    [{p.source_origin}] {p.filename} ({((p.file_size_bytes || 0) / 1024 / 1024).toFixed(2)} MB, {p.variables_count} vars)
                  </option>
                ))}
              </select>
            )}

            {selectedProduct && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px] font-mono p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 text-slate-300">
                <div>
                  <span className="text-slate-500 block">Origin:</span>
                  <span className="font-bold text-cyan-400">{selectedProduct.source_origin}</span>
                </div>
                <div>
                  <span className="text-slate-500 block">Format:</span>
                  <span className="text-white">{selectedProduct.file_format}</span>
                </div>
                <div>
                  <span className="text-slate-500 block">SHA-256 Digest:</span>
                  <span className="text-slate-400 truncate block" title={selectedProduct.sha256_hash}>
                    {selectedProduct.sha256_hash?.substring(0, 16)}...
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block">Pipeline Status:</span>
                  <span className="text-emerald-400 font-bold">2D Raster Direct Feed Active</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Benchmark Mode: Quick Select Storms */}
        {dataSourceMode === "benchmark" && (
          <div className="space-y-2">
            <div className="text-xs font-semibold text-slate-300 font-mono">
              Quick Select Benchmark Cyclones:
            </div>
            <div className="flex flex-wrap gap-2">
              {SAMPLE_STORMS.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => handleSelectSample(s)}
                  className={`px-3 py-1.5 rounded-xl text-sm font-mono transition-all ${
                    stormId === s.id
                      ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 font-bold"
                      : "bg-slate-900/60 hover:bg-slate-800/60 text-slate-300 border border-slate-800"
                  }`}
                >
                  {s.name} ({s.season})
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Main Analysis Form */}
      <form onSubmit={handleExecuteAnalysis} className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column: Model Architecture Selection */}
          <div className="p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-4">
            <h2 className="text-lg font-bold text-white flex items-center gap-2 border-b border-slate-800/80 pb-3">
              <IconCpu className="w-4 h-4 text-indigo-400" />
              <span>Select Model Architecture</span>
            </h2>

            <div className="space-y-3">
              {/* Multimodal Fusion */}
              <label
                className={`p-3.5 rounded-xl border flex flex-col gap-1 cursor-pointer transition-all ${
                  modelType === "fusion"
                    ? "bg-indigo-950/20 border-indigo-500/50 shadow-sm shadow-indigo-950/40"
                    : "bg-slate-900/60 hover:bg-slate-850/60 border-slate-800"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="modelType"
                      value="fusion"
                      checked={modelType === "fusion"}
                      onChange={() => setModelType("fusion")}
                      className="text-cyan-500 focus:ring-0"
                    />
                    <span className="text-xs font-bold text-white">Multimodal Fusion (Simple MLP)</span>
                  </div>
                  <span className="text-sm font-mono px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300">
                    Recommended
                  </span>
                </div>
                <p className="text-sm text-slate-400 pl-5">
                  Concatenates 2-channel CNN latent vector ($D=128$) with LayerNorm environment embeddings ($D=64$). Lowest bias (+0.22 kts) and highest F1 (0.7604).
                </p>
              </label>

              {/* Image-Only CNN */}
              <label
                className={`p-3.5 rounded-xl border flex flex-col gap-1 cursor-pointer transition-all ${
                  modelType === "image"
                    ? "bg-cyan-950/20 border-cyan-500/50"
                    : "bg-slate-900/60 hover:bg-slate-850/60 border-slate-800"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="modelType"
                      value="image"
                      checked={modelType === "image"}
                      onChange={() => setModelType("image")}
                      className="text-cyan-500 focus:ring-0"
                    />
                    <span className="text-xs font-bold text-white">Image-Only CNN</span>
                  </div>
                  <span className="text-sm font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                    MAE 2.02 kts
                  </span>
                </div>
                <p className="text-sm text-slate-400 pl-5">
                  Multi-scale residual CNN consuming 10.35 µm IR and 6.2 µm Water Vapor fields. Includes Grad-CAM hooks.
                </p>
              </label>

              {/* Environment-Only MLP */}
              <label
                className={`p-3.5 rounded-xl border flex flex-col gap-1 cursor-pointer transition-all ${
                  modelType === "environment"
                    ? "bg-teal-950/20 border-teal-500/50"
                    : "bg-slate-900/60 hover:bg-slate-850/60 border-slate-800"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="modelType"
                      value="environment"
                      checked={modelType === "environment"}
                      onChange={() => setModelType("environment")}
                      className="text-cyan-500 focus:ring-0"
                    />
                    <span className="text-xs font-bold text-white">Environment-Only MLP</span>
                  </div>
                  <span className="text-sm font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                    8 Covariates
                  </span>
                </div>
                <p className="text-sm text-slate-400 pl-5">
                  LayerNorm MLP consuming Coriolis, pressure deficit, kinematics, and seasonal phase covariates.
                </p>
              </label>

              {/* Baseline CLIPER */}
              <label
                className={`p-3.5 rounded-xl border flex flex-col gap-1 cursor-pointer transition-all ${
                  modelType === "baseline"
                    ? "bg-slate-800/40 border-slate-600"
                    : "bg-slate-900/60 hover:bg-slate-850/60 border-slate-800"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="modelType"
                      value="baseline"
                      checked={modelType === "baseline"}
                      onChange={() => setModelType("baseline")}
                      className="text-cyan-500 focus:ring-0"
                    />
                    <span className="text-xs font-bold text-white">Baseline CLIPER (Ridge)</span>
                  </div>
                  <span className="text-sm font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                    Closed-Form
                  </span>
                </div>
                <p className="text-sm text-slate-400 pl-5">
                  Analytical regularized Ridge regression physical persistence baseline (λ=1.0).
                </p>
              </label>
            </div>
          </div>

          {/* Right Column: Physical Observation Coordinates */}
          <div className="p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-4">
            <h2 className="text-lg font-bold text-white flex items-center gap-2 border-b border-slate-800/80 pb-3">
              <IconActivity className="w-4 h-4 text-cyan-400" />
              <span>Observation & Coordinate Parameters</span>
            </h2>

            <div className="grid grid-cols-2 gap-4 text-sm font-mono">
              <div>
                <label className="block text-slate-400 mb-1">
                  {dataSourceMode === "satellite" ? "Identifier" : "Storm Identifier"}
                </label>
                <input
                  type="text"
                  value={dataSourceMode === "satellite" ? (selectedProduct?.filename ? selectedProduct.filename.substring(0, 20) : "REALTIME_SAT") : stormId}
                  onChange={(e) => setStormId(e.target.value)}
                  disabled={dataSourceMode === "satellite"}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:border-cyan-500 disabled:opacity-60"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">
                  {dataSourceMode === "satellite" ? "Granule Name" : "Storm Name"}
                </label>
                <input
                  type="text"
                  value={dataSourceMode === "satellite" ? (selectedProduct?.filename || "REALTIME_GRANULE") : stormName}
                  onChange={(e) => setStormName(e.target.value)}
                  disabled={dataSourceMode === "satellite"}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:border-cyan-500 disabled:opacity-60"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Latitude (°N)</label>
                <input
                  type="number"
                  step="0.1"
                  min="-90"
                  max="90"
                  value={centerLat}
                  onChange={(e) => setCenterLat(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Longitude (°E)</label>
                <input
                  type="number"
                  step="0.1"
                  min="-180"
                  max="180"
                  value={centerLon}
                  onChange={(e) => setCenterLon(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Central Pressure (hPa)</label>
                <input
                  type="number"
                  step="1"
                  min="850"
                  max="1050"
                  value={pressureHpa}
                  onChange={(e) => setPressureHpa(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Forward Speed (km/h)</label>
                <input
                  type="number"
                  step="0.5"
                  min="0"
                  max="150"
                  value={forwardSpeedKmh}
                  onChange={(e) => setForwardSpeedKmh(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div className="col-span-2">
                <label className="block text-slate-400 mb-1">Forward Bearing (° azimuth)</label>
                <input
                  type="number"
                  step="1"
                  min="0"
                  max="360"
                  value={forwardBearingDeg}
                  onChange={(e) => setForwardBearingDeg(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:border-cyan-500"
                />
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-sm text-slate-400">
              <strong className="text-slate-300">Physics Sanity Check:</strong> Physical parameters are normalized within canonical meteorological bounds (175 K ≤ T_B ≤ 320 K, 850 hPa ≤ P ≤ 1050 hPa).
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={submitting}
            className="flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-600 via-indigo-600 to-teal-500 hover:from-cyan-500 hover:to-teal-400 text-white font-bold text-sm shadow-xl shadow-cyan-950/50 transition-all disabled:opacity-50"
          >
            {submitting ? (
              <>
                <IconRefresh className="w-4 h-4 animate-spin" />
                <span>Executing Multimodal Neural Inference...</span>
              </>
            ) : (
              <>
                <IconPlay className="w-4 h-4" />
                <span>
                  {dataSourceMode === "satellite"
                    ? "Execute Direct Satellite Granule Inference"
                    : "Execute Cyclone Analysis Job"}
                </span>
              </>
            )}
          </button>
        </div>
      </form>

      {/* Immediate Execution Result Card */}
      {jobResult && (
        <div className="p-6 rounded-2xl bg-emerald-950/20 border border-emerald-500/30 shadow-2xl space-y-4 animate-in fade-in slide-in-from-bottom-4">
          <div className="flex items-center justify-between border-b border-emerald-500/20 pb-3">
            <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm">
              <IconCheck className="w-5 h-5" />
              <span>Inference Executed Successfully & Provenance Logged</span>
            </div>
            <span className="text-sm font-mono text-slate-400">
              Job ID: {jobResult.job_id.substring(0, 8)}...
            </span>
          </div>

          {jobResult.input_parameters?.product_id && (
            <div className="p-2.5 rounded-xl bg-cyan-950/30 border border-cyan-500/30 text-xs font-mono text-cyan-300 flex items-center justify-between">
              <span>Direct Satellite Input Product:</span>
              <span className="font-bold text-white">{jobResult.input_parameters.product_id}</span>
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono">
            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
              <div className="text-xs text-slate-400">Predicted Wind Intensity</div>
              <div className="text-2xl font-extrabold text-cyan-300">
                {jobResult.predicted_intensity_kts?.toFixed(1)} kts
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
              <div className="text-xs text-slate-400">Pattern Severity</div>
              <div className="text-xs font-bold text-emerald-400 mt-2">
                {jobResult.category_name}
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
              <div className="text-xs text-slate-400">Eyewall Core Concentration</div>
              <div className="text-2xl font-extrabold text-indigo-300">
                {jobResult.explainability?.gradcam && jobResult.explainability.gradcam.is_valid !== false
                  ? `${(jobResult.explainability.gradcam.core_concentration_ratio * 100).toFixed(1)}%`
                  : jobResult.explainability?.gradcam
                  ? "No Positive Activation"
                  : "N/A (Closed-Form)"}
              </div>
            </div>
          </div>

          {/* Ground Truth comparison - shown only when reference is available */}
          {jobResult.reference_intensity_kts != null && (
            <div className="grid grid-cols-2 gap-4 pt-2 border-t border-emerald-500/20">
              <div className="p-3 rounded-xl bg-emerald-950/30 border border-emerald-500/30 font-mono">
                <div className="text-xs text-slate-400">IBTrACS Ground Truth (V_max)</div>
                <div className="text-xl font-extrabold text-emerald-300 mt-1">
                  {jobResult.reference_intensity_kts.toFixed(1)} kts
                </div>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 font-mono">
                <div className="text-xs text-slate-400">Absolute Error</div>
                <div className={`text-xl font-extrabold mt-1 ${
                  (jobResult.absolute_error_kts ?? 99) < 5 ? "text-emerald-300" :
                  (jobResult.absolute_error_kts ?? 99) < 15 ? "text-amber-300" : "text-rose-300"
                }`}>
                  {jobResult.absolute_error_kts != null ? `${jobResult.absolute_error_kts.toFixed(2)} kts` : "—"}
                </div>
              </div>
            </div>
          )}

          <div className="flex justify-end gap-3 pt-2">
            <button
              onClick={() => router.push(`/results?jobId=${jobResult.job_id}`)}
              className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold shadow-lg transition-all font-mono"
            >
              Inspect Complete Report & Grad-CAM Heatmap →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function AnalysisStudioPage() {
  return (
    <Suspense fallback={<div className="py-20 text-center text-sm font-mono text-slate-400">Loading Analysis Studio...</div>}>
      <AnalysisStudioContent />
    </Suspense>
  );
}
