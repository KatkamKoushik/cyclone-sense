"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  IconCompare,
  IconActivity,
  IconRefresh,
  IconAlert,
  IconShield,
} from "@/components/icons";
import { api, StormSummary, StormTrackResponse, TemporalComparisonResult } from "@/lib/api";

function TemporalComparisonContent() {
  const searchParams = useSearchParams();
  const initialStormId = searchParams.get("storm_id") || "2020136N10088";

  const [storms, setStorms] = useState<StormSummary[]>([]);
  const [selectedStormId, setSelectedStormId] = useState<string>(initialStormId);
  const [track, setTrack] = useState<StormTrackResponse | null>(null);

  const [t1Index, setT1Index] = useState<number>(0);
  const [t2Index, setT2Index] = useState<number>(1);

  const [comparisonResult, setComparisonResult] = useState<TemporalComparisonResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load catalog on mount
  useEffect(() => {
    let isMounted = true;
    api
      .getStormCatalog({ limit: 50 })
      .then((res) => {
        if (isMounted) {
          setStorms(res.storms);
        }
      })
      .catch((err) => {
        if (isMounted) setError(err instanceof Error ? err.message : "Failed to load storm catalog");
      });

    return () => {
      isMounted = false;
    };
  }, []);

  // Load track when selectedStormId changes
  useEffect(() => {
    if (!selectedStormId) return;
    let isMounted = true;
    api
      .getStormTrack(selectedStormId)
      .then((data) => {
        if (!isMounted) return;
        setTrack(data);
        const maxIdx = data.observations.length - 1;
        setT1Index(0);
        setT2Index(Math.min(3, maxIdx));
        setLoading(false);
      })
      .catch((err) => {
        if (isMounted) {
          setError(err instanceof Error ? err.message : "Failed to load storm track");
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [selectedStormId]);

  const handleRunComparison = async () => {
    if (!selectedStormId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.runTemporalComparison({
        storm_id: selectedStormId,
        obs_index_t1: t1Index,
        obs_index_t2: t2Index,
      });
      setComparisonResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Temporal comparison failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="pb-2 border-b border-slate-800/80">
        <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-white flex items-center gap-3">
          <span>T1 → T2 Temporal Cyclone Evolution</span>
          <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
            Spatial Differential Analytics
          </span>
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Perform spatially aligned differential kinematics, core convective cooling analysis, and Rapid Intensification detection across two authentic timesteps.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-center gap-3">
          <IconAlert className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Storm & Timestep Selection Bar */}
      <div className="p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm font-mono">
          {/* Storm Select */}
          <div>
            <label className="block text-slate-400 mb-1 font-semibold">Select Tropical Cyclone:</label>
            <select
              value={selectedStormId}
              onChange={(e) => setSelectedStormId(e.target.value)}
              className="w-full py-2 px-3 bg-slate-900 border border-slate-700 rounded-xl text-cyan-300 focus:outline-none focus:border-cyan-500"
            >
              {storms.map((s) => (
                <option key={s.storm_id} value={s.storm_id}>
                  {s.storm_name} ({s.season}) — {s.peak_intensity_kts.toFixed(0)} kts max
                </option>
              ))}
            </select>
          </div>

          {/* T1 Observation */}
          <div>
            <label className="block text-slate-400 mb-1 font-semibold">Observation T1 (Baseline):</label>
            <select
              value={t1Index}
              onChange={(e) => setT1Index(parseInt(e.target.value))}
              disabled={!track}
              className="w-full py-2 px-3 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:border-cyan-500"
            >
              {track?.observations.map((o) => (
                <option key={o.index} value={o.index}>
                  [{o.index}] {o.timestamp.replace("T", " ").substring(5, 16)} — {o.wind_kts.toFixed(0)} kts ({o.latitude.toFixed(1)}°N, {o.longitude.toFixed(1)}°E)
                </option>
              ))}
            </select>
          </div>

          {/* T2 Observation */}
          <div>
            <label className="block text-slate-400 mb-1 font-semibold">Observation T2 (Subsequent):</label>
            <select
              value={t2Index}
              onChange={(e) => setT2Index(parseInt(e.target.value))}
              disabled={!track}
              className="w-full py-2 px-3 bg-slate-900 border border-slate-700 rounded-xl text-white focus:outline-none focus:border-cyan-500"
            >
              {track?.observations.map((o) => (
                <option key={o.index} value={o.index}>
                  [{o.index}] {o.timestamp.replace("T", " ").substring(5, 16)} — {o.wind_kts.toFixed(0)} kts ({o.latitude.toFixed(1)}°N, {o.longitude.toFixed(1)}°E)
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex justify-end pt-2">
          <button
            onClick={handleRunComparison}
            disabled={loading || t1Index === t2Index}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 via-indigo-600 to-teal-500 hover:from-cyan-500 hover:to-teal-400 text-white font-bold text-xs shadow-lg shadow-cyan-950/40 transition-all disabled:opacity-50 font-mono"
          >
            {loading ? (
              <>
                <IconRefresh className="w-4 h-4 animate-spin" />
                <span>Computing Spatial & Thermal Deltas...</span>
              </>
            ) : (
              <>
                <IconCompare className="w-4 h-4" />
                <span>Execute T1 → T2 Differential Analysis</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Comparison Results Section */}
      {comparisonResult && (
        <div className="space-y-6 animate-in fade-in">
          {/* Primary Summary Header Banner */}
          <div className="p-6 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-2xl flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <div className="text-sm font-mono text-cyan-400 uppercase tracking-wider">
                Differential Evolution Analysis
              </div>
              <h2 className="text-xl font-extrabold text-white mt-0.5">
                {comparisonResult.storm_name} ({comparisonResult.storm_id})
              </h2>
              <div className="text-xs text-slate-400 font-mono mt-1">
                Interval: {comparisonResult.temporal_interval_hours.toFixed(1)} hours ({comparisonResult.t1.timestamp} → {comparisonResult.t2.timestamp})
              </div>
            </div>

            {/* Rapid Intensification Badge */}
            <div className="flex items-center gap-2">
              {comparisonResult.intensity_evolution.rapid_intensification_observed ? (
                <div className="px-3.5 py-1.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 font-mono text-xs font-bold flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-ping"></span>
                  <span>RAPID INTENSIFICATION (RI) DETECTED</span>
                </div>
              ) : (
                <div className="px-3 py-1.5 rounded-xl bg-slate-800 border border-slate-700 text-slate-300 font-mono text-xs">
                  Normal Evolution Rate
                </div>
              )}
            </div>
          </div>

          {/* 3-Column Detailed Diagnostics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Column 1: Translational Kinematics */}
            <div className="p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-3 font-mono">
              <h3 className="text-xs font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-2">
                <IconActivity className="w-4 h-4 text-cyan-400" />
                <span>Translational Kinematics</span>
              </h3>

              <div className="space-y-2.5 text-xs">
                <div className="flex justify-between items-center p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-slate-400">Total Displacement:</span>
                  <span className="text-white font-bold">
                    {comparisonResult.translational_motion.displacement_km.toFixed(1)} km
                  </span>
                </div>

                <div className="flex justify-between items-center p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-slate-400">Translation Speed:</span>
                  <span className="text-cyan-300 font-bold">
                    {comparisonResult.translational_motion.speed_kmh.toFixed(1)} km/h
                  </span>
                </div>

                <div className="flex justify-between items-center p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-slate-400">Forward Bearing:</span>
                  <span className="text-slate-300">
                    {comparisonResult.translational_motion.bearing_deg.toFixed(1)}° azimuth
                  </span>
                </div>

                <div className="p-2.5 rounded-xl bg-slate-900/40 text-sm text-slate-400 font-sans">
                  Coordinate motion: ({comparisonResult.t1.coords[0].toFixed(1)}°N, {comparisonResult.t1.coords[1].toFixed(1)}°E) → ({comparisonResult.t2.coords[0].toFixed(1)}°N, {comparisonResult.t2.coords[1].toFixed(1)}°E)
                </div>
              </div>
            </div>

            {/* Column 2: Structural & Thermal Convection */}
            <div className="p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-3 font-mono">
              <h3 className="text-xs font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-2">
                <IconShield className="w-4 h-4 text-teal-400" />
                <span>Convective Thermal Deltas</span>
              </h3>

              <div className="space-y-2.5 text-xs">
                <div className="flex justify-between items-center p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-slate-400">Eyewall Cooling (ΔT Eyewall):</span>
                  <span
                    className={`font-bold ${
                      comparisonResult.structural_evolution.delta_eyewall_cooling_kelvin <= 0
                        ? "text-emerald-400"
                        : "text-amber-400"
                    }`}
                  >
                    {comparisonResult.structural_evolution.delta_eyewall_cooling_kelvin.toFixed(2)} K
                  </span>
                </div>

                <div className="flex justify-between items-center p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-slate-400">Eye Warming (ΔT Eye):</span>
                  <span className="text-white font-bold">
                    {comparisonResult.structural_evolution.delta_eye_warming_kelvin.toFixed(2)} K
                  </span>
                </div>

                <div className="flex justify-between items-center p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-slate-400">Convective Vigor Delta:</span>
                  <span className="text-slate-300">
                    {comparisonResult.structural_evolution.delta_convective_vigor_ratio.toFixed(3)}
                  </span>
                </div>

                <div className="text-xs text-slate-500 truncate p-2">
                  Diff SHA256: {comparisonResult.structural_evolution.diff_grid_sha256.substring(0, 20)}...
                </div>
              </div>
            </div>

            {/* Column 3: Intensity & RI Evolution */}
            <div className="p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-3 font-mono">
              <h3 className="text-xs font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-2">
                <IconCompare className="w-4 h-4 text-indigo-400" />
                <span>Observed Intensity Rate</span>
              </h3>

              <div className="space-y-2.5 text-xs">
                <div className="flex justify-between items-center p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-slate-400">Intensity Change (ΔV):</span>
                  <span
                    className={`font-bold ${
                      comparisonResult.intensity_evolution.delta_wind_true_kts >= 0
                        ? "text-cyan-300"
                        : "text-slate-400"
                    }`}
                  >
                    {comparisonResult.intensity_evolution.delta_wind_true_kts >= 0 ? "+" : ""}
                    {comparisonResult.intensity_evolution.delta_wind_true_kts.toFixed(1)} kts
                  </span>
                </div>

                <div className="flex justify-between items-center p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-slate-400">Hourly Rate (ΔV/Δt):</span>
                  <span className="text-white font-bold">
                    {comparisonResult.intensity_evolution.rate_kts_per_hr.toFixed(2)} kts/h
                  </span>
                </div>

                <div className="flex justify-between items-center p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-slate-400">12-Hour Normalized Rate:</span>
                  <span className="text-indigo-300 font-bold">
                    {(comparisonResult.intensity_evolution.rate_kts_per_hr * 12).toFixed(1)} kts/12h
                  </span>
                </div>

                <div className="p-2.5 rounded-xl bg-slate-900/40 text-sm text-slate-400 font-sans">
                  Pressure evolution: {comparisonResult.t1.pressure_hpa || "—"} hPa → {comparisonResult.t2.pressure_hpa || "—"} hPa
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function TemporalComparisonPage() {
  return (
    <Suspense fallback={<div className="py-20 text-center text-sm font-mono text-slate-400">Loading Temporal Comparator...</div>}>
      <TemporalComparisonContent />
    </Suspense>
  );
}
