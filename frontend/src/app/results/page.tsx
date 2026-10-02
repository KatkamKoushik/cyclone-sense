"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  IconCpu,
  IconActivity,
  IconShield,
  IconAlert,
  IconRefresh,
  IconSparkles,
} from "@/components/icons";
import { api, AnalysisJob } from "@/lib/api";

const CATEGORY_NAMES = [
  "Tropical Depression (< 34 kts)",
  "Cyclonic Storm (34-63 kts)",
  "Very Severe Cyclonic Storm (64-89 kts)",
  "Extremely Severe Cyclonic Storm (90-119 kts)",
  "Super Cyclonic Storm (>= 120 kts)",
];

function ResultsContent() {
  const searchParams = useSearchParams();
  const queryJobId = searchParams.get("jobId");

  const [jobs, setJobs] = useState<AnalysisJob[]>([]);
  const [selectedJob, setSelectedJob] = useState<AnalysisJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchJobs = async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await api.listAnalysisJobs(30);
      setJobs(list);
      if (queryJobId) {
        const found = list.find((j) => j.job_id === queryJobId);
        if (found) {
          setSelectedJob(found);
        } else {
          // Fetch directly
          const direct = await api.getAnalysisJob(queryJobId);
          setSelectedJob(direct);
        }
      } else if (list.length > 0) {
        setSelectedJob(list[0]);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load analysis results");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let isMounted = true;
    api
      .listAnalysisJobs(30)
      .then(async (list) => {
        if (!isMounted) return;
        setJobs(list);
        if (queryJobId) {
          const found = list.find((j) => j.job_id === queryJobId);
          if (found) {
            setSelectedJob(found);
          } else {
            try {
              const direct = await api.getAnalysisJob(queryJobId);
              if (isMounted) setSelectedJob(direct);
            } catch {
              // Ignore direct fetch failure
            }
          }
        } else if (list.length > 0) {
          setSelectedJob(list[0]);
        }
        setLoading(false);
      })
      .catch((err) => {
        if (isMounted) {
          setError(err instanceof Error ? err.message : "Failed to load analysis results");
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [queryJobId]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 pb-2 border-b border-slate-800/80">
        <div>
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-white flex items-center gap-3">
            <span>Inference Results & Explainability Report</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
              Verified Model Execution
            </span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Detailed inspection of predicted intensity, category probabilities, Grad-CAM spatial activation, and environmental feature sensitivities.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={fetchJobs}
            disabled={loading}
            className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700 text-slate-300 transition-colors disabled:opacity-50"
          >
            <IconRefresh className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </button>
          <Link
            href="/analysis"
            className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-cyan-600 hover:bg-cyan-500 text-white transition-all font-mono"
          >
            + New Analysis
          </Link>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-center gap-3">
          <IconAlert className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {loading ? (
        <div className="py-24 text-center text-slate-400 font-mono text-xs space-y-2">
          <IconRefresh className="w-6 h-6 animate-spin mx-auto text-cyan-400" />
          <p>Loading analysis reports from database...</p>
        </div>
      ) : jobs.length === 0 ? (
        <div className="p-12 text-center rounded-2xl bg-[#0c121e]/90 border border-slate-800 text-slate-400 font-mono text-xs space-y-4">
          <p>No analysis jobs executed yet.</p>
          <Link
            href="/analysis"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-bold"
          >
            <span>Launch Analysis Studio</span>
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Job Selector List (4 cols) */}
          <div className="lg:col-span-4 p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <IconCpu className="w-4 h-4 text-cyan-400" />
                <span>Executed Analysis Jobs</span>
              </h2>
              <span className="text-sm font-mono text-slate-400">{jobs.length} Runs</span>
            </div>

            <div className="space-y-2 max-h-[580px] overflow-y-auto">
              {jobs.map((j) => {
                const isSelected = selectedJob?.job_id === j.job_id;
                return (
                  <div
                    key={j.job_id}
                    onClick={() => setSelectedJob(j)}
                    className={`p-3 rounded-xl border cursor-pointer transition-all ${
                      isSelected
                        ? "bg-cyan-500/10 border-cyan-500/50 shadow-sm shadow-cyan-950/40"
                        : "bg-slate-900/60 hover:bg-slate-850/60 border-slate-800"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-white">
                        {j.storm_name} ({j.storm_id})
                      </span>
                      <span
                        className={`text-xs px-1.5 py-0.5 rounded font-mono font-semibold ${
                          j.status === "COMPLETED"
                            ? "bg-emerald-500/20 text-emerald-400"
                            : "bg-amber-500/20 text-amber-400"
                        }`}
                      >
                        {j.status}
                      </span>
                    </div>

                    <div className="mt-2 flex items-center justify-between text-sm font-mono">
                      <span className="text-cyan-400 font-bold">
                        {j.predicted_intensity_kts ? `${j.predicted_intensity_kts.toFixed(1)} kts` : "—"}
                      </span>
                      <span className="text-slate-400">{j.model_type}</span>
                    </div>

                    <div className="mt-1 flex items-center justify-between text-xs text-slate-500 font-mono">
                      <span>{new Date(j.created_at).toLocaleTimeString()}</span>
                      <span>ID: {j.job_id.substring(0, 8)}...</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right Column: Detailed Result Dashboard (8 cols) */}
          <div className="lg:col-span-8 space-y-6">
            {selectedJob ? (
              <>
                {/* Top Summary Banner */}
                <div className="p-6 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-800/80 pb-3">
                    <div>
                      <div className="text-xs text-cyan-400 font-mono uppercase tracking-wider">
                        {selectedJob.model_type} Analysis Report
                      </div>
                      <h2 className="text-xl font-extrabold text-white mt-0.5">
                        {selectedJob.storm_name} ({selectedJob.storm_id})
                      </h2>
                    </div>

                    {selectedJob.provenance_id && (
                      <Link
                        href={`/provenance`}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-emerald-400 text-sm font-mono border border-slate-700 transition-colors"
                      >
                        <IconShield className="w-4 h-4" />
                        <span>Inspect Lineage SHA-256</span>
                      </Link>
                    )}
                  </div>

                  {/* Primary Outputs Grid */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 font-mono">
                    {/* Intensity */}
                    <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col justify-between">
                      <div className="text-xs text-slate-400">Predicted Sustained Intensity (V_max)</div>
                      <div className="text-3xl font-extrabold text-cyan-300 mt-2">
                        {selectedJob.predicted_intensity_kts ? `${selectedJob.predicted_intensity_kts.toFixed(1)} kts` : "—"}
                      </div>
                      <div className="text-sm text-slate-500 mt-1">
                        Physical units: Knots (1 kt ≈ 1.852 km/h)
                      </div>
                    </div>

                    {/* Pattern Classification */}
                    <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col justify-between">
                      <div className="text-xs text-slate-400">Pattern Severity Classification</div>
                      <div className="text-base font-bold text-emerald-400 mt-2">
                        {selectedJob.category_name || "Unclassified"}
                      </div>
                      <div className="text-sm text-slate-500 mt-1">
                        Category Index: {selectedJob.predicted_category ?? "—"} / 4
                      </div>
                    </div>
                  </div>

                  {/* Ground Truth vs Prediction (only shown for historical observations) */}
                  {selectedJob.reference_intensity_kts != null && (
                    <div className="grid grid-cols-2 gap-4 pt-2 border-t border-slate-800/60">
                      <div className="p-3 rounded-xl bg-emerald-950/20 border border-emerald-500/30 font-mono">
                        <div className="text-xs text-slate-400">IBTrACS Ground Truth (V_max)</div>
                        <div className="text-2xl font-extrabold text-emerald-300 mt-1">
                          {selectedJob.reference_intensity_kts.toFixed(1)} kts
                        </div>
                        <div className="text-xs text-slate-500 mt-1">Historical observation</div>
                      </div>
                      <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 font-mono">
                        <div className="text-xs text-slate-400">Absolute Error</div>
                        <div className={`text-2xl font-extrabold mt-1 ${
                          (selectedJob.absolute_error_kts ?? 99) < 5 ? "text-emerald-300" :
                          (selectedJob.absolute_error_kts ?? 99) < 15 ? "text-amber-300" : "text-rose-300"
                        }`}>
                          {selectedJob.absolute_error_kts != null ? `${selectedJob.absolute_error_kts.toFixed(2)} kts` : "—"}
                        </div>
                        <div className="text-xs text-slate-500 mt-1">|Predicted − Ground Truth|</div>
                      </div>
                    </div>
                  )}

                  {/* Probability Distribution Bar Chart */}
                  {selectedJob.probabilities && (
                    <div className="space-y-2 pt-2 border-t border-slate-800/80">
                      <div className="text-xs font-bold text-slate-300 font-mono">
                        IMD/WMO Severity Probability Distribution (Softmax):
                      </div>
                      <div className="space-y-1.5 text-sm font-mono">
                        {CATEGORY_NAMES.map((name, idx) => {
                          const prob = selectedJob.probabilities?.[idx] || 0;
                          const pct = (prob * 100).toFixed(1);
                          const isPred = selectedJob.predicted_category === idx;
                          return (
                            <div key={name} className="space-y-1">
                              <div className="flex justify-between text-sm">
                                <span className={isPred ? "text-cyan-300 font-bold" : "text-slate-400"}>
                                  {name}
                                </span>
                                <span className={isPred ? "text-cyan-300 font-bold" : "text-slate-500"}>
                                  {pct}%
                                </span>
                              </div>
                              <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                                <div
                                  className={`h-full rounded-full transition-all duration-500 ${
                                    isPred ? "bg-gradient-to-r from-cyan-500 to-teal-400" : "bg-slate-700"
                                  }`}
                                  style={{ width: `${Math.max(2, parseFloat(pct))}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>

                {/* Explainability Section: Grad-CAM + Environmental Sensitivity */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Grad-CAM Saliency Card */}
                  <div className="p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-3">
                    <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                      <h3 className="text-xs font-bold text-white flex items-center gap-2">
                        <IconSparkles className="w-4 h-4 text-cyan-400" />
                        <span>Grad-CAM Spatial Attribution</span>
                      </h3>
                      <span className="text-sm font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400">
                        Image Encoder
                      </span>
                    </div>

                    {selectedJob.explainability?.gradcam ? (
                      (() => {
                        const gc = selectedJob.explainability.gradcam;
                        if (gc.is_valid === false) {
                          return (
                            <div className="py-8 text-center font-mono text-xs space-y-2">
                              <div className="text-amber-400 font-semibold">No Positive Activation</div>
                              <p className="text-slate-500 text-sm">Grad-CAM returned zero gradients for this output head. This is physically meaningful: the model has no positive gradient flow through the spatial encoder for this target class given the current input. Try probing the intensity head instead.</p>
                            </div>
                          );
                        }
                        return (
                          <div className="space-y-3 font-mono text-xs">
                            <div className="flex justify-between items-center p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
                              <span className="text-slate-400">Eyewall Core Energy:</span>
                              <span className="text-cyan-300 font-bold text-sm">
                                {(gc.core_concentration_ratio * 100).toFixed(1)}%
                              </span>
                            </div>

                            <div className="flex justify-between items-center p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
                              <span className="text-slate-400">Peak Spatial Activation:</span>
                              <span className="text-white font-semibold">
                                {gc.peak_activation.toFixed(4)}
                              </span>
                            </div>

                            <div className="text-xs text-slate-500 truncate">
                              Saliency SHA256: {gc.saliency_sha256}
                            </div>

                            {/* Causal Disclaimer */}
                            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-sm text-slate-400 font-sans leading-relaxed">
                              <strong className="text-amber-400">Scientific Disclaimer:</strong>{" "}
                              {gc.causal_disclaimer}
                            </div>
                          </div>
                        );
                      })()
                    ) : (
                      <div className="py-8 text-center text-slate-500 font-mono text-xs">
                        Grad-CAM spatial attribution not applicable for tabular-only baseline model.
                      </div>
                    )}
                  </div>

                  {/* Environmental Sensitivity Card */}
                  <div className="p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-3">
                    <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                      <h3 className="text-xs font-bold text-white flex items-center gap-2">
                        <IconActivity className="w-4 h-4 text-teal-400" />
                        <span>Environmental Feature Sensitivity</span>
                      </h3>
                      <span className="text-sm font-mono px-2 py-0.5 rounded bg-teal-500/10 text-teal-400">
                        Gradient × Input
                      </span>
                    </div>

                    {selectedJob.explainability?.environmental ? (
                      <div className="space-y-2.5 font-mono text-xs">
                        <div className="space-y-1.5">
                          {selectedJob.explainability.environmental.ranked_features.slice(0, 4).map(([feat, score]) => (
                            <div key={feat} className="space-y-1">
                              <div className="flex justify-between text-sm">
                                <span className="text-slate-300 truncate">{feat}</span>
                                <span className="text-teal-400 font-bold">{(score * 100).toFixed(1)}%</span>
                              </div>
                              <div className="h-1.5 w-full bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                                <div
                                  className="h-full bg-teal-500 rounded-full"
                                  style={{ width: `${Math.min(100, score * 100)}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>

                        {/* Causal Disclaimer */}
                        <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-sm text-slate-400 font-sans leading-relaxed mt-3">
                          <strong className="text-amber-400">Scientific Disclaimer:</strong>{" "}
                          {selectedJob.explainability.environmental.causal_disclaimer}
                        </div>
                      </div>
                    ) : (
                      <div className="py-8 text-center text-slate-500 font-mono text-xs">
                        Environmental sensitivity not applicable for image-only model.
                      </div>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <div className="py-24 text-center text-slate-500 font-mono text-xs">
                Select an analysis run to inspect results.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function AnalysisResultsPage() {
  return (
    <Suspense fallback={<div className="py-20 text-center text-sm font-mono text-slate-400">Loading Analysis Results...</div>}>
      <ResultsContent />
    </Suspense>
  );
}
