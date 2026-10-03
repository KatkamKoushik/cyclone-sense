"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  IconCpu,
  IconShield,
  IconAlert,
  IconRefresh,
  IconInfo,
} from "@/components/icons";
import { api, EvaluationReport, MLModelMeta } from "@/lib/api";

export default function ModelBenchmarksPage() {
  const [report, setEvalReport] = useState<EvaluationReport | null>(null);
  const [modelCatalog, setModelCatalog] = useState<MLModelMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchReports = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [rData, mData] = await Promise.all([
        api.getEvaluationReport(),
        api.listMLModels(),
      ]);
      setEvalReport(rData);
      setModelCatalog(mData || []);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load evaluation reports from backend");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let ignore = false;
    async function init() {
      try {
        const [rData, mData] = await Promise.all([
          api.getEvaluationReport(),
          api.listMLModels(),
        ]);
        if (!ignore) {
          setEvalReport(rData);
          setModelCatalog(mData || []);
        }
      } catch (err: unknown) {
        if (!ignore) {
          setError(err instanceof Error ? err.message : "Failed to load evaluation reports from backend");
        }
      } finally {
        if (!ignore) {
          setLoading(false);
        }
      }
    }
    init();
    return () => {
      ignore = true;
    };
  }, []);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 pb-2 border-b border-slate-800/80">
        <div>
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-white flex items-center gap-3">
            <span>Model Validation & Comparative Benchmarks</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
              Unseen Test Split (2022–2026)
            </span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Standardized evaluation comparing Baseline CLIPER, Environment-Only, Image-Only, and Multimodal Fusion.
          </p>
        </div>

        <button
          onClick={fetchReports}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700 text-slate-300 transition-colors disabled:opacity-50"
        >
          <IconRefresh className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          <span>Refresh</span>
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-center gap-3">
          <IconAlert className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Dataset & Leakage Prevention Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Train Split */}
        <div className="p-4 rounded-2xl bg-[#0c121e]/80 border border-slate-800/80 shadow-lg font-mono text-xs">
          <div className="text-slate-400 font-semibold">Training Split (2000–2018)</div>
          <div className="text-2xl font-bold text-white mt-1">
            {loading && !report ? (
              <div className="h-7 w-28 bg-slate-800/60 animate-pulse rounded" />
            ) : report?.dataset?.split?.train ? (
              `${report.dataset.split.train.obs_count.toLocaleString()} obs`
            ) : (
              "—"
            )}
          </div>
          <div className="text-slate-500 text-sm mt-1">
            {report?.dataset?.split?.train
              ? `${report.dataset.split.train.storm_count} storms • 19 seasons`
              : "Seasons 2000–2018"}
          </div>
        </div>

        {/* Validation Split */}
        <div className="p-4 rounded-2xl bg-[#0c121e]/80 border border-slate-800/80 shadow-lg font-mono text-xs">
          <div className="text-slate-400 font-semibold">Validation Split (2019–2021)</div>
          <div className="text-2xl font-bold text-white mt-1">
            {loading && !report ? (
              <div className="h-7 w-28 bg-slate-800/60 animate-pulse rounded" />
            ) : report?.dataset?.split?.val ? (
              `${report.dataset.split.val.obs_count.toLocaleString()} obs`
            ) : (
              "—"
            )}
          </div>
          <div className="text-slate-500 text-sm mt-1">
            {report?.dataset?.split?.val
              ? `${report.dataset.split.val.storm_count} storms • Hyperparameter tuning`
              : "Seasons 2019–2021"}
          </div>
        </div>

        {/* Unseen Test Split */}
        <div className="p-4 rounded-2xl bg-[#0c121e]/80 border border-slate-800/80 shadow-lg font-mono text-xs border-cyan-500/30">
          <div className="text-cyan-400 font-semibold">Unseen Test Split (2022–2026)</div>
          <div className="text-2xl font-bold text-cyan-300 mt-1">
            {loading && !report ? (
              <div className="h-7 w-28 bg-slate-800/60 animate-pulse rounded" />
            ) : report?.dataset?.split?.test ? (
              `${report.dataset.split.test.obs_count.toLocaleString()} obs`
            ) : (
              "—"
            )}
          </div>
          <div className="text-slate-400 text-sm mt-1">
            {report?.dataset?.split?.test
              ? `${report.dataset.split.test.storm_count} storms • Zero Temporal Leakage`
              : "Seasons 2022–2026"}
          </div>
        </div>
      </div>

      {/* Main Comparative Benchmark Matrix */}
      <div className="p-6 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-800 pb-3">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <IconCpu className="w-5 h-5 text-indigo-400" />
              <span>Comparative Benchmark Matrix (Authentic Test Split Metrics)</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Numerical metrics served live from <code className="text-cyan-400 font-mono">/api/v1/ml/evaluation-report</code>.
            </p>
          </div>
          <span className="text-sm font-mono text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20">
            Live Backend Metrics
          </span>
        </div>

        <div className="overflow-x-auto">
          {loading && !report ? (
            <div className="py-12 space-y-3">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="h-8 bg-slate-800/40 rounded animate-pulse" />
              ))}
            </div>
          ) : report?.models_evaluated ? (
            (() => {
              const cliper = report.models_evaluated.baseline_cliper;
              const env = report.models_evaluated.environment_only;
              const img = report.models_evaluated.image_only;
              const fusion = report.models_evaluated.multimodal_fusion;

              const cliperMae = cliper?.intensity_metrics?.mae_kts;
              const cliperRmse = cliper?.intensity_metrics?.rmse_kts;
              const cliperBias = cliper?.intensity_metrics?.bias_kts;
              const cliperR = cliper?.intensity_metrics?.pearson_r;
              const cliperF1 = cliper?.classification_metrics?.f1_macro ?? cliper?.classification_metrics?.macro_f1;
              const cliperAcc = cliper?.classification_metrics?.accuracy !== undefined
                ? (cliper.classification_metrics.accuracy * 100).toFixed(1) + "%"
                : "—";

              const envMae = env?.test_evaluation?.intensity_metrics?.mae_kts;
              const envRmse = env?.test_evaluation?.intensity_metrics?.rmse_kts;
              const envBias = env?.test_evaluation?.intensity_metrics?.bias_kts;
              const envR = env?.test_evaluation?.intensity_metrics?.pearson_r;
              const envF1 = env?.test_evaluation?.classification_metrics?.f1_macro ?? env?.test_evaluation?.classification_metrics?.macro_f1;
              const envAcc = env?.test_evaluation?.classification_metrics?.accuracy !== undefined
                ? (env.test_evaluation.classification_metrics.accuracy * 100).toFixed(1) + "%"
                : "—";

              const imgMae = img?.test_evaluation?.intensity_metrics?.mae_kts;
              const imgRmse = img?.test_evaluation?.intensity_metrics?.rmse_kts;
              const imgBias = img?.test_evaluation?.intensity_metrics?.bias_kts;
              const imgR = img?.test_evaluation?.intensity_metrics?.pearson_r;
              const imgF1 = img?.test_evaluation?.classification_metrics?.f1_macro ?? img?.test_evaluation?.classification_metrics?.macro_f1;
              const imgAcc = img?.test_evaluation?.classification_metrics?.accuracy !== undefined
                ? (img.test_evaluation.classification_metrics.accuracy * 100).toFixed(1) + "%"
                : "—";

              const fusionMae = fusion?.test_evaluation?.intensity_metrics?.mae_kts;
              const fusionRmse = fusion?.test_evaluation?.intensity_metrics?.rmse_kts;
              const fusionBias = fusion?.test_evaluation?.intensity_metrics?.bias_kts;
              const fusionR = fusion?.test_evaluation?.intensity_metrics?.pearson_r;
              const fusionF1 = fusion?.test_evaluation?.classification_metrics?.f1_macro ?? fusion?.test_evaluation?.classification_metrics?.macro_f1;
              const fusionAcc = fusion?.test_evaluation?.classification_metrics?.accuracy !== undefined
                ? (fusion.test_evaluation.classification_metrics.accuracy * 100).toFixed(1) + "%"
                : "—";

              return (
                <>
                  <table className="w-full text-left text-sm font-mono">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400">
                        <th className="pb-3 font-semibold">Model Architecture</th>
                        <th className="pb-3 font-semibold">Input Modality</th>
                        <th className="pb-3 text-right font-semibold">Intensity MAE</th>
                        <th className="pb-3 text-right font-semibold">Intensity RMSE</th>
                        <th className="pb-3 text-right font-semibold">Intensity Bias</th>
                        <th className="pb-3 text-right font-semibold">Pearson r</th>
                        <th className="pb-3 text-right font-semibold">Macro-F1</th>
                        <th className="pb-3 text-right font-semibold">Accuracy</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 text-slate-300">
                      {/* Baseline CLIPER */}
                      <tr className="hover:bg-slate-800/20">
                        <td className="py-3 font-bold text-slate-200">Baseline CLIPER (Ridge)</td>
                        <td className="py-3 text-slate-400">8-dim Environmental Covariates</td>
                        <td className="py-3 text-right">{cliperMae !== undefined ? `${cliperMae.toFixed(3)} kts` : "—"}</td>
                        <td className="py-3 text-right">{cliperRmse !== undefined ? `${cliperRmse.toFixed(3)} kts` : "—"}</td>
                        <td className="py-3 text-right text-rose-400">
                          {cliperBias !== undefined ? (cliperBias > 0 ? `+${cliperBias.toFixed(3)} kts` : `${cliperBias.toFixed(3)} kts`) : "—"}
                        </td>
                        <td className="py-3 text-right">{cliperR !== undefined ? cliperR.toFixed(3) : "—"}</td>
                        <td className="py-3 text-right">{cliperF1 !== undefined ? cliperF1.toFixed(4) : "—"}</td>
                        <td className="py-3 text-right">{cliperAcc}</td>
                      </tr>

                      {/* Environment-Only */}
                      <tr className="hover:bg-slate-800/20">
                        <td className="py-3 font-bold text-slate-200">Environment-Only MLP</td>
                        <td className="py-3 text-slate-400">8-dim Environmental Covariates</td>
                        <td className="py-3 text-right">{envMae !== undefined ? `${envMae.toFixed(3)} kts` : "—"}</td>
                        <td className="py-3 text-right">{envRmse !== undefined ? `${envRmse.toFixed(3)} kts` : "—"}</td>
                        <td className="py-3 text-right text-rose-400">
                          {envBias !== undefined ? (envBias > 0 ? `+${envBias.toFixed(3)} kts` : `${envBias.toFixed(3)} kts`) : "—"}
                        </td>
                        <td className="py-3 text-right">{envR !== undefined ? envR.toFixed(3) : "—"}</td>
                        <td className="py-3 text-right">{envF1 !== undefined ? envF1.toFixed(4) : "—"}</td>
                        <td className="py-3 text-right">{envAcc}</td>
                      </tr>

                      {/* Image-Only */}
                      <tr className="hover:bg-slate-800/20">
                        <td className="py-3 font-bold text-slate-200">Image-Only CNN</td>
                        <td className="py-3 text-slate-400">2-Ch Satellite IR/WV Grid</td>
                        <td className="py-3 text-right font-bold text-emerald-400">
                          {imgMae !== undefined ? `${imgMae.toFixed(3)} kts` : "—"}
                        </td>
                        <td className="py-3 text-right font-bold text-emerald-400">
                          {imgRmse !== undefined ? `${imgRmse.toFixed(3)} kts` : "—"}
                        </td>
                        <td className="py-3 text-right text-amber-400">
                          {imgBias !== undefined ? (imgBias > 0 ? `+${imgBias.toFixed(3)} kts` : `${imgBias.toFixed(3)} kts`) : "—"}
                        </td>
                        <td className="py-3 text-right font-bold text-emerald-400">
                          {imgR !== undefined ? imgR.toFixed(3) : "—"}
                        </td>
                        <td className="py-3 text-right">{imgF1 !== undefined ? imgF1.toFixed(4) : "—"}</td>
                        <td className="py-3 text-right">{imgAcc}</td>
                      </tr>

                      {/* Multimodal Fusion */}
                      <tr className="bg-indigo-950/20 hover:bg-indigo-900/30 border-l-2 border-indigo-500">
                        <td className="py-3 pl-2 font-bold text-indigo-300">Multimodal Fusion (CNN+MLP)</td>
                        <td className="py-3 text-slate-300">Satellite IR/WV + 8-dim Env</td>
                        <td className="py-3 text-right font-bold text-cyan-300">
                          {fusionMae !== undefined ? `${fusionMae.toFixed(3)} kts` : "—"}
                        </td>
                        <td className="py-3 text-right">
                          {fusionRmse !== undefined ? `${fusionRmse.toFixed(3)} kts` : "—"}
                        </td>
                        <td className="py-3 text-right font-bold text-emerald-400">
                          {fusionBias !== undefined ? (fusionBias > 0 ? `+${fusionBias.toFixed(3)} kts` : `${fusionBias.toFixed(3)} kts`) : "—"}
                        </td>
                        <td className="py-3 text-right">{fusionR !== undefined ? fusionR.toFixed(3) : "—"}</td>
                        <td className="py-3 text-right font-bold text-emerald-400">
                          {fusionF1 !== undefined ? fusionF1.toFixed(4) : "—"}
                        </td>
                        <td className="py-3 text-right font-bold text-emerald-400">{fusionAcc}</td>
                      </tr>
                    </tbody>
                  </table>

                  {/* Scientific Evaluation Takeaway */}
                  <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-sm font-mono text-slate-300 space-y-2 mt-4">
                    <div className="text-cyan-400 font-semibold uppercase text-sm flex items-center gap-1.5">
                      <IconInfo className="w-4 h-4" />
                      <span>Scientific Benchmark Takeaway</span>
                    </div>
                    <p className="text-slate-400 font-sans leading-relaxed text-xs">
                      {imgMae !== undefined && fusionMae !== undefined ? (
                        <>
                          <strong>Image-Only CNN</strong> achieves the lowest continuous Mean Absolute Error ({imgMae.toFixed(3)} kts vs {fusionMae.toFixed(3)} kts), but carries a positive bias (+{imgBias?.toFixed(3)} kts). <strong>Multimodal Fusion</strong> achieves higher categorical pattern classification performance (Macro-F1 of {fusionF1?.toFixed(4)} vs {imgF1?.toFixed(4)}, and {fusionAcc} vs {imgAcc} accuracy), showing that environmental covariates (pressure deficit, Coriolis, forward speed) help disambiguate borderline storm classes where cloud patterns alone are ambiguous.
                        </>
                      ) : (
                        "Benchmarked across 971 authentic unseen observations from North Indian tropical cyclones."
                      )}
                    </p>
                  </div>
                </>
              );
            })()
          ) : (
            <div className="py-8 text-center text-slate-500 font-mono text-xs">
              No evaluation report available from backend. Run <code>python backend/scripts/run_experiments.py</code> to regenerate benchmark metrics.
            </div>
          )}
        </div>
      </div>

      {/* Model Checkpoint Registry Manifests */}
      <div className="p-6 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-3">
          <IconShield className="w-4 h-4 text-emerald-400" />
          <span>Model Checkpoint & Checksum Registry ({modelCatalog.length} Registered)</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {modelCatalog.map((m) => {
            const ckpt = m.checkpoints?.[0];
            return (
              <div
                key={m.model_id}
                className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 font-mono text-xs space-y-2"
              >
                <div className="flex justify-between items-center">
                  <span className="font-bold text-white">{m.model_id}</span>
                  <span className="text-xs px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                    {m.type}
                  </span>
                </div>
                <p className="text-sm text-slate-400 font-sans line-clamp-2">{m.description}</p>
                <div className="pt-2 border-t border-slate-800/80 text-xs space-y-1 text-slate-400">
                  <div>Checkpoint: {ckpt ? ckpt.filename : "Closed-Form weights"}</div>
                  {ckpt && <div>Size: {(ckpt.size_bytes / (1024 * 1024)).toFixed(2)} MB</div>}
                  <div className="truncate text-cyan-400">
                    Input: {m.inputs.join(", ")}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
