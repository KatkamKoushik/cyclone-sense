"use client";

import React, { useState, useEffect } from "react";
import {
  IconDatabase,
  IconSatellite,
  IconRefresh,
  IconAlert,
  IconShield,
  IconSparkles,
} from "@/components/icons";
import { api, SystemSettings } from "@/lib/api";

export default function SettingsPage() {
  const [settings, setSettings] = useState<SystemSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Adapter testing state
  const [testingAdapter, setTestingAdapter] = useState<string | null>(null);
  const [testResults, setTestResults] = useState<Record<string, { status: string; message: string }>>({});

  // Real-time satellite ingestion state
  const [selectedProvider, setSelectedProvider] = useState<"NOAA_GOES" | "NASA_EARTHDATA">("NOAA_GOES");
  const [searchLoading, setSearchLoading] = useState(false);
  const [discoveredGranules, setDiscoveredGranules] = useState<Array<{
    key?: string;
    granule_id?: string;
    title?: string;
    filename?: string;
    download_url: string;
    size_mb: number;
    time_start?: string;
  }>>([]);
  const [ingestingId, setIngestingId] = useState<string | null>(null);
  const [ingestionStatus, setIngestionStatus] = useState<string | null>(null);

  const fetchSettings = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getSystemSettings();
      setSettings(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load system settings from backend");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSettings();
  }, []);

  const handleTestAdapter = async (name: string) => {
    setTestingAdapter(name);
    try {
      const res = await api.testAdapter(name);
      setTestResults((prev) => ({ ...prev, [name]: res }));
    } catch (err: unknown) {
      setTestResults((prev) => ({
        ...prev,
        [name]: { status: "ERROR", message: err instanceof Error ? err.message : "Test failed" },
      }));
    } finally {
      setTestingAdapter(null);
    }
  };

  const handleSearchRealtime = async () => {
    setSearchLoading(true);
    setIngestionStatus(null);
    try {
      const res = await api.searchRealtimeGranules(selectedProvider, undefined, 4);
      setDiscoveredGranules(res.granules);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Real-time search failed");
    } finally {
      setSearchLoading(false);
    }
  };

  const handleIngestGranule = async (identifier: string) => {
    setIngestingId(identifier);
    setIngestionStatus(null);
    try {
      const res = await api.fetchRealtimeGranule(selectedProvider, identifier);
      setIngestionStatus(
        `Successfully ingested ${res.filename} | QC: ${res.qc_evaluations?.[0]?.passed ? "PASSED" : "FLAGGED"} | SHA-256: ${res.sha256_hash.substring(0, 16)}...`
      );
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Real-time granule ingestion failed");
    } finally {
      setIngestingId(null);
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 pb-2 border-b border-slate-800/80">
        <div>
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-white flex items-center gap-3">
            <span>Settings & Satellite Source Integrations</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
              Live Production Configuration
            </span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Audit external satellite data adapters, credentials permissions, and real-time scientific streaming channels.
          </p>
        </div>

        <button
          onClick={fetchSettings}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700 text-slate-300 transition-colors disabled:opacity-50"
        >
          <IconRefresh className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          <span>Refresh Settings</span>
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-center gap-3">
          <IconAlert className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {loading ? (
        <div className="py-20 text-center text-slate-400 font-mono text-xs space-y-2">
          <IconRefresh className="w-6 h-6 animate-spin mx-auto text-cyan-400" />
          <p>Loading operational configuration...</p>
        </div>
      ) : settings ? (
        <div className="space-y-6">
          {/* External Satellite Source Adapters */}
          <div className="p-6 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-4">
            <h2 className="text-lg font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-3">
              <IconSatellite className="w-4 h-4 text-cyan-400" />
              <span>External Earth Observation & Satellite Adapters</span>
            </h2>

            <div className="grid grid-cols-1 gap-4 font-mono text-xs">
              {/* NASA Earthdata Cloud */}
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-white text-sm">NASA Earthdata Cloud (MODIS, VIIRS, GPM)</span>
                    <span className="px-2 py-0.5 rounded text-xs bg-emerald-500/20 text-emerald-400 font-semibold">
                      BEARER TOKEN ACTIVE
                    </span>
                  </div>
                  <p className="text-sm text-slate-400 font-sans">
                    Authenticated access to NASA CMR API and Earthdata Cloud granules (Username: <code className="text-cyan-300">koushik_katkam</code>).
                  </p>
                  {testResults["nasa_earthdata"] && (
                    <div className="mt-2 text-sm text-cyan-300 font-mono">
                      Result: {testResults["nasa_earthdata"].message}
                    </div>
                  )}
                </div>

                <button
                  onClick={() => handleTestAdapter("nasa_earthdata")}
                  disabled={testingAdapter === "nasa_earthdata"}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold shrink-0 transition-colors disabled:opacity-50"
                >
                  {testingAdapter === "nasa_earthdata" ? "Verifying Token..." : "Verify NASA Auth"}
                </button>
              </div>

              {/* NOAA GOES */}
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-white text-sm">NOAA GOES-R (AWS Open Data)</span>
                    <span className="px-2 py-0.5 rounded text-xs bg-cyan-500/20 text-cyan-400 font-semibold">
                      PUBLIC S3 HTTPS ACTIVE
                    </span>
                  </div>
                  <p className="text-sm text-slate-400 font-sans">
                    Direct unsigned HTTPS retrieval from AWS public open-data bucket (<code className="text-cyan-300">noaa-goes16.s3.amazonaws.com</code>).
                  </p>
                  {testResults["noaa_goes"] && (
                    <div className="mt-2 text-sm text-cyan-300 font-mono">
                      Result: {testResults["noaa_goes"].message}
                    </div>
                  )}
                </div>

                <button
                  onClick={() => handleTestAdapter("noaa_goes")}
                  disabled={testingAdapter === "noaa_goes"}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold shrink-0 transition-colors disabled:opacity-50"
                >
                  {testingAdapter === "noaa_goes" ? "Checking S3..." : "Test S3 Reachability"}
                </button>
              </div>

              {/* NOAA IBTrACS */}
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-white text-sm">NOAA IBTrACS (North Indian Ocean)</span>
                    <span className="px-2 py-0.5 rounded text-xs bg-emerald-500/20 text-emerald-400 font-semibold">
                      LOCAL ARCHIVE ACTIVE
                    </span>
                  </div>
                  <p className="text-sm text-slate-400 font-sans">
                    Authentic NetCDF4 CF-compliant archive with 1,859 historical storms (1842–2026) verified with ground-truth intensity and pressures.
                  </p>
                  {testResults["noaa_ibtracs"] && (
                    <div className="mt-2 text-sm text-cyan-300 font-mono">
                      Result: {testResults["noaa_ibtracs"].message}
                    </div>
                  )}
                </div>

                <button
                  onClick={() => handleTestAdapter("noaa_ibtracs")}
                  disabled={testingAdapter === "noaa_ibtracs"}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold shrink-0 transition-colors disabled:opacity-50"
                >
                  {testingAdapter === "noaa_ibtracs" ? "Verifying..." : "Verify Archive"}
                </button>
              </div>

              {/* ISRO INSAT / MOSDAC */}
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-white text-sm">ISRO INSAT-3D / 3DR (MOSDAC)</span>
                    <span className="px-2 py-0.5 rounded text-xs bg-amber-500/20 text-amber-400 font-semibold">
                      ACCOUNT REGISTRATION PENDING
                    </span>
                  </div>
                  <p className="text-sm text-slate-400 font-sans">
                    Request for account creation submitted (<code className="text-amber-300">koushik_katkam</code>). Awaiting MOSDAC administrator SSO verification.
                  </p>
                  {testResults["isro_insat"] && (
                    <div className="mt-2 text-sm text-amber-300 font-mono">
                      Result: {testResults["isro_insat"].message}
                    </div>
                  )}
                </div>

                <button
                  onClick={() => handleTestAdapter("isro_insat")}
                  disabled={testingAdapter === "isro_insat"}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold shrink-0 transition-colors disabled:opacity-50"
                >
                  {testingAdapter === "isro_insat" ? "Testing MOSDAC..." : "Test Status"}
                </button>
              </div>
            </div>
          </div>

          {/* Real-Time Live Satellite Acquisition Console */}
          <div className="p-6 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-800 pb-3">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <IconSparkles className="w-5 h-5 text-cyan-400" />
                  <span>Real-Time Satellite Granule Acquisition Console</span>
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Stream authentic, live observation granules directly from NOAA AWS S3 or NASA Earthdata into CycloneSense.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <select
                  value={selectedProvider}
                  onChange={(e) => {
                    setSelectedProvider(e.target.value as "NOAA_GOES" | "NASA_EARTHDATA");
                    setDiscoveredGranules([]);
                  }}
                  className="px-3 py-1.5 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white font-mono"
                >
                  <option value="NOAA_GOES">NOAA GOES-16 (AWS S3)</option>
                  <option value="NASA_EARTHDATA">NASA Earthdata (MODIS 250m)</option>
                </select>

                <button
                  onClick={handleSearchRealtime}
                  disabled={searchLoading}
                  className="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-semibold transition-colors disabled:opacity-50 flex items-center gap-1.5"
                >
                  <IconRefresh className={`w-3.5 h-3.5 ${searchLoading ? "animate-spin" : ""}`} />
                  <span>Search Live Granules</span>
                </button>
              </div>
            </div>

            {ingestionStatus && (
              <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs font-mono">
                {ingestionStatus}
              </div>
            )}

            {discoveredGranules.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400">
                      <th className="pb-2">Granule Title / Key</th>
                      <th className="pb-2 text-right">Size (MB)</th>
                      <th className="pb-2 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-300">
                    {discoveredGranules.map((g, idx) => {
                      const id = g.key || g.download_url || g.title || String(idx);
                      const isIngesting = ingestingId === id;
                      return (
                        <tr key={idx} className="hover:bg-slate-800/20">
                          <td className="py-2.5 truncate max-w-[420px] font-semibold text-white">
                            {g.title || g.filename || g.key}
                          </td>
                          <td className="py-2.5 text-right text-slate-400">{g.size_mb} MB</td>
                          <td className="py-2.5 text-right">
                            <button
                              onClick={() => handleIngestGranule(id)}
                              disabled={isIngesting}
                              className="px-3 py-1 bg-teal-600 hover:bg-teal-500 text-white rounded-md text-xs font-semibold disabled:opacity-50 transition-colors"
                            >
                              {isIngesting ? "Fetching..." : "Fetch & Ingest"}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="py-6 text-center text-slate-500 text-xs font-mono">
                Click &quot;Search Live Granules&quot; to discover authentic satellite products from external providers.
              </div>
            )}
          </div>

          {/* Scientific Storage & Directories */}
          <div className="p-6 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-4">
            <h2 className="text-lg font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-3">
              <IconDatabase className="w-4 h-4 text-teal-400" />
              <span>Scientific Data Repositories & Storage</span>
            </h2>

            <div className="space-y-3 font-mono text-xs">
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-400">Raw Products Directory:</span>
                <span className="text-white truncate max-w-[340px]">{settings.storage.data_raw_dir}</span>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-400">Model Checkpoints Directory:</span>
                <span className="text-white truncate max-w-[340px]">{settings.storage.checkpoints_dir}</span>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-400">Active Database Dialect:</span>
                <span className="text-cyan-400 font-bold">{settings.database_backend}</span>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
