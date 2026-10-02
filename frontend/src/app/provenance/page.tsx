"use client";

import React, { useState, useEffect } from "react";
import {
  IconShield,
  IconRefresh,
  IconAlert,
  IconInfo,
  IconSearch,
} from "@/components/icons";
import { api, ProvenanceRecord } from "@/lib/api";

export default function ProvenanceAuditPage() {
  const [records, setRecords] = useState<ProvenanceRecord[]>([]);
  const [selectedRecord, setSelectedRecord] = useState<ProvenanceRecord | null>(null);
  const [searchHash, setSearchHash] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    api
      .listProvenance(50)
      .then((data) => {
        if (isMounted) {
          setRecords(data);
          if (data.length > 0) setSelectedRecord(data[0]);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err instanceof Error ? err.message : "Failed to load cryptographic provenance log");
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const fetchProvenance = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listProvenance(50);
      setRecords(data);
      if (data.length > 0) setSelectedRecord(data[0]);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load cryptographic provenance log");
    } finally {
      setLoading(false);
    }
  };

  const filtered = records.filter(
    (r) =>
      r.sha256_hash.toLowerCase().includes(searchHash.toLowerCase()) ||
      r.entity_id.toLowerCase().includes(searchHash.toLowerCase()) ||
      r.action.toLowerCase().includes(searchHash.toLowerCase())
  );

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 pb-2 border-b border-slate-800/80">
        <div>
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-white flex items-center gap-3">
            <span>Cryptographic Provenance & Audit Trail</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
              W3C PROV-O Standard
            </span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Immutable SHA-256 lineage tracking for raw NetCDF granules, extracted tensors, inference executions, and Grad-CAM saliency heatmaps.
          </p>
        </div>

        <button
          onClick={fetchProvenance}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700 text-slate-300 transition-colors disabled:opacity-50"
        >
          <IconRefresh className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          <span>Refresh Audit Trail</span>
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-center gap-3">
          <IconAlert className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Search Input */}
      <div className="p-4 rounded-2xl bg-[#0c121e]/80 border border-slate-800/80 shadow-lg">
        <div className="relative">
          <IconSearch className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search provenance by SHA-256 hash, entity ID, or action (e.g. MODEL_INFERENCE, STORM_EXTRACTION)..."
            value={searchHash}
            onChange={(e) => setSearchHash(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-xl text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono"
          />
        </div>
      </div>

      {/* 2-Column Grid: Event Table + Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Event List (7 cols) */}
        <div className="lg:col-span-7 p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <IconShield className="w-4 h-4 text-cyan-400" />
              <span>Audit Lineage Events</span>
            </h2>
            <span className="text-sm font-mono text-slate-400">{filtered.length} Recorded</span>
          </div>

          {loading ? (
            <div className="py-20 text-center text-slate-400 font-mono text-xs space-y-2">
              <IconRefresh className="w-6 h-6 animate-spin mx-auto text-cyan-400" />
              <p>Querying immutable database provenance records...</p>
            </div>
          ) : filtered.length === 0 ? (
            <div className="py-20 text-center text-slate-500 font-mono text-xs">
              No provenance records found.
            </div>
          ) : (
            <div className="overflow-x-auto max-h-[580px] overflow-y-auto">
              <table className="w-full text-left text-sm font-mono">
                <thead className="sticky top-0 bg-[#0c121e] border-b border-slate-800 text-slate-400">
                  <tr>
                    <th className="pb-2">Action</th>
                    <th className="pb-2">Entity Type</th>
                    <th className="pb-2">SHA-256 Digest</th>
                    <th className="pb-2 text-right">Timestamp</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {filtered.map((r) => {
                    const isSelected = selectedRecord?.id === r.id;
                    return (
                      <tr
                        key={r.id}
                        onClick={() => setSelectedRecord(r)}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? "bg-cyan-500/10 border-l-2 border-cyan-400"
                            : "hover:bg-slate-800/30"
                        }`}
                      >
                        <td className="py-2.5 font-bold text-white">
                          <span
                            className={`px-1.5 py-0.5 rounded text-xs ${
                              r.action === "MODEL_INFERENCE"
                                ? "bg-indigo-500/20 text-indigo-300"
                                : r.action === "STORM_EXTRACTION"
                                ? "bg-cyan-500/20 text-cyan-300"
                                : "bg-emerald-500/20 text-emerald-300"
                            }`}
                          >
                            {r.action}
                          </span>
                        </td>
                        <td className="py-2.5 text-slate-400">{r.entity_type}</td>
                        <td className="py-2.5 text-cyan-400 font-bold truncate max-w-[160px]">
                          {r.sha256_hash.substring(0, 16)}...
                        </td>
                        <td className="py-2.5 text-right text-slate-400 text-sm">
                          {new Date(r.timestamp).toLocaleTimeString()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Right Column: Record Detail Inspector (5 cols) */}
        <div className="lg:col-span-5 p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-4">
          <div className="border-b border-slate-800 pb-3">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <IconInfo className="w-4 h-4 text-teal-400" />
              <span>Lineage Metadata & Parameters</span>
            </h2>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              {selectedRecord ? `Entity ID: ${selectedRecord.entity_id}` : "Select a record"}
            </p>
          </div>

          {selectedRecord ? (
            <div className="space-y-4 font-mono text-xs">
              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
                <div>
                  <span className="text-xs text-slate-400 block">Cryptographic Digest (SHA-256)</span>
                  <span className="text-xs text-cyan-300 font-bold break-all select-all">
                    {selectedRecord.sha256_hash}
                  </span>
                </div>
                <div className="pt-2 border-t border-slate-800 flex justify-between text-sm">
                  <span className="text-slate-400">Software Version:</span>
                  <span className="text-white">{selectedRecord.software_version}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Execution Time:</span>
                  <span className="text-white">{new Date(selectedRecord.timestamp).toISOString()}</span>
                </div>
              </div>

              {/* JSON Parameters */}
              <div className="space-y-1.5">
                <span className="text-sm font-bold text-slate-300">Logged Action Parameters:</span>
                <pre className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 text-sm text-slate-300 overflow-x-auto max-h-[360px] leading-relaxed">
                  {JSON.stringify(selectedRecord.parameters, null, 2)}
                </pre>
              </div>
            </div>
          ) : (
            <div className="py-24 text-center text-slate-500 font-mono text-xs">
              Select an audit record to inspect parameters.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
