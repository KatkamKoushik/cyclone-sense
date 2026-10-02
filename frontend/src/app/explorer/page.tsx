"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  IconSearch,
  IconRefresh,
  IconAlert,
  IconPlay,
  IconCompare,
  IconInfo,
} from "@/components/icons";
import { api, StormSummary, StormTrackResponse, StormTrackObservation } from "@/lib/api";

const CATEGORY_NAMES = [
  "Tropical Depression (< 34 kts)",
  "Cyclonic Storm (34-63 kts)",
  "Very Severe Cyclonic Storm (64-89 kts)",
  "Extremely Severe Cyclonic Storm (90-119 kts)",
  "Super Cyclonic Storm (>= 120 kts)",
];

const CATEGORY_COLORS = [
  "bg-blue-500/10 text-blue-400 border-blue-500/20",
  "bg-teal-500/10 text-teal-400 border-teal-500/20",
  "bg-amber-500/10 text-amber-400 border-amber-500/20",
  "bg-orange-500/10 text-orange-400 border-orange-500/20",
  "bg-rose-500/10 text-rose-400 border-rose-500/20",
];

export default function CycloneExplorerPage() {
  const [storms, setStorms] = useState<StormSummary[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [searchQuery, setSearchQuery] = useState("");
  const [seasonFilter, setSeasonFilter] = useState<number | undefined>(undefined);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Selected storm track
  const [selectedStorm, setSelectedStorm] = useState<StormSummary | null>(null);
  const [trackData, setTrackData] = useState<StormTrackResponse | null>(null);
  const [loadingTrack, setLoadingTrack] = useState(false);
  const [trackError, setTrackError] = useState<string | null>(null);

  const fetchCatalog = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getStormCatalog({
        search: searchQuery || undefined,
        season: seasonFilter,
        limit: 100,
      });
      setStorms(res.storms);
      setTotalCount(res.total_storms);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load cyclone catalog");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let isMounted = true;
    api
      .getStormCatalog({
        search: searchQuery || undefined,
        season: seasonFilter,
        limit: 100,
      })
      .then((res) => {
        if (isMounted) {
          setStorms(res.storms);
          setTotalCount(res.total_storms);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err instanceof Error ? err.message : "Failed to load cyclone catalog");
          setLoading(false);
        }
      });
    return () => {
      isMounted = false;
    };
  }, [seasonFilter, searchQuery]);

  const handleSelectStorm = async (storm: StormSummary) => {
    setSelectedStorm(storm);
    setLoadingTrack(true);
    setTrackError(null);
    try {
      const track = await api.getStormTrack(storm.storm_id);
      setTrackData(track);
    } catch (err: unknown) {
      setTrackError(err instanceof Error ? err.message : "Failed to load storm track");
    } finally {
      setLoadingTrack(false);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 pb-2 border-b border-slate-800/80">
        <div>
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-white flex items-center gap-3">
            <span>Cyclone Explorer & Historical Catalog</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
              NOAA IBTrACS v04r01
            </span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Browse authentic tropical cyclone tracks and validated physical observations in the North Indian Ocean basin.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={fetchCatalog}
            disabled={loading}
            className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700 text-slate-300 transition-colors disabled:opacity-50"
          >
            <IconRefresh className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="p-4 rounded-2xl bg-[#0c121e]/80 border border-slate-800/80 shadow-lg flex flex-col sm:flex-row items-center gap-3">
        <div className="relative flex-1 w-full">
          <IconSearch className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search cyclone by name (e.g. AMPHAN, FANI, BIPARJOY) or ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && fetchCatalog()}
            className="w-full pl-9 pr-4 py-2 bg-slate-900/80 border border-slate-700/80 rounded-xl text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono"
          />
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          <select
            value={seasonFilter || ""}
            onChange={(e) => setSeasonFilter(e.target.value ? parseInt(e.target.value) : undefined)}
            className="w-full sm:w-40 py-2 px-3 bg-slate-900/80 border border-slate-700/80 rounded-xl text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
          >
            <option value="">All Seasons</option>
            <option value="2026">Season 2026</option>
            <option value="2025">Season 2025</option>
            <option value="2024">Season 2024</option>
            <option value="2023">Season 2023</option>
            <option value="2022">Season 2022</option>
            <option value="2021">Season 2021</option>
            <option value="2020">Season 2020</option>
            <option value="2019">Season 2019</option>
            <option value="2018">Season 2018</option>
          </select>

          <button
            onClick={fetchCatalog}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl text-xs font-semibold shrink-0 transition-colors"
          >
            Search
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-center gap-3">
          <IconAlert className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Main 2-Column Grid: Catalog Table + Track Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Storms List Table (7 cols on lg) */}
        <div className="lg:col-span-7 p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
            <div>
              <h2 className="text-lg font-bold text-white">
                Historical Storms ({totalCount} Available)
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Click any storm row to inspect its authentic chronological observation track.
              </p>
            </div>
            <span className="text-sm font-mono text-cyan-400">
              Showing {storms.length}
            </span>
          </div>

          {loading ? (
            <div className="py-16 text-center text-slate-400 font-mono text-xs space-y-2">
              <IconRefresh className="w-6 h-6 animate-spin mx-auto text-cyan-400" />
              <p>Querying authentic IBTrACS NetCDF archive...</p>
            </div>
          ) : storms.length === 0 ? (
            <div className="py-16 text-center text-slate-400 font-mono text-xs">
              No tropical cyclones found matching criteria.
            </div>
          ) : (
            <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
              <table className="w-full text-left text-sm font-mono">
                <thead className="sticky top-0 bg-[#0c121e] border-b border-slate-800 text-slate-400">
                  <tr>
                    <th className="pb-2">Season</th>
                    <th className="pb-2">Name</th>
                    <th className="pb-2">Storm ID</th>
                    <th className="pb-2 text-right">Peak Wind</th>
                    <th className="pb-2 text-right">Min Pressure</th>
                    <th className="pb-2">Category</th>
                    <th className="pb-2 text-right">Obs</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {storms.map((s) => {
                    const isSelected = selectedStorm?.storm_id === s.storm_id;
                    return (
                      <tr
                        key={s.storm_id}
                        onClick={() => handleSelectStorm(s)}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? "bg-cyan-500/10 border-l-2 border-cyan-400"
                            : "hover:bg-slate-800/30"
                        }`}
                      >
                        <td className="py-2.5 font-bold text-slate-400">{s.season}</td>
                        <td className="py-2.5 font-bold text-white flex items-center gap-1.5">
                          <span>{s.storm_name}</span>
                        </td>
                        <td className="py-2.5 text-slate-400 truncate max-w-[100px]">{s.storm_id}</td>
                        <td className="py-2.5 text-right font-bold text-cyan-300">
                          {s.peak_intensity_kts.toFixed(0)} kts
                        </td>
                        <td className="py-2.5 text-right text-slate-400">
                          {s.min_pressure_hpa ? `${s.min_pressure_hpa.toFixed(0)} hPa` : "—"}
                        </td>
                        <td className="py-2.5">
                          <span
                            className={`px-1.5 py-0.5 rounded text-xs font-semibold border ${
                              CATEGORY_COLORS[s.peak_category] || "bg-slate-800 text-slate-400"
                            }`}
                          >
                            Cat {s.peak_category}
                          </span>
                        </td>
                        <td className="py-2.5 text-right text-slate-400">{s.obs_count}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Track Inspector Drawer (5 cols on lg) */}
        <div className="lg:col-span-5 p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <span>Observation Track Inspector</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                {selectedStorm
                  ? `${selectedStorm.storm_name} (Season ${selectedStorm.season})`
                  : "Select a cyclone to view chronological observations"}
              </p>
            </div>
            {selectedStorm && (
              <div className="flex items-center gap-1.5">
                <Link
                  href={`/temporal?storm_id=${selectedStorm.storm_id}`}
                  className="p-1.5 rounded-lg bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-300 border border-indigo-500/20 text-sm font-mono flex items-center gap-1"
                  title="Compare T1 vs T2"
                >
                  <IconCompare className="w-4 h-4" />
                  <span>T1/T2</span>
                </Link>
              </div>
            )}
          </div>

          {!selectedStorm ? (
            <div className="py-28 text-center text-slate-500 font-mono text-xs space-y-2">
              <IconInfo className="w-8 h-8 text-slate-600 mx-auto" />
              <p>Click any storm from the catalog table on the left to inspect its authentic chronological track.</p>
            </div>
          ) : loadingTrack ? (
            <div className="py-28 text-center text-slate-400 font-mono text-xs space-y-2">
              <IconRefresh className="w-6 h-6 animate-spin mx-auto text-cyan-400" />
              <p>Loading authentic track observations for {selectedStorm.storm_name}...</p>
            </div>
          ) : trackError ? (
            <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs">
              {trackError}
            </div>
          ) : trackData ? (
            <div className="space-y-4">
              {/* Summary Metrics */}
              <div className="grid grid-cols-3 gap-2 text-sm font-mono">
                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="text-xs text-slate-400">Total Timesteps</div>
                  <div className="text-lg font-bold text-white">{trackData.total_observations}</div>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="text-xs text-slate-400">Peak Intensity</div>
                  <div className="text-lg font-bold text-cyan-300">
                    {selectedStorm.peak_intensity_kts.toFixed(0)} kts
                  </div>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="text-xs text-slate-400">Classification</div>
                  <div className="text-xs font-semibold text-emerald-400 truncate mt-1">
                    {CATEGORY_NAMES[selectedStorm.peak_category].split("(")[0]}
                  </div>
                </div>
              </div>

              {/* Observations Track Table */}
              <div className="overflow-x-auto max-h-[460px] overflow-y-auto">
                <table className="w-full text-left text-sm font-mono">
                  <thead className="sticky top-0 bg-[#0c121e] border-b border-slate-800 text-slate-400">
                    <tr>
                      <th className="pb-1.5">UTC Time</th>
                      <th className="pb-1.5">Coords (Lat, Lon)</th>
                      <th className="pb-1.5 text-right">Wind</th>
                      <th className="pb-1.5 text-right">Pres</th>
                      <th className="pb-1.5 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-300">
                    {trackData.observations.map((obs: StormTrackObservation) => (
                      <tr key={obs.index} className="hover:bg-slate-800/30">
                        <td className="py-2 text-slate-400 truncate max-w-[110px]">
                          {obs.timestamp.replace("T", " ").substring(5, 16)}
                        </td>
                        <td className="py-2 text-slate-200">
                          {obs.latitude.toFixed(1)}°N, {obs.longitude.toFixed(1)}°E
                        </td>
                        <td className="py-2 text-right font-bold text-cyan-300">
                          {obs.wind_kts.toFixed(0)} kts
                        </td>
                        <td className="py-2 text-right text-slate-400">
                          {obs.pressure_hpa ? `${obs.pressure_hpa.toFixed(0)}` : "—"}
                        </td>
                        <td className="py-2 text-right">
                          <Link
                            href={`/analysis?storm_id=${encodeURIComponent(
                              selectedStorm.storm_id
                            )}&storm_name=${encodeURIComponent(
                              selectedStorm.storm_name
                            )}&lat=${obs.latitude}&lon=${obs.longitude}&wind=${obs.wind_kts}&pres=${
                              obs.pressure_hpa || 980
                            }&speed=${obs.forward_speed_kmh}&bearing=${obs.forward_bearing_deg}`}
                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 text-xs font-semibold"
                          >
                            <IconPlay className="w-2.5 h-2.5" />
                            <span>Analyze</span>
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
