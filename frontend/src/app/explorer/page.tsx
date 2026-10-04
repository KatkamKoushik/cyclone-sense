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
import {
  api,
  StormSummary,
  StormTrackResponse,
  StormTrackObservation,
  LocationPreset,
  LocationSearchResponse,
} from "@/lib/api";

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
  const [activeMode, setActiveMode] = useState<"location" | "catalog">("location");

  // Catalog State
  const [storms, setStorms] = useState<StormSummary[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [searchQuery, setSearchQuery] = useState("");
  const [seasonFilter, setSeasonFilter] = useState<number | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Location / Spatial Search State
  const [presets, setPresets] = useState<LocationPreset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>("PURI");
  const [customLocationQuery, setCustomLocationQuery] = useState<string>("");
  const [radiusKm, setRadiusKm] = useState<number>(150);
  const [locationResult, setLocationResult] = useState<LocationSearchResponse | null>(null);
  const [loadingLocation, setLoadingLocation] = useState<boolean>(true);
  const [locationError, setLocationError] = useState<string | null>(null);

  // Selected storm track
  const [selectedStorm, setSelectedStorm] = useState<{
    storm_id: string;
    storm_name: string;
    season: number;
    peak_intensity_kts: number;
    peak_category: number;
  } | null>(null);
  const [trackData, setTrackData] = useState<StormTrackResponse | null>(null);
  const [loadingTrack, setLoadingTrack] = useState(false);
  const [trackError, setTrackError] = useState<string | null>(null);

  // Initial load
  useEffect(() => {
    let isMounted = true;
    api
      .getLocationPresets()
      .then((pList) => {
        if (isMounted) setPresets(pList);
      })
      .catch(() => {});

    // Initial search for Puri, Odisha
    api
      .searchStormsByLocation({ query: "Puri", radius_km: 150 })
      .then((res) => {
        if (isMounted) {
          setLocationResult(res);
          setLoadingLocation(false);
          // Auto-select the top encounter storm if available
          if (res.encounters.length > 0) {
            const top = res.encounters[0];
            setSelectedStorm({
              storm_id: top.storm_id,
              storm_name: top.storm_name,
              season: top.season,
              peak_intensity_kts: top.wind_experienced_kts,
              peak_category: top.category_at_encounter,
            });
            api
              .getStormTrack(top.storm_id)
              .then((t) => {
                if (isMounted) setTrackData(t);
              })
              .catch(() => {});
          }
        }
      })
      .catch((err) => {
        if (isMounted) {
          setLocationError(err instanceof Error ? err.message : "Failed to query location");
          setLoadingLocation(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const handleLocationSearch = async (locQuery?: string, radius?: number) => {
    setLoadingLocation(true);
    setLocationError(null);
    const targetQuery = locQuery || customLocationQuery || selectedPresetId;
    const targetRadius = radius || radiusKm;
    try {
      const res = await api.searchStormsByLocation({
        query: targetQuery,
        radius_km: targetRadius,
      });
      setLocationResult(res);
      if (res.encounters.length > 0) {
        handleSelectEncounter(res.encounters[0]);
      }
    } catch (err: unknown) {
      setLocationError(err instanceof Error ? err.message : "Failed to search location encounters");
    } finally {
      setLoadingLocation(false);
    }
  };

  const handleSelectPreset = (p: LocationPreset) => {
    setSelectedPresetId(p.id);
    setCustomLocationQuery(p.name);
    handleLocationSearch(p.name, radiusKm);
  };

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

  const handleSelectStorm = async (storm: StormSummary) => {
    setSelectedStorm({
      storm_id: storm.storm_id,
      storm_name: storm.storm_name,
      season: storm.season,
      peak_intensity_kts: storm.peak_intensity_kts,
      peak_category: storm.peak_category,
    });
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

  const handleSelectEncounter = async (enc: {
    storm_id: string;
    storm_name: string;
    season: number;
    wind_experienced_kts: number;
    category_at_encounter: number;
  }) => {
    setSelectedStorm({
      storm_id: enc.storm_id,
      storm_name: enc.storm_name,
      season: enc.season,
      peak_intensity_kts: enc.wind_experienced_kts,
      peak_category: enc.category_at_encounter,
    });
    setLoadingTrack(true);
    setTrackError(null);
    try {
      const track = await api.getStormTrack(enc.storm_id);
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
            <span>Cyclone Explorer & Risk Engine</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
              NOAA IBTrACS Ground Truth
            </span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Query authentic historical tropical cyclone encounters by coastal location or browse the full 2000–2026 archive.
          </p>
        </div>

        {/* Mode Switch Tabs */}
        <div className="flex items-center gap-1.5 p-1 bg-slate-900 border border-slate-800 rounded-xl">
          <button
            onClick={() => setActiveMode("location")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeMode === "location"
                ? "bg-cyan-600 text-white shadow-md shadow-cyan-950/40"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            📍 Spatial Location Search
          </button>
          <button
            onClick={() => {
              setActiveMode("catalog");
              if (storms.length === 0) fetchCatalog();
            }}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeMode === "catalog"
                ? "bg-cyan-600 text-white shadow-md shadow-cyan-950/40"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            📋 Full Storm Catalog
          </button>
        </div>
      </div>

      {/* MODE 1: LOCATION SEARCH CONTROLS */}
      {activeMode === "location" && (
        <div className="p-5 rounded-2xl bg-gradient-to-r from-slate-900 via-[#0c1322] to-slate-900 border border-cyan-500/20 shadow-xl space-y-4">
          <div className="flex flex-col md:flex-row items-stretch md:items-center gap-3">
            <div className="relative flex-1">
              <IconSearch className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search any coastal city, district, or coordinates (e.g. Puri, Visakhapatnam, Kolkata, 19.81, 85.83)..."
                value={customLocationQuery}
                onChange={(e) => setCustomLocationQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleLocationSearch()}
                className="w-full pl-9 pr-4 py-2.5 bg-slate-900/90 border border-slate-700/80 rounded-xl text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono"
              />
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400 font-mono shrink-0">Radius:</span>
              <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-xl border border-slate-800 font-mono text-xs">
                {[50, 100, 150, 250, 400].map((r) => (
                  <button
                    key={r}
                    onClick={() => {
                      setRadiusKm(r);
                      handleLocationSearch(undefined, r);
                    }}
                    className={`px-2 py-1 rounded-lg transition-colors ${
                      radiusKm === r ? "bg-cyan-600 text-white font-bold" : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {r}km
                  </button>
                ))}
              </div>

              <button
                onClick={() => handleLocationSearch()}
                disabled={loadingLocation}
                className="px-5 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl text-xs font-semibold shrink-0 transition-colors disabled:opacity-50"
              >
                {loadingLocation ? "Searching..." : "Calculate Encounters"}
              </button>
            </div>
          </div>

          {/* Quick Coastal Hub Presets */}
          <div className="flex items-center gap-2 flex-wrap pt-1">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">Coastal Hubs:</span>
            {presets.map((p) => (
              <button
                key={p.id}
                onClick={() => handleSelectPreset(p)}
                className={`px-2.5 py-1 rounded-lg text-xs font-mono transition-all ${
                  selectedPresetId === p.id && !customLocationQuery
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold"
                    : "bg-slate-800/60 text-slate-400 border border-slate-800 hover:text-slate-200"
                }`}
              >
                {p.name.split(",")[0]} ({p.state})
              </button>
            ))}
          </div>
        </div>
      )}

      {/* MODE 2: CATALOG SEARCH CONTROLS */}
      {activeMode === "catalog" && (
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
            </select>

            <button
              onClick={fetchCatalog}
              className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl text-xs font-semibold shrink-0 transition-colors"
            >
              Search
            </button>
          </div>
        </div>
      )}

      {locationError && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-center gap-3">
          <IconAlert className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{locationError}</span>
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-center gap-3">
          <IconAlert className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Main 2-Column Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (7 cols): Either Location Encounter Dossier OR Full Catalog */}
        <div className="lg:col-span-7 space-y-4">
          {activeMode === "location" ? (
            /* LOCATION DOSSIER & ENCOUNTER LIST */
            <div className="p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-5">
              {loadingLocation ? (
                <div className="py-24 text-center text-slate-400 font-mono text-xs space-y-2">
                  <IconRefresh className="w-6 h-6 animate-spin mx-auto text-cyan-400" />
                  <p>Calculating spherical Haversine distances across IBTrACS archive...</p>
                </div>
              ) : locationResult ? (
                <div className="space-y-4">
                  {/* Location Header & Climate Scenario */}
                  <div className="border-b border-slate-800/80 pb-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div>
                      <h2 className="text-xl font-bold text-white flex items-center gap-2">
                        <span>{locationResult.location.name}</span>
                        <span className="text-xs px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 font-mono">
                          {locationResult.location.latitude.toFixed(2)}°N, {locationResult.location.longitude.toFixed(2)}°E
                        </span>
                      </h2>
                      <p className="text-xs text-slate-400 mt-0.5">
                        {locationResult.location.coastal_zone || locationResult.location.basin} • {locationResult.location.radius_km} km Analysis Buffer
                      </p>
                    </div>

                    <div className="text-right">
                      <span className="text-xs font-mono text-slate-400">Qualifying Storms</span>
                      <div className="text-2xl font-bold font-mono text-cyan-400">
                        {locationResult.historical_encounter_frequency.total_qualifying_cyclones}
                      </div>
                    </div>
                  </div>

                  {/* Summary Metric Cards */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
                    <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
                      <div className="text-slate-400">Highest Wind Encountered</div>
                      <div className="text-lg font-bold text-white mt-1">
                        {locationResult.historical_encounter_frequency.highest_wind_encountered_kts} kts
                      </div>
                      <div className="text-[10px] text-cyan-400 mt-0.5">Within {locationResult.location.radius_km}km Buffer</div>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
                      <div className="text-slate-400">Major Encounters (Cat 3+)</div>
                      <div className="text-lg font-bold text-rose-400 mt-1">
                        {locationResult.historical_encounter_frequency.major_cyclone_encounters_count} storms
                      </div>
                      <div className="text-[10px] text-slate-400 mt-0.5">Wind &gt;= 64 kts in radius</div>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
                      <div className="text-slate-400">Total Track Observations</div>
                      <div className="text-lg font-bold text-teal-400 mt-1">
                        {locationResult.historical_encounter_frequency.total_observations_in_radius} points
                      </div>
                      <div className="text-[10px] text-slate-400 mt-0.5">2000–2026 Archive</div>
                    </div>
                  </div>

                  {/* Climate Resilience Scenario Card (Properly Framed) */}
                  <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs space-y-1.5">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-300 uppercase tracking-wider text-[11px]">
                        Climate Resilience Scenario Classification
                      </span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/10 text-amber-300 border border-amber-500/30">
                        {locationResult.climate_resilience_scenario.scenario_band.replace(/_/g, " ")}
                      </span>
                    </div>
                    <p className="text-slate-400 text-xs">
                      {locationResult.climate_resilience_scenario.scenario_description}
                    </p>
                    <p className="text-[11px] text-slate-500 italic pt-1 border-t border-slate-800/60">
                      * {locationResult.climate_resilience_scenario.disclaimer}
                    </p>
                  </div>

                  {/* Encounter Table */}
                  <div className="space-y-2">
                    <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider font-mono">
                      Historical Encounters ({locationResult.encounters.length} Storms)
                    </h3>
                    <div className="overflow-x-auto max-h-[380px] overflow-y-auto">
                      <table className="w-full text-left text-xs font-mono">
                        <thead className="sticky top-0 bg-[#0c121e] border-b border-slate-800 text-slate-400">
                          <tr>
                            <th className="pb-2">Season</th>
                            <th className="pb-2">Storm Name</th>
                            <th className="pb-2 text-right">Closest Approach</th>
                            <th className="pb-2 text-right">Max Wind</th>
                            <th className="pb-2">Category</th>
                            <th className="pb-2 text-right">Action</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/60 text-slate-300">
                          {locationResult.encounters.map((enc) => {
                            const isSelected = selectedStorm?.storm_id === enc.storm_id;
                            return (
                              <tr
                                key={enc.storm_id}
                                onClick={() => handleSelectEncounter(enc)}
                                className={`cursor-pointer transition-colors ${
                                  isSelected
                                    ? "bg-cyan-500/15 border-l-2 border-cyan-400"
                                    : "hover:bg-slate-800/30"
                                }`}
                              >
                                <td className="py-2 text-slate-400 font-bold">{enc.season}</td>
                                <td className="py-2 text-white font-bold">{enc.storm_name}</td>
                                <td className="py-2 text-right font-bold text-cyan-300">
                                  {enc.closest_distance_km} km
                                </td>
                                <td className="py-2 text-right text-white">
                                  {enc.wind_experienced_kts} kts
                                </td>
                                <td className="py-2">
                                  <span
                                    className={`px-1.5 py-0.5 rounded text-[10px] font-semibold border ${
                                      CATEGORY_COLORS[enc.category_at_encounter] || "bg-slate-800 text-slate-400"
                                    }`}
                                  >
                                    Cat {enc.category_at_encounter}
                                  </span>
                                </td>
                                <td className="py-2 text-right">
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleSelectEncounter(enc);
                                    }}
                                    className="px-2 py-0.5 rounded bg-cyan-600/20 hover:bg-cyan-600/40 text-cyan-300 text-[10px] font-semibold"
                                  >
                                    Inspect Track
                                  </button>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          ) : (
            /* FULL CATALOG TABLE */
            <div className="p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-4">
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
          )}
        </div>

        {/* Right Column (5 cols): Track Inspector Drawer */}
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
                  className="p-1.5 rounded-lg bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-300 border border-indigo-500/20 text-xs font-mono flex items-center gap-1"
                  title="Compare T1 vs T2"
                >
                  <IconCompare className="w-3.5 h-3.5" />
                  <span>T1/T2</span>
                </Link>
              </div>
            )}
          </div>

          {!selectedStorm ? (
            <div className="py-28 text-center text-slate-500 font-mono text-xs space-y-2">
              <IconInfo className="w-8 h-8 text-slate-600 mx-auto" />
              <p>Click any storm from the encounters or catalog table to inspect its chronological track.</p>
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
                  <div className="text-[11px] text-slate-400">Total Timesteps</div>
                  <div className="text-base font-bold text-white">{trackData.total_observations}</div>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="text-[11px] text-slate-400">Intensity</div>
                  <div className="text-base font-bold text-cyan-300">
                    {selectedStorm.peak_intensity_kts.toFixed(0)} kts
                  </div>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="text-[11px] text-slate-400">Classification</div>
                  <div className="text-xs font-semibold text-emerald-400 truncate mt-1">
                    {CATEGORY_NAMES[selectedStorm.peak_category].split("(")[0]}
                  </div>
                </div>
              </div>

              {/* Observations Track Table */}
              <div className="overflow-x-auto max-h-[460px] overflow-y-auto">
                <table className="w-full text-left text-xs font-mono">
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
                        <td className="py-2 text-slate-400 truncate max-w-[100px]">
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
