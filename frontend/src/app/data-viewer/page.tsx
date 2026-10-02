"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  IconSatellite,
  IconDatabase,
  IconRefresh,
  IconAlert,
} from "@/components/icons";
import { api, ScientificProduct, ScientificProductDetails, ProductSlice } from "@/lib/api";

export default function ScientificDataViewerPage() {
  const [products, setProducts] = useState<ScientificProduct[]>([]);
  const [selectedProduct, setSelectedProduct] = useState<ScientificProduct | null>(null);
  const [productDetails, setProductDetails] = useState<ScientificProductDetails | null>(null);
  const [selectedVariable, setSelectedVariable] = useState<string>("");
  const [sliceData, setSliceData] = useState<ProductSlice | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingSlice, setLoadingSlice] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sliceError, setSliceError] = useState<string | null>(null);

  const [hoveredValue, setHoveredValue] = useState<{ val: number | null; x: number; y: number } | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const loadProducts = async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await api.listProducts(20);
      setProducts(list);
      if (list.length > 0) {
        handleSelectProduct(list[0]);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load scientific products");
    } finally {
      setLoading(false);
    }
  };

  const handleSelectProduct = async (prod: ScientificProduct) => {
    setSelectedProduct(prod);
    setSliceData(null);
    setSliceError(null);
    try {
      const details = await api.getProductDetails(prod.id);
      setProductDetails(details);
      // Pick first non-coordinate variable
      const firstVar =
        details.channels_manifest?.[0]?.name ||
        details.variables_manifest?.find(
          (v: { name: string }) => !["lat", "lon", "time", "date_time", "charsn"].includes(v.name)
        )?.name ||
        details.variables_manifest?.[0]?.name;

      if (firstVar) {
        setSelectedVariable(firstVar);
        loadVariableSlice(prod.id, firstVar);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load product details");
    }
  };

  const loadVariableSlice = async (prodId: string, varName: string) => {
    setLoadingSlice(true);
    setSliceError(null);
    try {
      const slice = await api.getProductSlice(prodId, varName, 64);
      setSliceData(slice);
    } catch (err: unknown) {
      setSliceError(err instanceof Error ? err.message : `Failed to slice variable '${varName}'`);
    } finally {
      setLoadingSlice(false);
    }
  };

  useEffect(() => {
    let isMounted = true;
    api
      .listProducts(20)
      .then(async (list) => {
        if (!isMounted) return;
        setProducts(list);
        setLoading(false);
        if (list.length > 0) {
          const first = list[0];
          setSelectedProduct(first);
          try {
            const details = await api.getProductDetails(first.id);
            if (!isMounted) return;
            setProductDetails(details);
            const firstVar =
              details.channels_manifest?.[0]?.name ||
              details.variables_manifest?.find(
                (v: { name: string }) => !["lat", "lon", "time", "date_time", "charsn"].includes(v.name)
              )?.name ||
              details.variables_manifest?.[0]?.name;

            if (firstVar) {
              setSelectedVariable(firstVar);
              setLoadingSlice(true);
              const slice = await api.getProductSlice(first.id, firstVar, 64);
              if (isMounted) setSliceData(slice);
            }
          } catch (e: unknown) {
            if (isMounted) setError(e instanceof Error ? e.message : "Failed to load product details");
          } finally {
            if (isMounted) setLoadingSlice(false);
          }
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err instanceof Error ? err.message : "Failed to load scientific products");
          setLoading(false);
        }
      });
    return () => {
      isMounted = false;
    };
  }, []);

  // Draw Heatmap on Canvas when sliceData updates
  useEffect(() => {
    if (!sliceData || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const grid = sliceData.grid;
    const rows = grid.length;
    if (rows === 0) return;
    const cols = grid[0].length;

    canvas.width = cols;
    canvas.height = rows;

    const min = sliceData.statistics.min;
    const max = sliceData.statistics.max;
    const range = max - min || 1;

    const imgData = ctx.createImageData(cols, rows);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = grid[r][c];
        const idx = (r * cols + c) * 4;

        if (val === null || isNaN(val)) {
          imgData.data[idx] = 15;
          imgData.data[idx + 1] = 23;
          imgData.data[idx + 2] = 42;
          imgData.data[idx + 3] = 255;
          continue;
        }

        // Colormap: Deep convective core (Dark Red/Violet) -> Eyewall (Cyan/Green) -> Outer Warm Sea (Yellow/Red)
        const norm = Math.max(0, Math.min(1, (val - min) / range));

        let red = 0,
          green = 0,
          blue = 0;
        if (norm < 0.25) {
          // Cold convective clouds (170K - 210K) -> White to Cyan
          const t = norm / 0.25;
          red = Math.round(255 - t * 150);
          green = Math.round(255 - t * 50);
          blue = 255;
        } else if (norm < 0.5) {
          // Mid level (210K - 250K) -> Cyan to Deep Blue/Indigo
          const t = (norm - 0.25) / 0.25;
          red = Math.round(105 * (1 - t));
          green = Math.round(205 * (1 - t));
          blue = Math.round(255 * (1 - t * 0.4));
        } else if (norm < 0.75) {
          // Deep Blue to Yellow
          const t = (norm - 0.5) / 0.25;
          red = Math.round(255 * t);
          green = Math.round(200 * t);
          blue = Math.round(150 * (1 - t));
        } else {
          // Warm surface / eye warming (280K - 310K) -> Orange to Crimson
          const t = (norm - 0.75) / 0.25;
          red = 255;
          green = Math.round(200 * (1 - t));
          blue = Math.round(50 * (1 - t));
        }

        imgData.data[idx] = red;
        imgData.data[idx + 1] = green;
        imgData.data[idx + 2] = blue;
        imgData.data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [sliceData]);

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!sliceData || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * sliceData.sampled_shape[1]);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * sliceData.sampled_shape[0]);

    if (
      y >= 0 &&
      y < sliceData.sampled_shape[0] &&
      x >= 0 &&
      x < sliceData.sampled_shape[1]
    ) {
      setHoveredValue({ val: sliceData.grid[y][x], x, y });
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 pb-2 border-b border-slate-800/80">
        <div>
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-white flex items-center gap-3">
            <span>Scientific Earth Observation & Satellite Viewer</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
              NetCDF4 / HDF5 CF-1.8
            </span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Direct numerical slice rendering and quality control validation for authentic satellite grid products.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={loadProducts}
            disabled={loading}
            className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700 text-slate-300 transition-colors disabled:opacity-50"
          >
            <IconRefresh className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-center gap-3">
          <IconAlert className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Grid: Product Catalog + Variable Viewer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Ingested Products List (4 cols) */}
        <div className="lg:col-span-4 p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <IconDatabase className="w-4 h-4 text-cyan-400" />
              <span>Available Scientific Products</span>
            </h2>
            <span className="text-sm font-mono text-slate-400">{products.length} Ingested</span>
          </div>

          <div className="space-y-2.5 max-h-[560px] overflow-y-auto">
            {products.map((p) => {
              const isSelected = selectedProduct?.id === p.id;
              return (
                <div
                  key={p.id}
                  onClick={() => handleSelectProduct(p)}
                  className={`p-3 rounded-xl border cursor-pointer transition-all ${
                    isSelected
                      ? "bg-cyan-500/10 border-cyan-500/40 shadow-sm shadow-cyan-950/40"
                      : "bg-slate-900/60 hover:bg-slate-800/40 border-slate-800"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-white truncate max-w-[200px]">
                      {p.filename}
                    </span>
                    <span className="text-xs px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
                      {p.file_format}
                    </span>
                  </div>
                  <div className="mt-2 flex items-center justify-between text-sm text-slate-400 font-mono">
                    <span>{(p.file_size_bytes / (1024 * 1024)).toFixed(2)} MB</span>
                    <span className="text-cyan-400">{p.source_origin}</span>
                  </div>
                  <div className="mt-1 text-xs text-slate-500 font-mono truncate">
                    SHA256: {p.sha256_hash.substring(0, 16)}...
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Variable Slice Visualizer & QC (8 cols) */}
        <div className="lg:col-span-8 p-5 rounded-2xl bg-[#0c121e]/90 border border-slate-800/80 shadow-xl space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-800/80 pb-3">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <IconSatellite className="w-4 h-4 text-teal-400" />
                <span>Numerical Slice & Quality Control</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                {selectedProduct ? selectedProduct.filename : "Select a product"}
              </p>
            </div>

            {/* Variable Select Dropdown */}
            {productDetails && (
              <div className="flex items-center gap-2">
                <label className="text-xs text-slate-400 font-mono">Variable:</label>
                <select
                  value={selectedVariable}
                  onChange={(e) => {
                    setSelectedVariable(e.target.value);
                    if (selectedProduct) loadVariableSlice(selectedProduct.id, e.target.value);
                  }}
                  className="py-1.5 px-3 bg-slate-900 border border-slate-700 rounded-lg text-xs text-cyan-300 font-mono focus:outline-none focus:border-cyan-500"
                >
                  {productDetails.variables_manifest?.map((v: { name: string; units?: string }) => (
                    <option key={v.name} value={v.name}>
                      {v.name} {v.units ? `(${v.units})` : ""}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          {loadingSlice ? (
            <div className="py-24 text-center text-slate-400 font-mono text-xs space-y-2">
              <IconRefresh className="w-6 h-6 animate-spin mx-auto text-cyan-400" />
              <p>Extracting calibrated numerical 2D slice from NetCDF dataset...</p>
            </div>
          ) : sliceError ? (
            <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-300 text-sm font-mono">
              <IconAlert className="w-4 h-4 inline mr-2 text-amber-400" />
              {sliceError}
            </div>
          ) : sliceData ? (
            <div className="space-y-4">
              {/* Statistics Row */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm font-mono">
                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="text-xs text-slate-400">Min Observed</div>
                  <div className="text-base font-bold text-cyan-300">
                    {sliceData.statistics.min.toFixed(2)} {sliceData.units}
                  </div>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="text-xs text-slate-400">Max Observed</div>
                  <div className="text-base font-bold text-rose-300">
                    {sliceData.statistics.max.toFixed(2)} {sliceData.units}
                  </div>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="text-xs text-slate-400">Mean / Average</div>
                  <div className="text-base font-bold text-white">
                    {sliceData.statistics.mean.toFixed(2)} {sliceData.units}
                  </div>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="text-xs text-slate-400">Std Deviation</div>
                  <div className="text-base font-bold text-slate-300">
                    ±{sliceData.statistics.std.toFixed(2)}
                  </div>
                </div>
              </div>

              {/* Canvas Heatmap Display */}
              <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800/80 flex flex-col items-center">
                <div className="relative group">
                  <canvas
                    ref={canvasRef}
                    onMouseMove={handleCanvasMouseMove}
                    onMouseLeave={() => setHoveredValue(null)}
                    className="w-full max-w-[420px] aspect-square rounded-lg shadow-2xl image-rendering-pixelated cursor-crosshair border border-slate-800"
                  />
                  {hoveredValue && (
                    <div className="absolute top-2 left-2 px-2 py-1 rounded bg-black/80 backdrop-blur-md text-sm font-mono text-cyan-300 border border-cyan-500/30">
                      Pixel ({hoveredValue.x}, {hoveredValue.y}):{" "}
                      {hoveredValue.val !== null ? `${hoveredValue.val.toFixed(2)} ${sliceData.units}` : "NaN / Masked"}
                    </div>
                  )}
                </div>

                {/* Legend Bar */}
                <div className="w-full max-w-[420px] mt-4 space-y-1">
                  <div className="h-3 w-full rounded-full bg-gradient-to-r from-white via-cyan-400 via-indigo-600 via-yellow-400 to-rose-600 border border-slate-700/60" />
                  <div className="flex justify-between text-xs text-slate-400 font-mono">
                    <span>Cold Convective Clouds ({sliceData.statistics.min.toFixed(0)} {sliceData.units})</span>
                    <span>Warm Surface / Core ({sliceData.statistics.max.toFixed(0)} {sliceData.units})</span>
                  </div>
                </div>
              </div>

              {/* QC Details & Provenance */}
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-sm font-mono space-y-2 text-slate-300">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-400">Original Resolution:</span>
                  <span className="text-white">
                    {sliceData.original_shape[0]} × {sliceData.original_shape[1]}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-400">Browser Sampled Grid:</span>
                  <span className="text-white">
                    {sliceData.sampled_shape[0]} × {sliceData.sampled_shape[1]}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-400">Slice Cryptographic Hash:</span>
                  <span className="text-cyan-400 truncate max-w-[240px]">
                    SHA256: {sliceData.slice_sha256}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="py-24 text-center text-slate-500 font-mono text-xs">
              Select a scientific product and variable to render calibrated 2D slice.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
