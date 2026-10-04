"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  IconVortex,
  IconActivity,
  IconSatellite,
  IconCpu,
  IconSearch,
  IconPlay,
  IconCompare,
  IconShield,
  IconSettings,
  IconSparkles,
  IconMenu,
  IconClose,
  IconAlert,
} from "./icons";
import { api, SystemHealth } from "@/lib/api";

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
  badge?: string;
}

const NAV_ITEMS: NavItem[] = [
  { name: "Overview Dashboard", href: "/", icon: IconActivity },
  { name: "Impact Intelligence", href: "/impact", icon: IconSatellite, badge: "PROTOTYPE" },
  { name: "Cyclone Explorer", href: "/explorer", icon: IconSearch },
  { name: "Satellite Data Viewer", href: "/data-viewer", icon: IconSatellite },
  { name: "Analysis Studio", href: "/analysis", icon: IconPlay },
  { name: "Inference Results", href: "/results", icon: IconCpu },
  { name: "T1 → T2 Evolution", href: "/temporal", icon: IconCompare },
  { name: "Explainability & Grad-CAM", href: "/explainability", icon: IconSparkles },
  { name: "Model Benchmarks", href: "/models", icon: IconCpu },
  { name: "Provenance Audit", href: "/provenance", icon: IconShield },
  { name: "Settings & Sources", href: "/settings", icon: IconSettings },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [currentTimeUTC, setCurrentTimeUTC] = useState<string>("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setCurrentTimeUTC(now.toISOString().replace("T", " ").substring(0, 19) + " UTC");
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    let isMounted = true;
    api
      .getSystemHealth()
      .then((data) => {
        if (isMounted) {
          setHealth(data);
          setHealthError(null);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setHealthError(err.message || "Failed to connect to backend");
        }
      });
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="min-h-screen bg-[#070b12] text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Top Header */}
      <header className="sticky top-0 z-40 bg-[#0c121e]/90 backdrop-blur-md border-b border-slate-800/80 px-5 lg:px-8 h-16 flex items-center justify-between shadow-lg shadow-black/40">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="lg:hidden p-2 rounded-lg bg-slate-800/60 hover:bg-slate-700/60 text-slate-300 border border-slate-700/50"
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? <IconClose className="w-5 h-5" /> : <IconMenu className="w-5 h-5" />}
          </button>

          <Link href="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 via-indigo-600 to-teal-400 flex items-center justify-center shadow-md shadow-cyan-900/30 group-hover:scale-105 transition-transform duration-200">
              <IconVortex className="w-5 h-5 text-white animate-spin-slow" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-xl tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-300 bg-clip-text text-transparent">
                  CycloneSense
                </span>
                <span className="text-xs font-semibold uppercase px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                  V3
                </span>
              </div>
              <p className="text-xs text-slate-400 tracking-wider hidden sm:block">
                Explainable Tropical Cyclone Intelligence
              </p>
            </div>
          </Link>
        </div>

        {/* Middle Basin Badge */}
        <div className="hidden md:flex items-center gap-2 text-sm bg-slate-800/40 border border-slate-700/60 px-4 py-2 rounded-full text-slate-300">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
          <span>Basin:</span>
          <span className="text-cyan-300 font-semibold">North Indian (NI)</span>
          <span className="text-slate-500">|</span>
          <span className="text-slate-400">IBTrACS v04r01</span>
        </div>

        {/* Right Status Indicators */}
        <div className="flex items-center gap-4">
          <div className="hidden sm:flex flex-col text-right">
            <span className="text-sm text-slate-300 font-medium font-mono">{currentTimeUTC}</span>
          </div>

          {health ? (
            <div
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-semibold border ${
                health.status === "OPERATIONAL"
                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                  : "bg-amber-500/10 text-amber-400 border-amber-500/20"
              }`}
            >
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  health.status === "OPERATIONAL" ? "bg-emerald-400" : "bg-amber-400"
                } animate-ping`}
              ></span>
              <span className="font-mono">{health.status}</span>
            </div>
          ) : healthError ? (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-400"></span>
              <span className="font-mono">API OFFLINE</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-semibold bg-slate-800 text-slate-400 border border-slate-700">
              <span className="w-2.5 h-2.5 rounded-full bg-slate-400 animate-pulse"></span>
              <span className="font-mono">CONNECTING</span>
            </div>
          )}
        </div>
      </header>

      {/* Main Workspace Body */}
      <div className="flex-1 flex overflow-hidden">
        {/* Desktop Sidebar */}
        <aside className="hidden lg:flex w-72 flex-col bg-[#090e18] border-r border-slate-800/80 p-4 justify-between">
          <nav className="space-y-1">
            <div className="px-3 py-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Scientific Modules
            </div>
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center justify-between px-3 py-3 rounded-xl text-sm font-medium transition-all duration-150 ${
                    isActive
                      ? "bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 shadow-sm shadow-cyan-950/40"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <Icon className={`w-5 h-5 ${isActive ? "text-cyan-400" : "text-slate-400"}`} />
                    <span>{item.name}</span>
                  </div>
                  {item.badge && (
                    <span
                      className={`text-xs px-2 py-0.5 rounded font-mono font-medium ${
                        isActive ? "bg-cyan-500/20 text-cyan-200" : "bg-slate-800 text-slate-400"
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </nav>

          {/* System Info Box */}
          <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-800/80 text-sm space-y-2 text-slate-400">
            <div className="flex justify-between items-center text-slate-300">
              <span className="font-semibold">Compute Hardware</span>
              <span className="text-cyan-400 text-xs font-semibold">
                {health?.compute?.gpu?.available ? "GPU ACCELERATED" : "CPU MODE"}
              </span>
            </div>
            <div className="text-xs text-slate-500 truncate" title={health?.compute?.gpu?.device_name || health?.compute?.platform || "Hardware details"}>
              {health?.compute?.gpu?.device_name || (health?.compute?.platform ? health.compute.platform : (health ? "Unknown / unavailable" : "Detecting compute..."))}
            </div>
            <div className="pt-2 border-t border-slate-800/60 flex justify-between text-xs">
              <span>DB Dialect:</span>
              <span className="text-slate-300">{health?.database?.backend || "SQLite"}</span>
            </div>
          </div>
        </aside>

        {/* Mobile Navigation Drawer */}
        {mobileMenuOpen && (
          <div
            className="lg:hidden fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex"
            onClick={() => setMobileMenuOpen(false)}
          >
            <div
              className="w-80 bg-[#090e18] border-r border-slate-800 p-5 flex flex-col justify-between"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="space-y-4">
                <div className="flex items-center justify-between pb-4 border-b border-slate-800">
                  <div className="flex items-center gap-2">
                    <IconVortex className="w-5 h-5 text-cyan-400" />
                    <span className="font-bold text-lg text-slate-100">CycloneSense</span>
                  </div>
                  <button
                    onClick={() => setMobileMenuOpen(false)}
                    className="p-1 rounded text-slate-400 hover:text-white"
                  >
                    <IconClose className="w-5 h-5" />
                  </button>
                </div>

                <nav className="space-y-1">
                  {NAV_ITEMS.map((item) => {
                    const Icon = item.icon;
                    const isActive = pathname === item.href;
                    return (
                      <Link
                        key={item.href}
                        href={item.href}
                        onClick={() => setMobileMenuOpen(false)}
                        className={`flex items-center justify-between px-3 py-3 rounded-lg text-sm font-medium ${
                          isActive
                            ? "bg-cyan-500/10 text-cyan-300 border border-cyan-500/20"
                            : "text-slate-400 hover:text-white hover:bg-slate-800/40"
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          <Icon className="w-5 h-5 text-cyan-400" />
                          <span>{item.name}</span>
                        </div>
                        {item.badge && (
                          <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                            {item.badge}
                          </span>
                        )}
                      </Link>
                    );
                  })}
                </nav>
              </div>

              <div className="p-4 bg-slate-900 rounded-lg border border-slate-800 text-sm text-slate-400">
                <span className="font-semibold text-slate-300">Active Basin: </span>
                <span>North Indian Ocean (IBTrACS)</span>
              </div>
            </div>
          </div>
        )}

        {/* Page Content Viewport */}
        <main className="flex-1 overflow-y-auto p-5 md:p-8 lg:p-10 bg-gradient-to-b from-[#070b12] to-[#0a0f1d]">
          {children}
        </main>
      </div>

      {/* Mandatory Scientific Governance Footer Banner */}
      <footer className="border-t border-slate-800/80 bg-[#080d16] px-5 py-3 text-sm text-slate-400 flex flex-col sm:flex-row items-center justify-between gap-2">
        <div className="flex items-center gap-2.5 text-center sm:text-left">
          <IconAlert className="w-4 h-4 text-amber-400 shrink-0" />
          <span>
            <strong className="text-slate-300 font-semibold">Scientific Research System:</strong>{" "}
            Not an operational forecast warning service. Attribution heatmaps indicate mathematical neural sensitivity, not causal proof.
          </span>
        </div>
        <div className="flex items-center gap-3 shrink-0 text-slate-500 text-xs font-mono">
          <span>W3C PROV Integrity</span>
          <span>•</span>
          <span>CF-1.8 NetCDF4</span>
        </div>
      </footer>
    </div>
  );
}
