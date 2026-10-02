import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "@/components/AppShell";

export const metadata: Metadata = {
  title: "CycloneSense — Explainable Multi-Source Tropical Cyclone Pattern Intelligence",
  description:
    "Scientific intelligence, physical quality control, and explainable neural attribution for tropical cyclone pattern analysis derived directly from authentic NetCDF4 and HDF5 Earth Observation products.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#070b12] text-slate-100 antialiased selection:bg-cyan-500/30 selection:text-cyan-200">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}

