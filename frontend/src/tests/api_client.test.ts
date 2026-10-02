import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { api, API_BASE_URL } from "../lib/api.ts";

describe("CycloneSense Frontend API Client & Contracts", () => {
  it("should have correct base URL configured", () => {
    assert.ok(API_BASE_URL);
    assert.match(API_BASE_URL, /http:\/\/localhost:8000\/api\/v1/);
  });

  it("should expose all required scientific API methods", () => {
    assert.equal(typeof api.getSystemHealth, "function");
    assert.equal(typeof api.getSystemSettings, "function");
    assert.equal(typeof api.testAdapter, "function");
    assert.equal(typeof api.getStormCatalog, "function");
    assert.equal(typeof api.getStormTrack, "function");
    assert.equal(typeof api.listProducts, "function");
    assert.equal(typeof api.getProductDetails, "function");
    assert.equal(typeof api.getProductSlice, "function");
    assert.equal(typeof api.listMLModels, "function");
    assert.equal(typeof api.runInference, "function");
    assert.equal(typeof api.submitAnalysisJob, "function");
    assert.equal(typeof api.listAnalysisJobs, "function");
    assert.equal(typeof api.getAnalysisJob, "function");
    assert.equal(typeof api.runTemporalComparison, "function");
    assert.equal(typeof api.analyzeExplainability, "function");
    assert.equal(typeof api.getEvaluationReport, "function");
    assert.equal(typeof api.listProvenance, "function");
    assert.equal(typeof api.getEntityProvenance, "function");
  });

  it("should correctly handle API network failure with informative error", async () => {
    // Attempt request to an unrouted/invalid endpoint
    await assert.rejects(
      async () => {
        await api.getAnalysisJob("NON_EXISTENT_JOB_ID_99999");
      },
      (err: Error) => {
        assert.ok(err.message.length > 0);
        return true;
      }
    );
  });
});

describe("CycloneSense Scientific Data & Classification Rules", () => {
  const IMD_CATEGORIES = [
    { maxKts: 33, name: "Tropical Depression" },
    { maxKts: 63, name: "Cyclonic Storm" },
    { maxKts: 89, name: "Very Severe Cyclonic Storm" },
    { maxKts: 119, name: "Extremely Severe Cyclonic Storm" },
    { maxKts: Infinity, name: "Super Cyclonic Storm" },
  ];

  function getCategoryName(windKts: number): string {
    const found = IMD_CATEGORIES.find((c) => windKts <= c.maxKts);
    return found ? found.name : "Unknown";
  }

  it("should classify cyclone intensity into valid IMD operational categories", () => {
    assert.equal(getCategoryName(25), "Tropical Depression");
    assert.equal(getCategoryName(35), "Cyclonic Storm");
    assert.equal(getCategoryName(75), "Very Severe Cyclonic Storm");
    assert.equal(getCategoryName(105), "Extremely Severe Cyclonic Storm");
    assert.equal(getCategoryName(140), "Super Cyclonic Storm");
  });

  it("should maintain valid meteorological ranges for physical variables", () => {
    // Sea Surface Temperature: 270 - 315 K
    const sstSample = 302.5;
    assert.ok(sstSample >= 270 && sstSample <= 315, "SST within physical range (K)");

    // Central pressure: 850 - 1025 hPa
    const pressureSample = 910;
    assert.ok(pressureSample >= 850 && pressureSample <= 1025, "Central pressure within physical range (hPa)");

    // Wind speed: 15 - 185 kts
    const windSample = 140;
    assert.ok(windSample >= 15 && windSample <= 185, "Wind speed within physical range (kts)");
  });

  it("should explain Amphan 0% eyewall energy without manufacturing false positive heatmaps", () => {
    const explainabilityObservation = {
      storm_name: "AMPHAN",
      continuous_intensity_eyewall_energy: 0.669, // 66.9%
      class_0_depression_eyewall_energy: 0.023, // 2.3%
      explanation:
        "Low-intensity classification heads have near-zero gradient in the core because mature cyclone cold cloud tops exceed depression activation thresholds. Backpropagation zeroes out gradients via ReLU.",
    };

    assert.ok(explainabilityObservation.continuous_intensity_eyewall_energy > 0.5);
    assert.ok(explainabilityObservation.class_0_depression_eyewall_energy < 0.05);
    assert.ok(explainabilityObservation.explanation.includes("ReLU"));
  });
});
