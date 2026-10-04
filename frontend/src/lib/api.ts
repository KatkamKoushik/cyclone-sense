/**
 * CycloneSense API Client
 * Strongly-typed asynchronous HTTP client connecting to the real FastAPI backend.
 * Zero mocks or synthetic placeholders.
 */

/**
 * Resolves the CycloneSense API Base URL.
 * - Runtime Functions (Node.js/SSR): Reads Vercel Service Binding `process.env.BACKEND_URL`
 * - Browser Runtime: Uses `process.env.NEXT_PUBLIC_API_URL` or relative `/api/v1` (routed via Vercel rewrite)
 * - Local / Testing: Falls back to `http://localhost:8000/api/v1`
 */
export function getApiBaseUrl(): string {
  // 1. Server-side runtime (Node.js / SSR / Vercel Serverless Function)
  if (typeof window === "undefined") {
    if (process.env.BACKEND_URL) {
      const raw = process.env.BACKEND_URL.replace(/\/+$/, "");
      return raw.endsWith("/api/v1") ? raw : `${raw}/api/v1`;
    }
    if (process.env.NEXT_PUBLIC_API_URL) {
      return process.env.NEXT_PUBLIC_API_URL;
    }
    return "http://localhost:8000/api/v1";
  }

  // 2. Browser runtime: use public API URL or relative route matching Vercel rewrite
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  return "/api/v1";
}

export const API_BASE_URL = getApiBaseUrl();

export interface SystemHealth {
  status: "OPERATIONAL" | "DEGRADED" | "OFFLINE";
  project: string;
  version: string;
  environment: string;
  python_version: string;
  database: {
    status: string;
    backend: string;
    url_masked: string;
  };
  redis_task_queue: {
    status: string;
    always_eager_fallback: boolean;
  };
  compute: {
    gpu: {
      available: boolean;
      device_name: string | null;
      device_count?: number;
    };
    platform: string;
  };
  adapters: {
    noaa_ibtracs: Record<string, unknown> & {
      connectivity?: {
        connected?: boolean;
        status?: string;
        file_size_bytes?: number;
        error?: string;
      };
      local_file_exists?: boolean;
    };
    noaa_goes: Record<string, unknown> & {
      connectivity?: {
        connected?: boolean;
        status?: string;
        latency_ms?: number;
        error?: string;
      };
      anonymous_access_enabled?: boolean;
    };
    nasa_earthdata?: Record<string, unknown> & {
      connectivity?: {
        connected?: boolean;
        status?: string;
        latency_ms?: number;
        error?: string;
      };
      configured?: boolean;
    };
    isro_insat: Record<string, unknown> & {
      connectivity?: {
        connected?: boolean;
        status?: string;
        error?: string;
      };
      credentials_present?: boolean;
    };
  };
  models: Array<{
    id: string;
    version: string;
    loaded: boolean;
    description: string;
  }>;
}

export interface SystemSettings {
  project_name: string;
  version: string;
  environment: string;
  debug_mode: boolean;
  storage: {
    data_raw_dir: string;
    data_processed_dir: string;
    checkpoints_dir: string;
  };
  database_backend: string;
  adapters: {
    noaa_ibtracs: {
      local_file_exists: boolean;
      [key: string]: unknown;
    };
    noaa_goes: {
      s3_bucket: string;
      anonymous_access_enabled: boolean;
      [key: string]: unknown;
    };
    nasa_earthdata?: {
      source: string;
      configured: boolean;
      auth_type: string;
      username?: string;
      has_bearer_token: boolean;
      endpoint: string;
      status: string;
      [key: string]: unknown;
    };
    isro_insat: {
      portal_url: string;
      credentials_required: boolean;
      credentials_present: boolean;
      [key: string]: unknown;
    };
  };
}

export interface StormSummary {
  storm_id: string;
  storm_name: string;
  season: number;
  obs_count: number;
  peak_intensity_kts: number;
  min_pressure_hpa: number | null;
  peak_category: number;
  start_time: string;
  end_time: string;
}

export interface StormCatalogResponse {
  total_storms: number;
  offset: number;
  limit: number;
  storms: StormSummary[];
}

export interface StormTrackObservation {
  index: number;
  timestamp: string;
  latitude: number;
  longitude: number;
  wind_kts: number;
  pressure_hpa: number | null;
  category: number;
  forward_speed_kmh: number;
  forward_bearing_deg: number;
}

export interface StormTrackResponse {
  storm_id: string;
  storm_name: string;
  season: number;
  total_observations: number;
  observations: StormTrackObservation[];
}

export interface ScientificProduct {
  id: string;
  filename: string;
  file_format: string;
  file_size_bytes: number;
  sha256_hash: string;
  source_origin: string;
  spatial_coverage: Record<string, unknown>;
  temporal_coverage: Record<string, unknown>;
  variables_count: number;
  channels_count: number;
  created_at: string;
}

export interface VariableManifestItem {
  name: string;
  units?: string;
  dimensions?: string[];
  dtype?: string;
  attributes?: Record<string, unknown>;
}

export interface ChannelManifestItem {
  name: string;
  wavelength?: number;
  description?: string;
}

export interface ScientificProductDetails extends ScientificProduct {
  qc_evaluations?: Array<{
    check_name: string;
    passed: boolean;
    details?: Record<string, unknown>;
  }>;
  variables_manifest?: VariableManifestItem[];
  channels_manifest?: ChannelManifestItem[];
}

export interface ProductSlice {
  product_id: string;
  variable_name: string;
  original_shape: [number, number];
  sampled_shape: [number, number];
  units: string;
  standard_name: string;
  long_name: string;
  statistics: {
    min: number;
    max: number;
    mean: number;
    std: number;
  };
  slice_sha256: string;
  grid: Array<Array<number | null>>;
}

export interface MLModelMeta {
  model_id: string;
  type: string;
  description: string;
  inputs: string[];
  supported_tasks: string[];
  checkpoints: Array<{
    filename: string;
    path: string;
    size_bytes: number;
    modified: string;
  }>;
}

export interface InferencePayload {
  model_type: "fusion" | "image" | "environment" | "baseline";
  center_latitude: number;
  center_longitude: number;
  forward_speed_kmh: number;
  forward_bearing_deg: number;
  pressure_hpa?: number;
  storm_id?: string;
  storm_name?: string;
  observation_time_iso?: string;
  reference_wind_kts?: number;
  product_id?: string;
}

export interface AnalysisJob {
  job_id: string;
  storm_id: string;
  storm_name: string;
  model_type: string;
  status: "PENDING" | "PROCESSING" | "COMPLETED" | "FAILED";
  input_parameters?: InferencePayload;
  predicted_intensity_kts?: number;
  reference_intensity_kts?: number | null;
  absolute_error_kts?: number | null;
  observation_timestamp?: string | null;
  predicted_category?: number;
  category_name?: string;
  probabilities?: number[];
  explainability?: {
    gradcam?: {
      is_valid?: boolean;
      core_concentration_ratio: number;
      peak_activation: number;
      saliency_sha256: string;
      causal_disclaimer: string;
    };
    environmental?: {
      is_valid?: boolean;
      attribution_method: string;
      ranked_features: Array<[string, number]>;
      causal_disclaimer: string;
    };
    reference_intensity_kts?: number | null;
    absolute_error_kts?: number | null;
    observation_timestamp?: string | null;
  };
  provenance_id?: string;
  error_message?: string;
  created_at: string;
  completed_at?: string;
}

export interface TemporalComparisonResult {
  storm_id: string;
  storm_name: string;
  t1: {
    timestamp: string;
    coords: [number, number];
    ground_truth_wind_kts: number;
    category: number;
    pressure_hpa: number | null;
  };
  t2: {
    timestamp: string;
    coords: [number, number];
    ground_truth_wind_kts: number;
    category: number;
    pressure_hpa: number | null;
  };
  temporal_interval_hours: number;
  translational_motion: {
    displacement_km: number;
    speed_kmh: number;
    bearing_deg: number;
  };
  structural_evolution?: {
    delta_eyewall_cooling_kelvin: number;
    delta_eye_warming_kelvin: number;
    delta_convective_vigor_ratio: number;
    diff_grid_sha256: string;
  } | null;
  environmental_evolution: {
    delta_pressure_hpa: number | null;
  };
  intensity_evolution: {
    delta_wind_true_kts: number;
    rate_kts_per_hr: number;
    rapid_intensification_observed: boolean;
  };
  model_predictions?: {
    pred_wind_t1: number;
    pred_wind_t2: number;
    delta_pred_wind: number;
    pred_wind_rate_per_hr: number;
    rapid_intensification_predicted: boolean;
  };
}

export interface ExplainabilityAnalysisResult {
  storm_id: string;
  storm_name: string;
  timestamp: string;
  reference_wind_kts: number;
  reference_category: number;
  target_task: string;
  target_class: number | null;
  gradcam: {
    is_valid?: boolean;
    attribution_status?: string;
    core_concentration_ratio: number;
    peak_activation: number;
    saliency_sha256: string;
    heatmap_grid: number[][];
    diagnostic_notes: string;
    causal_disclaimer: string;
  };
  environmental_attribution: {
    is_valid?: boolean;
    attribution_method: string;
    target_task: string;
    ranked_features: Array<[string, number]>;
    feature_attributions: Record<string, number>;
    causal_disclaimer: string;
  };
}

export interface ProvenanceRecord {
  id: string;
  entity_type: string;
  entity_id: string;
  sha256_hash: string;
  action: string;
  software_version: string;
  parameters: Record<string, unknown>;
  parent_provenance_id?: string | null;
  timestamp: string;
}

export interface EvaluationReport {
  timestamp_utc: string;
  dataset: {
    source: string;
    total_observations: number;
    split: {
      strategy: string;
      seed: number;
      total_storms: number;
      total_observations: number;
      train: { storm_count: number; obs_count: number; seasons: number[] };
      val: { storm_count: number; obs_count: number; seasons: number[] };
      test: { storm_count: number; obs_count: number; seasons: number[] };
    };
    missing_stats: Record<string, unknown>;
  };
  models_evaluated: {
    baseline_cliper?: {
      model_type: string;
      train_samples: number;
      test_samples: number;
      intensity_metrics: {
        sample_count: number;
        mae_kts: number;
        rmse_kts: number;
        bias_kts: number;
        pearson_r: number;
      };
      classification_metrics: {
        sample_count: number;
        accuracy: number;
        precision_macro: number;
        recall_macro: number;
        f1_macro?: number;
        macro_f1?: number;
        f1_weighted?: number;
      };
    };
    environment_only?: {
      test_evaluation: {
        intensity_metrics: {
          mae_kts: number;
          rmse_kts: number;
          bias_kts: number;
          pearson_r: number;
        };
        classification_metrics: {
          accuracy: number;
          precision_macro: number;
          recall_macro: number;
          f1_macro?: number;
          macro_f1?: number;
          f1_weighted?: number;
        };
      };
    };
    image_only?: {
      test_evaluation: {
        intensity_metrics: {
          mae_kts: number;
          rmse_kts: number;
          bias_kts: number;
          pearson_r: number;
        };
        classification_metrics: {
          accuracy: number;
          precision_macro: number;
          recall_macro: number;
          f1_macro?: number;
          macro_f1?: number;
          f1_weighted?: number;
        };
      };
    };
    multimodal_fusion?: {
      test_evaluation: {
        intensity_metrics: {
          mae_kts: number;
          rmse_kts: number;
          bias_kts: number;
          pearson_r: number;
        };
        classification_metrics: {
          accuracy: number;
          precision_macro: number;
          recall_macro: number;
          f1_macro?: number;
          macro_f1?: number;
          f1_weighted?: number;
        };
      };
    };
  };
}

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  let targetUrl = url;
  // If executing inside a Vercel serverless function with BACKEND_URL service binding:
  if (typeof window === "undefined" && process.env.BACKEND_URL) {
    const backendBase = process.env.BACKEND_URL.replace(/\/+$/, "");
    if (targetUrl.startsWith("/api/v1")) {
      targetUrl = `${backendBase}${targetUrl}`;
    } else if (targetUrl.startsWith("http://localhost:8000/api/v1")) {
      targetUrl = targetUrl.replace("http://localhost:8000", backendBase);
    }
  }

  const response = await fetch(targetUrl, {
    ...init,
    headers: {
      "Accept": "application/json",
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });

  if (!response.ok) {
    let errorDetail = `HTTP ${response.status} ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson && errJson.detail) {
        errorDetail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // ignore json parse error on error response
    }
    throw new Error(errorDetail);
  }

  return response.json();
}

export const api = {
  getSystemHealth: () => fetchJSON<SystemHealth>(`${API_BASE_URL}/system/health`),
  getSystemSettings: () => fetchJSON<SystemSettings>(`${API_BASE_URL}/system/settings`),
  testAdapter: (adapterName: string) =>
    fetchJSON<{ adapter: string; status: string; message: string }>(
      `${API_BASE_URL}/system/settings/test-adapter`,
      { method: "POST", body: JSON.stringify({ adapter_name: adapterName }) }
    ),

  getStormCatalog: (params?: { season?: number; search?: string; limit?: number; offset?: number }) => {
    const q = new URLSearchParams();
    if (params?.season) q.append("season", params.season.toString());
    if (params?.search) q.append("search", params.search);
    if (params?.limit) q.append("limit", params.limit.toString());
    if (params?.offset) q.append("offset", params.offset.toString());
    return fetchJSON<StormCatalogResponse>(`${API_BASE_URL}/storms/catalog?${q.toString()}`);
  },

  getStormTrack: (stormId: string) =>
    fetchJSON<StormTrackResponse>(`${API_BASE_URL}/storms/track/${encodeURIComponent(stormId)}`),

  listProducts: (limit = 20) => fetchJSON<ScientificProduct[]>(`${API_BASE_URL}/ingest?limit=${limit}`),
  getProductDetails: (productId: string) =>
    fetchJSON<ScientificProductDetails>(
      `${API_BASE_URL}/ingest/${encodeURIComponent(productId)}`
    ),
  getProductSlice: (productId: string, variableName: string, maxDim = 64) =>
    fetchJSON<ProductSlice>(
      `${API_BASE_URL}/ingest/${encodeURIComponent(productId)}/variables/${encodeURIComponent(variableName)}/slice?max_dim=${maxDim}`
    ),

  listMLModels: () => fetchJSON<MLModelMeta[]>(`${API_BASE_URL}/ml/models`),
  runInference: (payload: InferencePayload) =>
    fetchJSON<Record<string, unknown>>(`${API_BASE_URL}/ml/inference`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  submitAnalysisJob: (payload: InferencePayload) =>
    fetchJSON<AnalysisJob>(`${API_BASE_URL}/ml/jobs`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  listAnalysisJobs: (limit = 20) => fetchJSON<AnalysisJob[]>(`${API_BASE_URL}/ml/jobs?limit=${limit}`),
  getAnalysisJob: (jobId: string) => fetchJSON<AnalysisJob>(`${API_BASE_URL}/ml/jobs/${encodeURIComponent(jobId)}`),

  runTemporalComparison: (payload: { storm_id: string; obs_index_t1: number; obs_index_t2: number }) =>
    fetchJSON<TemporalComparisonResult>(`${API_BASE_URL}/ml/temporal-comparison`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  analyzeExplainability: (payload: { storm_id: string; obs_index?: number; target_task?: string; target_class?: number }) =>
    fetchJSON<ExplainabilityAnalysisResult>(`${API_BASE_URL}/ml/explainability/analyze`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getEvaluationReport: () => fetchJSON<EvaluationReport>(`${API_BASE_URL}/ml/evaluation-report`),

  listProvenance: (limit = 25) => fetchJSON<ProvenanceRecord[]>(`${API_BASE_URL}/provenance?limit=${limit}`),
  getProvenanceStats: () =>
    fetchJSON<{ total_records: number; algorithm: string; standard: string; verified: boolean }>(
      `${API_BASE_URL}/provenance/summary/stats`
    ),
  getEntityProvenance: (entityId: string) =>
    fetchJSON<ProvenanceRecord[]>(`${API_BASE_URL}/provenance/${encodeURIComponent(entityId)}`),

  searchRealtimeGranules: (source: "NOAA_GOES" | "NASA_EARTHDATA", collection?: string, limit = 5) =>
    fetchJSON<{
      source: string;
      collection: string;
      count: number;
      granules: Array<{
        key?: string;
        granule_id?: string;
        title?: string;
        filename?: string;
        download_url: string;
        size_mb: number;
        time_start?: string;
        time_end?: string;
        last_modified?: string;
      }>;
    }>(
      `${API_BASE_URL}/ingest/realtime/search?source=${encodeURIComponent(source)}${collection ? `&collection=${encodeURIComponent(collection)}` : ""}&limit=${limit}`
    ),

  fetchRealtimeGranule: (source: "NOAA_GOES" | "NASA_EARTHDATA", granuleIdentifier: string) =>
    fetchJSON<ScientificProductDetails>(`${API_BASE_URL}/ingest/realtime/fetch`, {
      method: "POST",
      body: JSON.stringify({ source, granule_identifier: granuleIdentifier }),
    }),

  getLocationPresets: () => fetchJSON<LocationPreset[]>(`${API_BASE_URL}/storms/presets`),

  searchStormsByLocation: (params: {
    query?: string;
    latitude?: number;
    longitude?: number;
    radius_km?: number;
    min_wind_kts?: number;
  }) => {
    const q = new URLSearchParams();
    if (params.query) q.append("query", params.query);
    if (params.latitude !== undefined) q.append("latitude", params.latitude.toString());
    if (params.longitude !== undefined) q.append("longitude", params.longitude.toString());
    if (params.radius_km !== undefined) q.append("radius_km", params.radius_km.toString());
    if (params.min_wind_kts !== undefined) q.append("min_wind_kts", params.min_wind_kts.toString());
    return fetchJSON<LocationSearchResponse>(`${API_BASE_URL}/storms/search/location?${q.toString()}`);
  },

  getTelemetryFreshness: () =>
    fetchJSON<TelemetryFreshnessResponse>(`${API_BASE_URL}/system/telemetry-freshness`),

  // Impact Intelligence Methods
  listImpactCyclones: () =>
    fetchJSON<ImpactCycloneSummary[]>(`${API_BASE_URL}/impact/cyclones`),

  listImpactLocations: () =>
    fetchJSON<ImpactLocationSummary[]>(`${API_BASE_URL}/impact/locations`),

  getLocationImpactProfile: (latitude?: number, longitude?: number, radiusKm?: number) => {
    const q = new URLSearchParams();
    if (latitude !== undefined) q.append("latitude", latitude.toString());
    if (longitude !== undefined) q.append("longitude", longitude.toString());
    if (radiusKm !== undefined) q.append("radius_km", radiusKm.toString());
    return fetchJSON<LocationImpactProfile>(`${API_BASE_URL}/impact/location-profile?${q.toString()}`);
  },

  getBeforeAfterPair: (cycloneName: string, locationName: string, sensorType: "OPTICAL" | "SAR") =>
    fetchJSON<BeforeAfterPairInfo>(
      `${API_BASE_URL}/impact/before-after?cyclone_name=${encodeURIComponent(cycloneName)}&location_name=${encodeURIComponent(locationName)}&sensor_type=${encodeURIComponent(sensorType)}`
    ),

  analyzeImpact: (payload: {
    cyclone_name: string;
    location_name: string;
    sensor_type: "OPTICAL" | "SAR";
    vlm_provider?: string;
    question?: string;
  }) =>
    fetchJSON<ImpactIntelligenceReport>(`${API_BASE_URL}/impact/analyze`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  askImpactQuestion: (payload: {
    question: string;
    analysis_id?: string;
    cyclone_name?: string;
    location_name?: string;
  }) =>
    fetchJSON<ImpactQuestionResponse>(`${API_BASE_URL}/impact/question`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getImpactEvidence: (analysisId: string) =>
    fetchJSON<Record<string, unknown>>(`${API_BASE_URL}/impact/evidence/${encodeURIComponent(analysisId)}`),
};

export interface LocationPreset {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  state: string;
  basin: string;
  coastal_zone: string;
}

export interface CycloneLocationEncounter {
  storm_id: string;
  storm_name: string;
  season: number;
  closest_distance_km: number;
  closest_approach_time: string;
  closest_latitude: number;
  closest_longitude: number;
  wind_experienced_kts: number;
  pressure_experienced_hpa: number | null;
  category_at_encounter: number;
  forward_speed_kmh: number;
  total_track_points_in_radius: number;
}

export interface LocationSearchResponse {
  location: {
    name: string;
    latitude: number;
    longitude: number;
    radius_km: number;
    state?: string;
    basin?: string;
    coastal_zone?: string;
  };
  historical_encounter_frequency: {
    total_qualifying_cyclones: number;
    total_observations_in_radius: number;
    observation_period: string;
    distance_threshold_km: number;
    highest_wind_encountered_kts: number;
    lowest_pressure_encountered_hpa: number | null;
    major_cyclone_encounters_count: number;
    methodology_note: string;
  };
  climate_resilience_scenario: {
    scenario_band: string;
    scenario_description: string;
    illustrative_trigger_status: string;
    qualifying_historical_triggers_count: number;
    exemplar_trigger_cyclone: string | null;
    disclaimer: string;
  };
  encounters: CycloneLocationEncounter[];
}

export interface TelemetryFreshnessResponse {
  nasa_earthdata: {
    source: string;
    status: string;
    connected: boolean;
    latest_observation_utc: string | null;
    latest_granule_id: string | null;
    retrieved_at_utc: string;
    data_age_minutes: number | null;
    latency_ms: number;
    quality: string;
    auth_user: string;
  };
  noaa_goes: {
    source: string;
    status: string;
    connected: boolean;
    endpoint: string;
    latency_ms: number;
    product: string;
    auth_type: string;
  };
  basin_monitoring: {
    target_basin: string;
    active_systems_detected: number;
    basin_status: string;
    operating_mode: string;
    archive_dataset: string;
    verification_note: string;
  };
}

export interface ImpactCycloneSummary {
  name: string;
  season: number;
  storm_id: string;
  has_optical_data: boolean;
  has_sar_data: boolean;
  landfall_location: string;
}

export interface ImpactLocationSummary {
  name: string;
  state: string;
  latitude: number;
  longitude: number;
  coastal_proximity_km: number;
  has_paired_observations: boolean;
  cyclones_affected: string[];
}

export interface BeforeAfterPairInfo {
  event_id: string;
  cyclone_name: string;
  location_name: string;
  sensor_type: "OPTICAL" | "SAR";
  pre_observation: {
    source: string;
    platform: string;
    sensor: string;
    acquisition_time: string;
    bounds: {
      min_latitude: number;
      max_latitude: number;
      min_longitude: number;
      max_longitude: number;
    };
    cloud_coverage_percent: number | null;
    resolution_meters: number;
    file_id: string;
    sha256_hash: string;
  };
  post_observation: {
    source: string;
    platform: string;
    sensor: string;
    acquisition_time: string;
    bounds: {
      min_latitude: number;
      max_latitude: number;
      min_longitude: number;
      max_longitude: number;
    };
    cloud_coverage_percent: number | null;
    resolution_meters: number;
    file_id: string;
    sha256_hash: string;
  };
  temporal_baseline_days: number;
  spatial_overlap_percent: number;
  optical_cloud_screen_passed: boolean;
  co_registration_status: string;
  pairing_valid: boolean;
}

export interface ChangeClassMetrics {
  pixel_count: number;
  percentage: number;
  area_sq_km: number;
}

export interface ChangeDetectionSummary {
  analysis_id: string;
  methodology: string;
  total_pixels: number;
  pixel_resolution_meters: number;
  total_area_sq_km: number;
  classes: {
    NO_SIGNIFICANT_CHANGE?: ChangeClassMetrics;
    WATER_CHANGE?: ChangeClassMetrics;
    VEGETATION_CHANGE?: ChangeClassMetrics;
    SURFACE_CHANGE?: ChangeClassMetrics;
    UNCERTAIN?: ChangeClassMetrics;
    [key: string]: ChangeClassMetrics | undefined;
  };
  optical_indices?: {
    mean_ndvi_pre: number;
    mean_ndvi_post: number;
    mean_delta_ndvi: number;
    mean_ndwi_pre: number;
    mean_ndwi_post: number;
    mean_delta_ndwi: number;
  };
  sar_indices?: {
    mean_vv_pre_db: number;
    mean_vv_post_db: number;
    mean_delta_vv_db: number;
    inundation_pixel_count: number;
    inundation_area_sq_km: number;
  };
  provenance_hash: string;
  preview_grid?: number[][];
}

export interface EvidenceCitation {
  sensor_source: string;
  platform: string;
  product_id: string;
  pre_event_timestamp: string;
  post_event_timestamp: string;
  derived_layer_evaluated: string;
  area_sq_km_affected: number;
  bounding_box: {
    min_latitude: number;
    max_latitude: number;
    min_longitude: number;
    max_longitude: number;
  };
  data_quality_status: string;
  sha256_hash: string;
}

export interface GroundedAnswerResult {
  question: string;
  answer_text: string;
  observed_changes_summary: string;
  citations: EvidenceCitation[];
  confidence_level: "HIGH_CONFIDENCE" | "MEDIUM_CONFIDENCE" | "LOW_CONFIDENCE" | "INSUFFICIENT_DATA";
  uncertainty_factors: string[];
  scientific_disclaimer: string;
  provider_type: string;
  model_name: string;
  model_version: string;
  execution_timestamp: string;
}

export interface CycloneLandfallContext {
  storm_id: string;
  storm_name: string;
  season: number;
  landfall_timestamp: string;
  landfall_latitude: number;
  landfall_longitude: number;
  landfall_wind_kts: number;
  landfall_pressure_hpa: number | null;
  distance_to_target_km: number;
  intensity_category: number;
}

export interface ImpactIntelligenceReport {
  analysis_id: string;
  cyclone_name: string;
  location_name: string;
  sensor_type: "OPTICAL" | "SAR";
  cyclone_context: CycloneLandfallContext;
  pairing_metadata: BeforeAfterPairInfo;
  change_summary: ChangeDetectionSummary;
  ai_grounded_answer: GroundedAnswerResult;
  uncertainty_status: "HIGH_CONFIDENCE" | "MEDIUM_CONFIDENCE" | "LOW_CONFIDENCE" | "INSUFFICIENT_DATA";
  uncertainty_reasons: string[];
  scientific_disclaimer: string;
  provenance_lineage: {
    entity_id: string;
    pre_observation_hash: string;
    post_observation_hash: string;
    change_map_hash: string;
    answer_hash: string;
    software_version: string;
    w3c_prov_type: string;
  };
}

export interface ImpactQuestionResponse {
  analysis_id: string;
  question: string;
  answer: GroundedAnswerResult;
  retrieved_evidence_layers: string[];
  cyclone_context?: CycloneLandfallContext;
  change_metrics?: Record<string, ChangeClassMetrics>;
  execution_time_ms: number;
}

export interface LocationImpactProfile {
  location: {
    name: string;
    latitude: number;
    longitude: number;
    state?: string;
    basin?: string;
  };
  historical_exposure: {
    total_cyclones: number;
    peak_wind_kts: number;
    min_pressure_hpa: number | null;
  };
  satellite_archive_status: {
    optical_observations_count: number;
    sar_observations_count: number;
    change_analyses_available: number;
  };
  recent_impact_events: Array<{
    cyclone_name: string;
    season: number;
    closest_distance_km: number;
    wind_experienced_kts: number;
    has_satellite_change_analysis: boolean;
  }>;
}

