/**
 * CycloneSense API Client
 * Strongly-typed asynchronous HTTP client connecting to the real FastAPI backend.
 * Zero mocks or synthetic placeholders.
 */

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

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
  const response = await fetch(url, {
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
};
