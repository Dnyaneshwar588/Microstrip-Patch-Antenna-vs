// ─── Shared types ──────────────────────────────────────────────────────────────

export interface DatasetMetadata {
  rows: number;
  columns: number;
  numeric_columns: number;
  missing_values: number;
  duplicate_rows: number;
  non_numeric_columns: string[];
  numeric_ranges: Record<string, { min: number; max: number }>;
  s11_points?: number;
  frequency_labels_verified?: boolean;
  frequency_available?: boolean;
  input_rows?: number;
}

export interface Dataset {
  rows: number;
  columns: string[];
  inputColumns: string[];
  targetColumns: string[];
  warnings: string[];
  metadata?: DatasetMetadata;
  frequencyHz?: number[];
  preview: { columns: string[]; rows: Record<string, unknown>[]; total: number };
}

export interface Evaluation {
  model: string;
  r2: number;
  rmse: number;
  mae: number;
  explained_variance?: number;
}

export interface OptimizationResult {
  algorithm: string;
  best_fitness: number;
  best_position: number[];
  history: number[];
  elapsed_seconds: number;
  objective_evaluations?: number;
}

export interface CompareRow {
  algorithm: string;
  best_fitness: number;
  average_fitness: number;
  worst_fitness: number;
  std_fitness: number;
  elapsed_seconds: number;
  objective_evaluations: number;
  history: number[];
  best_position: number[];
}

export interface CompareResult {
  results: CompareRow[];
  best_algorithm: string;
}

export interface ResponseMetrics {
  minimum_s11_db: number;
  minimum_s11_index: number;
  reflection_coefficient_magnitude: number[];
  vswr: number[];
  frequency_available: boolean;
  resonant_frequency_hz?: number;
  bandwidth_hz?: number;
  fractional_bandwidth?: number;
  frequency_hz?: number[];
  threshold_db?: number;
  note?: string;
}

export interface PredictResponse {
  prediction: Record<string, number>;
  metrics: ResponseMetrics;
  warnings: string[];
}

export interface HfssCompareResponse {
  validation_status: string;
  frequency_hz: number[];
  predicted_s11_db: number[];
  hfss_s11_db: number[];
  mae_db: number;
  rmse_db: number;
  predicted_metrics: ResponseMetrics;
  hfss_metrics: ResponseMetrics;
}

export interface HfssSimpleResponse {
  columns: string[];
  rows: Record<string, unknown>[];
  total: number;
  validation_status: string;
}

export interface AppState {
  dataset: Dataset | null;
  models: string[];
  evaluation: Evaluation[] | null;
  optimizations: Record<string, OptimizationResult>;
}
