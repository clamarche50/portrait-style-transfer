import type { components } from "./schema";

type ApiSchemas = components["schemas"];

export type AssetKind = ApiSchemas["AssetKind"];
export type JobStatus = ApiSchemas["JobStatus"];
export type BackgroundMode = ApiSchemas["BackgroundMode"];
export type OutputFormat = ApiSchemas["OutputFormat"];

export interface QualityWarning {
  code: string;
  message: string;
  severity: "info" | "warning" | "error";
}

export interface PortraitAnalysisSummary {
  face_box?: { x: number; y: number; width: number; height: number };
  pose?: { yaw: number; pitch: number; roll: number };
  quality_score?: number;
  warnings?: QualityWarning[];
}

export interface AssetRecord {
  id: string;
  kind: AssetKind;
  mime_type: string;
  width: number;
  height: number;
  byte_size: number;
  created_at: string;
  expires_at: string;
  preview_url?: string | null;
  analysis?: PortraitAnalysisSummary | null;
}

export type TransferSettings = Omit<ApiSchemas["TransferSettingsRequest"], "algorithm_profile" | "background_color"> & {
  algorithm_profile: "source_2014_compat" | "paper_exact";
  background_color: string | null;
};

export interface JobArtifact {
  id?: string;
  kind: string;
  label?: string;
  url?: string;
  width?: number;
  height?: number;
}

export interface JobDiagnosticsRecord {
  job_id: string;
  diagnostics: Record<string, unknown>;
  artifacts: Array<{
    asset_id: string;
    kind: string;
    download_url?: string | null;
  }>;
}

export interface JobRecord {
  id: string;
  status: JobStatus;
  stage: string;
  progress: number;
  input_asset_id: string;
  reference_asset_id?: string | null;
  style_id?: string | null;
  settings: TransferSettings;
  warnings?: Array<QualityWarning | string>;
  output_asset_id?: string | null;
  output_url?: string | null;
  input_preview_url?: string | null;
  diagnostics_summary?: Record<string, unknown>;
  diagnostics?: Record<string, unknown>;
  artifacts?: JobArtifact[];
  error_code?: string | null;
  error_message_safe?: string | null;
  created_at: string;
  finished_at?: string | null;
}

export interface StyleRecord {
  id: string;
  name: string;
  description: string;
  rights_confirmed: boolean;
  example_count?: number;
  preview_url?: string | null;
  created_at: string;
  updated_at: string;
}

export interface CorrectionPayload {
  type: "mask" | "alignment" | "gain_copy" | "eye" | "background";
  [key: string]: unknown;
}
