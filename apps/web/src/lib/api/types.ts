export type AnnotationStatus = "unlabeled" | "labeled" | "skipped" | "needs_review" | "approved";
export type ProjectTaskType = "classification" | "classification_single" | "bbox" | "segmentation";
export type TaskKind = "classification" | "bbox" | "segmentation";
export type TaskLabelMode = "single_label" | "multi_label";
export type PrelabelSourceType = "active_deployment" | "florence2";
export type PrelabelSamplingMode = "every_n_frames" | "every_n_seconds";
export type PrelabelSessionStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
export type PrelabelProposalStatus = "pending" | "accepted" | "edited" | "rejected";
export type PrelabelDebugDetectionStatus = "matched" | "unmatched" | "discarded";

export interface Project {
  id: string;
  name: string;
  task_type: ProjectTaskType;
  default_task_id: string | null;
  schema_version: string;
}

export interface ProjectCreatePayload {
  name: string;
  task_type?: ProjectTaskType;
}

export interface Category {
  id: string;
  project_id: string;
  task_id: string;
  name: string;
  display_order: number;
  is_active: boolean;
}

export interface CategoryCreatePayload {
  task_id: string;
  name: string;
  display_order?: number;
}

export interface CategoryUpdatePayload {
  name?: string;
  display_order?: number;
  is_active?: boolean;
}

export interface Asset {
  id: string;
  project_id: string;
  type: string;
  folder_id: string | null;
  folder_path: string | null;
  file_name: string | null;
  relative_path: string | null;
  sequence_id: string | null;
  source_kind: "image" | "video_frame" | "webcam_frame";
  frame_index: number | null;
  timestamp_seconds: number | null;
  uri: string;
  mime_type: string;
  width: number | null;
  height: number | null;
  checksum: string;
  metadata_json: Record<string, unknown>;
}

export interface AssetCreatePayload {
  type?: "image" | "video" | "frame";
  folder_id?: string | null;
  file_name?: string | null;
  sequence_id?: string | null;
  source_kind?: "image" | "video_frame" | "webcam_frame";
  frame_index?: number | null;
  timestamp_seconds?: number | null;
  uri: string;
  mime_type: string;
  width?: number | null;
  height?: number | null;
  checksum: string;
  metadata_json?: Record<string, unknown>;
}

export interface Folder {
  id: string;
  project_id: string;
  parent_id: string | null;
  name: string;
  path: string;
  asset_count: number;
  sequence_id: string | null;
  sequence_name: string | null;
  sequence_source_type: "video_file" | "webcam" | null;
  sequence_status: "processing" | "ready" | "failed" | null;
  sequence_frame_count: number | null;
  sequence_processed_frames: number | null;
}

export interface SequenceFrameAsset {
  id: string;
  file_name: string;
  folder_id: string | null;
  folder_path: string | null;
  relative_path: string;
  source_kind: "image" | "video_frame" | "webcam_frame";
  frame_index: number | null;
  timestamp_seconds: number | null;
  image_url: string;
  thumbnail_url: string;
  has_annotations: boolean;
  pending_prelabel_count: number;
}

export interface AssetSequence {
  id: string;
  project_id: string;
  task_id: string | null;
  folder_id: string | null;
  folder_path: string | null;
  name: string;
  source_type: "video_file" | "webcam";
  source_filename: string | null;
  status: "processing" | "ready" | "failed";
  frame_count: number;
  processed_frames: number;
  fps: number | null;
  duration_seconds: number | null;
  width: number | null;
  height: number | null;
  error_message: string | null;
  pending_prelabel_count: number;
  latest_prelabel_session_id: string | null;
  latest_prelabel_session_status: PrelabelSessionStatus | null;
  assets: SequenceFrameAsset[];
}

export interface SequenceStatus {
  id: string;
  status: "processing" | "ready" | "failed";
  frame_count: number;
  processed_frames: number;
  error_message: string | null;
  pending_prelabel_count: number;
}

export interface PrelabelFrameSampling {
  mode: PrelabelSamplingMode;
  value: number;
}

export interface PrelabelConfig {
  source_type: PrelabelSourceType;
  prompts: string[];
  frame_sampling: PrelabelFrameSampling;
  confidence_threshold: number;
  max_detections_per_frame: number;
}

export interface PrelabelSourceStatus {
  ok: boolean;
  source_type: PrelabelSourceType;
  source_ref: string | null;
  source_label: string;
  device_selected: "cuda" | "cpu" | null;
  device_preference: DeploymentDevicePreference | null;
}

export interface PrelabelSession {
  id: string;
  project_id: string;
  task_id: string;
  sequence_id: string;
  source_type: PrelabelSourceType;
  source_ref: string | null;
  source_label: string | null;
  device_preference: DeploymentDevicePreference | null;
  prompts: string[];
  sampling_mode: PrelabelSamplingMode;
  sampling_value: number;
  confidence_threshold: number;
  max_detections_per_frame: number;
  live_mode: boolean;
  status: PrelabelSessionStatus;
  input_closed_at: string | null;
  enqueued_assets: number;
  processed_assets: number;
  generated_proposals: number;
  skipped_unmatched: number;
  error_message: string | null;
  debug_detections: PrelabelDebugDetection[];
  created_at: string | null;
  updated_at: string | null;
}

export interface PrelabelDebugDetection {
  asset_id: string;
  asset_frame_index: number | null;
  label_text: string;
  resolved_category_id: string | null;
  resolved_category_name: string | null;
  confidence: number;
  bbox_xyxy: number[];
  status: PrelabelDebugDetectionStatus;
}

export interface PrelabelProposal {
  id: string;
  session_id: string;
  asset_id: string;
  project_id: string;
  task_id: string;
  category_id: string;
  label_text: string;
  prompt_text: string | null;
  confidence: number;
  bbox: number[];
  status: PrelabelProposalStatus;
  reviewed_bbox: number[] | null;
  reviewed_category_id: string | null;
  promoted_annotation_id: string | null;
  promoted_object_id: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface WebcamSessionCreatePayload {
  task_id?: string | null;
  folder_id?: string | null;
  folder_path?: string | null;
  name: string;
  fps: number;
  prelabel_config?: PrelabelConfig | null;
}

export interface VideoImportPayload {
  task_id?: string | null;
  folder_id?: string | null;
  name?: string | null;
  fps: number;
  max_frames: number;
  resize_mode: "original" | "width" | "height";
  resize_width?: number | null;
  resize_height?: number | null;
  prelabel_config?: PrelabelConfig | null;
}

export interface VideoImportResponse {
  sequence: AssetSequence;
  prelabel_session_id: string | null;
}

export interface WebcamSessionCreateResponse {
  sequence: AssetSequence;
  prelabel_session_id: string | null;
}

export interface Annotation {
  id: string;
  asset_id: string;
  project_id: string;
  task_id: string;
  status: AnnotationStatus;
  payload_json: Record<string, unknown>;
  annotated_by: string | null;
}

export interface AnnotationUpsert {
  task_id: string;
  asset_id: string;
  status: AnnotationStatus;
  payload_json: Record<string, unknown>;
  annotated_by?: string;
}

export interface Task {
  id: string;
  project_id: string;
  name: string;
  kind: TaskKind;
  label_mode: TaskLabelMode | null;
  created_at: string | null;
  updated_at: string | null;
  is_default: boolean;
}

export interface TaskCreatePayload {
  name: string;
  kind: TaskKind;
  label_mode?: TaskLabelMode;
}

export interface ExportCreatePayload {
  selection_criteria_json?: Record<string, unknown>;
}

export interface ExportVersion {
  id: string;
  project_id: string;
  selection_criteria_json: Record<string, unknown>;
  manifest_json: Record<string, unknown>;
  export_uri: string;
  hash: string;
}

export interface DatasetSelectionFilters {
  include_labeled_only?: boolean;
  include_statuses?: AnnotationStatus[];
  exclude_statuses?: AnnotationStatus[];
  include_category_ids?: string[];
  exclude_category_ids?: string[];
  include_folder_paths?: string[];
  exclude_folder_paths?: string[];
  include_negative_images?: boolean;
}

export interface DatasetSplitConfig {
  seed?: number;
  ratios?: { train: number; val: number; test: number };
  stratify?: {
    enabled?: boolean;
    by?: "label_primary" | "label_multi_hot" | "embedding_cluster";
    strict_stratify?: boolean;
  };
}

export interface DatasetVersionSummaryEnvelope {
  version: Record<string, unknown>;
  is_archived: boolean;
  is_active: boolean;
}

export interface DatasetVersionListPayload {
  active_dataset_version_id: string | null;
  items: DatasetVersionSummaryEnvelope[];
}

export interface DatasetPreviewPayload {
  asset_ids: string[];
  sample_asset_ids: string[];
  sample_assets: Array<{
    asset_id: string;
    filename: string;
    relative_path: string;
    status: AnnotationStatus;
    split?: "train" | "val" | "test" | null;
    label_summary: Record<string, unknown>;
  }>;
  class_names: Record<string, string>;
  counts: {
    total: number;
    class_counts: Record<string, number>;
    split_counts: { train: number; val: number; test: number };
  };
  warnings: string[];
}

export interface DatasetVersionAssetsPayload {
  items: Array<{
    asset_id: string;
    filename: string;
    relative_path: string;
    status: AnnotationStatus;
    split?: "train" | "val" | "test" | null;
    label_summary: Record<string, unknown>;
  }>;
  page: number;
  page_size: number;
  total: number;
}

export interface DatasetVersionExportPayload {
  dataset_version_id: string;
  hash: string;
  export_uri: string;
}

export interface ProjectModelSummary {
  id: string;
  task_id: string | null;
  name: string;
  created_at: string;
  updated_at: string;
  task: string;
  backbone_name: string;
  num_classes: number;
}

export interface ProjectModelCreatePayload {
  name?: string;
  dataset_version_id?: string;
}

export interface ProjectModelCreateResponse {
  id: string;
  name: string;
  config: Record<string, unknown>;
}

export interface ProjectModelRecord {
  id: string;
  project_id: string;
  task_id: string | null;
  name: string;
  config_json: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ProjectModelUpdatePayload {
  config_json: Record<string, unknown>;
}

export type ExperimentStatus = "draft" | "queued" | "running" | "completed" | "failed" | "canceled";
export type ExperimentTask = "classification" | "detection" | "segmentation";

export interface ExperimentSummaryJson {
  best_metric_name: string | null;
  best_metric_value: number | null;
  best_epoch: number | null;
  last_epoch: number | null;
}

export interface ExperimentCheckpoint {
  kind: "best_loss" | "best_metric" | "latest";
  epoch: number | null;
  metric_name: string | null;
  value: number | null;
  uri?: string | null;
  updated_at: string | null;
  status?: "pending" | "ok" | "error" | null;
  error?: string | null;
}

export interface ExperimentMetricPoint {
  attempt?: number | null;
  epoch: number;
  train_loss?: number;
  train_accuracy?: number;
  val_loss?: number;
  val_accuracy?: number;
  val_macro_f1?: number;
  val_macro_precision?: number;
  val_macro_recall?: number;
  val_map?: number;
  val_map_50_95?: number;
  val_iou?: number;
  epoch_seconds?: number;
  eta_seconds?: number;
  created_at?: string;
}

export interface ProjectExperimentSummary {
  id: string;
  project_id: string;
  task_id: string | null;
  task?: string | null;
  model_id: string;
  name: string;
  created_at: string;
  updated_at: string;
  status: ExperimentStatus;
  summary_json: ExperimentSummaryJson;
  current_run_attempt?: number | null;
  last_completed_attempt?: number | null;
  active_job_id?: string | null;
  error?: string | null;
}

export interface ProjectExperimentRecord extends ProjectExperimentSummary {
  config_json: Record<string, unknown>;
  artifacts_json: Record<string, unknown>;
  checkpoints: ExperimentCheckpoint[];
  metrics: ExperimentMetricPoint[];
}

export interface ProjectExperimentListResponse {
  items: ProjectExperimentSummary[];
}

export interface ProjectExperimentCreatePayload {
  model_id: string;
  dataset_version_id?: string;
  name?: string;
  config_overrides?: Record<string, unknown>;
}

export interface ProjectExperimentUpdatePayload {
  name?: string;
  config_json?: Record<string, unknown>;
  selected_checkpoint_kind?: "best_loss" | "best_metric" | "latest";
}

export interface ExperimentActionResponse {
  ok: boolean;
  status?: ExperimentStatus | null;
  attempt?: number | null;
  job_id?: string | null;
}

export interface ExperimentAnalyticsBest {
  metric_name?: string | null;
  metric_value?: number | null;
  epoch?: number | null;
}

export interface ExperimentAnalyticsConfig {
  optimizer?: { type?: string | null; lr?: number | null };
  batch_size?: number | null;
  epochs?: number | null;
  augmentation?: string | null;
}

export interface ExperimentAnalyticsItem {
  experiment_id: string;
  name: string;
  model_id: string;
  model_name: string;
  task_id?: string | null;
  task?: string | null;
  status: ExperimentStatus;
  updated_at: string;
  config: ExperimentAnalyticsConfig;
  best: ExperimentAnalyticsBest;
  final: Record<string, number | null>;
  series: Record<string, unknown>;
  runtime?: { device_selected?: string } | null;
}

export interface ProjectExperimentAnalyticsResponse {
  items: ExperimentAnalyticsItem[];
  available_series: string[];
}

export interface ExperimentEvaluationOverall {
  accuracy?: number;
  macro_f1?: number;
  macro_precision?: number;
  macro_recall?: number;
  mAP50?: number;
  mAP50_95?: number;
}

export interface ExperimentEvaluationPerClassRow {
  class_index: number;
  class_id: string;
  name: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ExperimentEvaluationSampleRow {
  asset_id: string;
  relative_path?: string;
  true_class_index: number;
  pred_class_index: number;
  confidence: number;
  margin?: number | null;
}

export interface ExperimentEvaluationPayload {
  attempt: number;
  schema_version?: string;
  task?: string;
  computed_at?: string;
  split?: string;
  num_samples?: number;
  classes?: {
    class_order?: string[];
    class_names?: string[];
    id_to_index?: Record<string, number>;
  };
  overall?: ExperimentEvaluationOverall;
  per_class?: ExperimentEvaluationPerClassRow[];
  confusion_matrix?: {
    matrix?: number[][];
    normalized_by?: string;
    labels?: Record<string, string>;
  };
  samples?: {
    misclassified?: ExperimentEvaluationSampleRow[];
    lowest_confidence_correct?: ExperimentEvaluationSampleRow[];
    highest_confidence_wrong?: ExperimentEvaluationSampleRow[];
  };
}

export interface ExperimentSamplesResponse {
  attempt: number;
  mode: "misclassified" | "lowest_confidence_correct" | "highest_confidence_wrong";
  items: ExperimentEvaluationSampleRow[];
  message?: string | null;
}

export interface ExperimentRuntimePayload {
  attempt: number;
  device_selected: string;
  cuda_available: boolean;
  mps_available: boolean;
  amp_enabled: boolean;
  torch_version: string;
  torchvision_version: string;
  num_workers: number;
  pin_memory: boolean;
  persistent_workers: boolean;
  prefetch_factor?: number;
  cache_resized_images?: boolean;
  max_cached_images?: number;
}

export interface ExperimentOnnxPayload {
  attempt: number;
  status: "exported" | "failed";
  model_onnx_url?: string | null;
  metadata_url: string;
  input_shape?: number[];
  class_names?: string[];
  class_order?: string[];
  preprocess?: Record<string, unknown>;
  validation?: Record<string, unknown> | null;
  error?: string | null;
}

export interface ExperimentLogsChunk {
  attempt: number;
  from_byte: number;
  to_byte: number;
  content: string;
}

export type DeploymentDevicePreference = "auto" | "cuda" | "cpu";
export type DeploymentStatus = "available" | "archived";

export interface DeploymentSource {
  experiment_id: string;
  attempt: number;
  checkpoint_kind: "best_metric" | "best_loss" | "latest";
  onnx_relpath: string;
  metadata_relpath: string;
}

export interface DeploymentItem {
  deployment_id: string;
  name: string;
  task_id: string | null;
  task: TaskKind;
  provider: "onnxruntime";
  device_preference: DeploymentDevicePreference;
  model_key: string;
  source: DeploymentSource;
  status: DeploymentStatus;
  created_at: string;
  updated_at: string;
}

export interface DeploymentListResponse {
  active_deployment_id: string | null;
  items: DeploymentItem[];
}

export interface CreateDeploymentPayload {
  name: string;
  task?: TaskKind;
  device_preference?: DeploymentDevicePreference;
  source: {
    experiment_id: string;
    attempt: number;
    checkpoint_kind?: "best_metric" | "best_loss" | "latest";
  };
  is_active?: boolean;
}

export interface PatchDeploymentPayload {
  is_active?: boolean;
  name?: string;
  device_preference?: DeploymentDevicePreference;
  status?: DeploymentStatus;
}

export interface PredictPayload {
  asset_id: string;
  deployment_id?: string | null;
  top_k?: number;
  score_threshold?: number;
}

export interface PredictBatchPayload {
  asset_ids: string[];
  deployment_id?: string | null;
  top_k?: number;
  score_threshold?: number;
}

export interface PredictPrediction {
  class_index: number;
  class_id: string;
  class_name: string;
  score: number;
}

export interface PredictDetectionBox {
  class_index: number;
  class_id: string;
  class_name: string;
  score: number;
  bbox: number[];
}

export interface PredictClassificationResponse {
  asset_id: string;
  deployment_id: string;
  task: "classification";
  device_selected: "cuda" | "cpu";
  predictions: PredictPrediction[];
  deployment_name?: string | null;
  device_preference?: DeploymentDevicePreference | null;
}

export interface PredictBBoxResponse {
  asset_id: string;
  deployment_id: string;
  task: "bbox";
  device_selected: "cuda" | "cpu";
  boxes: PredictDetectionBox[];
  deployment_name?: string | null;
  device_preference?: DeploymentDevicePreference | null;
}

export type PredictResponse = PredictClassificationResponse | PredictBBoxResponse;

export interface PredictBatchError {
  asset_id: string;
  code: string;
  message: string;
}

export interface PredictBatchResponse {
  deployment_id: string;
  task: "classification" | "bbox";
  requested_count: number;
  completed_count: number;
  pending_review_count: number;
  empty_count: number;
  error_count: number;
  predictions: PredictResponse[];
  errors: PredictBatchError[];
  deployment_name?: string | null;
  device_preference?: DeploymentDevicePreference | null;
}

export type ExperimentEvent =
  | { type: "status"; status: ExperimentStatus; attempt?: number; job_id?: string; ts?: string; message?: string }
  | ({ type: "metric"; attempt?: number; ts?: string } & ExperimentMetricPoint)
  | ({ type: "checkpoint"; attempt?: number; ts?: string } & ExperimentCheckpoint)
  | { type: "onnx_export"; status: "exported" | "failed"; attempt?: number; model_uri?: string; metadata_uri?: string; error?: string; ts?: string }
  | { type: "done"; status: ExperimentStatus; attempt?: number; job_id?: string; ts?: string; message?: string; error_code?: string };

export interface ExperimentEventEnvelope {
  line: number;
  attempt: number | null;
  event: ExperimentEvent;
}

export interface StreamExperimentHandlers {
  onEvent?: (event: ExperimentEvent) => void;
  onEnvelope?: (payload: ExperimentEventEnvelope) => void;
  onError?: (event: Event) => void;
}
