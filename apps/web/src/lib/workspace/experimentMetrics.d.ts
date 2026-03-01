export type ExperimentTask = "classification" | "detection" | "segmentation";

export interface MetricPoint {
  epoch: number;
  train_loss?: number;
  val_loss?: number;
  val_accuracy?: number;
  val_map?: number;
  val_iou?: number;
}

export interface MetricDomain {
  min: number;
  max: number;
}

export interface BuildLineOptions {
  width?: number;
  height?: number;
  padding?: number;
  seriesKeys?: string[];
}

export interface CheckpointRow {
  kind: "best_metric" | "best_loss" | "latest";
  epoch: number | null;
  metric_name: string | null;
  value: number | null;
  updated_at?: string | null;
}

export function metricKeyForTask(task: ExperimentTask | string): "val_accuracy" | "val_map" | "val_iou";
export function mergeMetricPoints(existing: MetricPoint[], incoming: MetricPoint[]): MetricPoint[];
export function metricDomain(metrics: MetricPoint[], seriesKeys: string[]): MetricDomain;
export function buildLinePoints(metrics: MetricPoint[], seriesKey: string, options?: BuildLineOptions): string;
export function indexCheckpointsByKind(checkpoints: CheckpointRow[]): {
  best_metric: CheckpointRow | null;
  best_loss: CheckpointRow | null;
  latest: CheckpointRow | null;
};
