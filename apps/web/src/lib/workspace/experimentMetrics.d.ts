export type ExperimentTask = "classification" | "detection" | "segmentation";

export interface MetricPoint {
  epoch: number;
  train_loss?: number;
  val_loss?: number;
  val_accuracy?: number;
  val_map?: number;
  val_map_50_95?: number;
  val_iou?: number;
}

export interface MetricDomain {
  min: number;
  max: number;
}

export interface TickBuildOptions {
  useLog?: boolean;
  count?: number;
  clamp01?: boolean;
}

export interface TickFormatOptions {
  useLog?: boolean;
  bounded?: boolean;
}

export interface BuildLineOptions {
  width?: number;
  height?: number;
  padding?: number;
  seriesKeys?: string[];
  domain?: MetricDomain;
  useLog?: boolean;
}

export interface CheckpointRow {
  kind: "best_metric" | "best_loss" | "latest";
  epoch: number | null;
  metric_name: string | null;
  value: number | null;
  updated_at?: string | null;
}

export function metricKeyForTask(task: ExperimentTask | string): "val_accuracy" | "val_map" | "val_iou";
export function isLossMetricKey(key: string): boolean;
export function isBoundedMetricKey(key: string): boolean;
export function isBoundedSeries(rowsOrPoints: Array<Record<string, unknown>>, key?: string): boolean;
export function computeSeriesDomain(values: Array<number | string | null | undefined>, options?: { useLog?: boolean; clamp01?: boolean }): MetricDomain;
export function buildTicks(domain: MetricDomain, options?: TickBuildOptions): number[];
export function formatTick(value: number, options?: TickFormatOptions): string;
export function mergeMetricPoints(existing: MetricPoint[], incoming: MetricPoint[]): MetricPoint[];
export function metricDomain(metrics: MetricPoint[], seriesKeys: string[], options?: { useLog?: boolean; clampBounded?: boolean }): MetricDomain;
export function buildLinePoints(metrics: MetricPoint[], seriesKey: string, options?: BuildLineOptions): string;
export function indexCheckpointsByKind(checkpoints: CheckpointRow[]): {
  best_metric: CheckpointRow | null;
  best_loss: CheckpointRow | null;
  latest: CheckpointRow | null;
};
