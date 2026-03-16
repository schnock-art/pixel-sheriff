export type ExperimentTask = "classification" | "detection" | "segmentation";

export interface MetricPoint {
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
  mAP50?: number;
  mAP50_95?: number;
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

export interface MetricLegendItem {
  key: string;
  label: string;
  color: string;
  enabled: boolean;
  axis: "left" | "right";
}

export interface ExperimentMetricChartModel {
  chartWidth: number;
  chartHeight: number;
  chartPadding: number;
  chartInnerWidth: number;
  chartInnerHeight: number;
  chartKeys: string[];
  chartMaxEpoch: number;
  primaryMetricIsBounded: boolean;
  useSecondaryAxis: boolean;
  leftAxisDomain: MetricDomain;
  rightAxisDomain: MetricDomain | null;
  leftAxisTicks: number[];
  rightAxisTicks: number[];
  xTickValues: number[];
  seriesLegend: MetricLegendItem[];
  primaryLinePoints: string;
  valLossLinePoints: string;
}

export interface ExperimentMetricHoverPlotRow {
  key: string;
  label: string;
  color: string;
  value: number;
  y: number;
}

export interface ExperimentMetricHoverModel {
  hoveredMetric: MetricPoint | null;
  hoveredEpochValue: number | null;
  hoveredX: number | null;
  hoveredSeriesValues: Array<{
    key: string;
    label: string;
    color: string;
    value: number;
  }>;
  hoveredPlotRows: ExperimentMetricHoverPlotRow[];
  hoverTooltip: {
    x: number;
    y: number;
    width: number;
    height: number;
  } | null;
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
export function metricValueByKey(row: MetricPoint | Record<string, unknown>, key: string): number | null;
export function findNearestMetricEpoch(metrics: MetricPoint[], approximateEpoch: number): number | null;
export function buildExperimentMetricChartModel(
  metrics: MetricPoint[],
  options?: {
    primaryMetricKey?: string;
    primaryMetricLabel?: string;
    showPrimary?: boolean;
    showValLoss?: boolean;
    chartWidth?: number;
    chartHeight?: number;
    chartPadding?: number;
    primaryColor?: string;
    lossColor?: string;
  },
): ExperimentMetricChartModel;
export function buildExperimentMetricHoverModel(
  metrics: MetricPoint[],
  options?: {
    hoveredEpoch?: number | null;
    seriesLegend?: MetricLegendItem[];
    chartWidth?: number;
    chartHeight?: number;
    chartPadding?: number;
    chartInnerWidth?: number;
    chartInnerHeight?: number;
    chartMaxEpoch?: number;
    leftAxisDomain?: MetricDomain;
    lossDomain?: MetricDomain;
    useSecondaryAxis?: boolean;
  },
): ExperimentMetricHoverModel;
export function indexCheckpointsByKind(checkpoints: CheckpointRow[]): {
  best_metric: CheckpointRow | null;
  best_loss: CheckpointRow | null;
  latest: CheckpointRow | null;
};
