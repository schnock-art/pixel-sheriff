"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useProjectNavigationGuard } from "../../../../../components/workspace/ProjectNavigationContext";
import {
  ApiError,
  cancelExperiment,
  getExperiment,
  getExperimentEvaluation,
  getExperimentLogs,
  getExperimentRuntime,
  listExperimentSamples,
  listProjectModels,
  startExperiment,
  streamExperimentEvents,
  updateExperiment,
  type ExperimentCheckpoint,
  type ExperimentEvaluationPayload,
  type ExperimentEvaluationSampleRow,
  type ExperimentMetricPoint,
  type ExperimentRuntimePayload,
  type ExperimentStatus,
  type ProjectExperimentRecord,
} from "../../../../../lib/api";
import {
  buildLinePoints,
  buildTicks,
  computeSeriesDomain,
  formatTick,
  indexCheckpointsByKind,
  isBoundedMetricKey,
  mergeMetricPoints,
  metricKeyForTask,
} from "../../../../../lib/workspace/experimentMetrics";
import { filterPredictionRows, normalizeConfusion } from "../../../../../lib/workspace/experimentDashboard";
import { mergeLogChunk, runtimeBadgeLabel } from "../../../../../lib/workspace/experimentRuntime";

interface ExperimentDetailPageProps {
  params: {
    projectId: string;
    experimentId: string;
  };
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function cloneConfig(value: Record<string, unknown>): Record<string, unknown> {
  return JSON.parse(JSON.stringify(value)) as Record<string, unknown>;
}

function parseApiErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof ApiError && error.responseBody) {
    try {
      const parsed = JSON.parse(error.responseBody) as { error?: { message?: string } };
      if (parsed.error?.message) return parsed.error.message;
      return error.responseBody;
    } catch {
      return error.responseBody;
    }
  }
  if (error instanceof Error) return error.message;
  return fallback;
}

function configValidation(config: Record<string, unknown>): { isValid: boolean; issues: string[] } {
  const issues: string[] = [];
  const optimizer = asRecord(config.optimizer);
  const lr = Number(optimizer.lr);
  const epochs = Number(config.epochs);
  const batchSize = Number(config.batch_size);
  if (!Number.isFinite(lr) || lr <= 0) issues.push("Learning rate must be > 0");
  if (!Number.isFinite(epochs) || epochs < 1) issues.push("Epochs must be >= 1");
  if (!Number.isFinite(batchSize) || batchSize < 1) issues.push("Batch size must be >= 1");
  return { isValid: issues.length === 0, issues };
}

function formatDateTime(value: string | null | undefined): string {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "-";
  return parsed.toLocaleString();
}

function formatCheckpoint(checkpoint: Pick<ExperimentCheckpoint, "epoch" | "metric_name" | "value"> | null): string {
  if (!checkpoint || checkpoint.epoch == null) return "Not available yet";
  const metricName = checkpoint.metric_name ?? "metric";
  const value = typeof checkpoint.value === "number" ? checkpoint.value.toFixed(4) : "-";
  return `epoch ${checkpoint.epoch} | ${metricName}: ${value}`;
}

function patchNumber(value: string): number | null {
  if (value.trim() === "") return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return null;
  return parsed;
}

function asYesNo(value: boolean | null | undefined): string {
  if (value === true) return "yes";
  if (value === false) return "no";
  return "-";
}

function asEpoch(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : Number.parseInt(String(value), 10);
  if (!Number.isFinite(parsed) || parsed < 1) return null;
  return parsed;
}

function asMetricValue(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return value;
}

function metricValueByKey(row: ExperimentMetricPoint, key: string): number | null {
  if (key === "train_loss") return asMetricValue(row.train_loss);
  if (key === "val_loss") return asMetricValue(row.val_loss);
  if (key === "val_accuracy") return asMetricValue(row.val_accuracy);
  if (key === "val_macro_f1") return asMetricValue(row.val_macro_f1);
  if (key === "val_macro_precision") return asMetricValue(row.val_macro_precision);
  if (key === "val_macro_recall") return asMetricValue(row.val_macro_recall);
  if (key === "val_map") return asMetricValue(row.val_map);
  if (key === "val_iou") return asMetricValue(row.val_iou);
  return null;
}

export default function ExperimentDetailPage({ params }: ExperimentDetailPageProps) {
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);
  const experimentId = useMemo(() => decodeURIComponent(params.experimentId), [params.experimentId]);
  const { setHasUnsavedDrafts } = useProjectNavigationGuard();

  const [savedRecord, setSavedRecord] = useState<ProjectExperimentRecord | null>(null);
  const [draftName, setDraftName] = useState("");
  const [draftConfig, setDraftConfig] = useState<Record<string, unknown> | null>(null);
  const [metrics, setMetrics] = useState<ExperimentMetricPoint[]>([]);
  const [checkpoints, setCheckpoints] = useState<ExperimentCheckpoint[]>([]);
  const [status, setStatus] = useState<ExperimentStatus>("draft");
  const [modelName, setModelName] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [isCanceling, setIsCanceling] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [lastRunMessage, setLastRunMessage] = useState<string | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [toastTone, setToastTone] = useState<"success" | "error">("success");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showValLoss, setShowValLoss] = useState(true);
  const [showPrimary, setShowPrimary] = useState(true);
  const [hoveredEpoch, setHoveredEpoch] = useState<number | null>(null);
  const [activeAttempt, setActiveAttempt] = useState<number | null>(null);
  const [evaluation, setEvaluation] = useState<ExperimentEvaluationPayload | null>(null);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);
  const [isEvaluationLoading, setIsEvaluationLoading] = useState(false);
  const [runtimeInfo, setRuntimeInfo] = useState<ExperimentRuntimePayload | null>(null);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [logsContent, setLogsContent] = useState("");
  const [logsCursor, setLogsCursor] = useState(0);
  const [logsError, setLogsError] = useState<string | null>(null);
  const [isLogsLoading, setIsLogsLoading] = useState(false);
  const [isLogsExpanded, setIsLogsExpanded] = useState(true);
  const [logsAutoRefresh, setLogsAutoRefresh] = useState(true);
  const [dashboardChartTab, setDashboardChartTab] = useState<"loss" | "accuracy" | "prf">("loss");
  const [dashboardLogScale, setDashboardLogScale] = useState(false);
  const [confusionNormalize, setConfusionNormalize] = useState<"none" | "by_true" | "by_pred">("none");
  const [perClassSort, setPerClassSort] = useState<"f1_desc" | "f1_asc" | "precision_desc" | "recall_desc" | "support_desc">("f1_desc");
  const [predictionMode, setPredictionMode] = useState<"misclassified" | "lowest_confidence_correct" | "highest_confidence_wrong">("misclassified");
  const [predictionLimit, setPredictionLimit] = useState(50);
  const [predictionTrueClass, setPredictionTrueClass] = useState<string>("all");
  const [predictionPredClass, setPredictionPredClass] = useState<string>("all");
  const [cellDrawer, setCellDrawer] = useState<{ trueClassIndex: number; predClassIndex: number } | null>(null);
  const [cellSamples, setCellSamples] = useState<ExperimentEvaluationSampleRow[]>([]);
  const [cellSamplesMessage, setCellSamplesMessage] = useState<string | null>(null);
  const [selectedSampleImage, setSelectedSampleImage] = useState<ExperimentEvaluationSampleRow | null>(null);
  const eventCursorRef = useRef(0);
  const logsCursorRef = useRef(0);
  const logsContentRef = useRef("");

  const isEditable = status === "draft" || status === "failed" || status === "canceled";
  const isRunningLike = status === "running" || status === "queued";
  const task = (typeof draftConfig?.task === "string" ? draftConfig.task : "classification") as string;
  const runtimeBadge = useMemo(() => runtimeBadgeLabel(runtimeInfo), [runtimeInfo]);
  const primaryMetricKey = metricKeyForTask(task);
  const primaryMetricLabel = primaryMetricKey.replace("val_", "val ");
  const primaryColor = "#2f6fca";
  const lossColor = "#c96262";

  const validation = useMemo(() => configValidation(draftConfig ?? {}), [draftConfig]);
  const isDirty = useMemo(() => {
    if (!savedRecord || !draftConfig) return false;
    return savedRecord.name !== draftName || JSON.stringify(savedRecord.config_json) !== JSON.stringify(draftConfig);
  }, [draftConfig, draftName, savedRecord]);

  const checkpointIndex = useMemo(() => indexCheckpointsByKind(checkpoints), [checkpoints]);
  const surfacedRunError = useMemo(() => {
    if (typeof savedRecord?.error === "string" && savedRecord.error.trim()) return savedRecord.error.trim();
    if (typeof lastRunMessage === "string" && lastRunMessage.trim()) return lastRunMessage.trim();
    return null;
  }, [lastRunMessage, savedRecord?.error]);

  const chartKeys = useMemo(() => {
    const keys: string[] = [];
    if (showPrimary) keys.push(primaryMetricKey);
    if (showValLoss) keys.push("val_loss");
    return keys;
  }, [primaryMetricKey, showPrimary, showValLoss]);

  const chartWidth = 760;
  const chartHeight = 280;
  const chartPadding = 44;
  const chartInnerWidth = chartWidth - (chartPadding * 2);
  const chartInnerHeight = chartHeight - (chartPadding * 2);
  const chartMaxEpoch = useMemo(() => {
    const epochs = metrics
      .map((row) => (typeof row.epoch === "number" ? row.epoch : Number.parseInt(String(row.epoch), 10)))
      .filter((epoch) => Number.isFinite(epoch) && epoch >= 1);
    if (epochs.length === 0) return 1;
    return Math.max(...epochs);
  }, [metrics]);
  const primaryMetricIsBounded = useMemo(() => isBoundedMetricKey(primaryMetricKey), [primaryMetricKey]);
  const useSecondaryAxis = showPrimary && showValLoss && primaryMetricIsBounded;
  const primarySeriesValues = useMemo(
    () => metrics.map((row) => metricValueByKey(row, primaryMetricKey)).filter((value): value is number => value != null),
    [metrics, primaryMetricKey],
  );
  const lossSeriesValues = useMemo(
    () => metrics.map((row) => metricValueByKey(row, "val_loss")).filter((value): value is number => value != null),
    [metrics],
  );
  const combinedSeriesValues = useMemo(() => [...primarySeriesValues, ...lossSeriesValues], [lossSeriesValues, primarySeriesValues]);
  const primaryDomain = useMemo(
    () =>
      computeSeriesDomain(primarySeriesValues, {
        useLog: false,
        clamp01: primaryMetricIsBounded,
      }),
    [primaryMetricIsBounded, primarySeriesValues],
  );
  const lossDomain = useMemo(
    () =>
      computeSeriesDomain(lossSeriesValues, {
        useLog: false,
        clamp01: false,
      }),
    [lossSeriesValues],
  );
  const combinedDomain = useMemo(
    () =>
      computeSeriesDomain(combinedSeriesValues, {
        useLog: false,
        clamp01: false,
      }),
    [combinedSeriesValues],
  );
  const leftAxisDomain = useMemo(() => {
    if (useSecondaryAxis) return primaryDomain;
    if (showPrimary && !showValLoss) return primaryDomain;
    if (!showPrimary && showValLoss) return lossDomain;
    return combinedDomain;
  }, [combinedDomain, lossDomain, primaryDomain, showPrimary, showValLoss, useSecondaryAxis]);
  const rightAxisDomain = useMemo(() => (useSecondaryAxis ? lossDomain : null), [lossDomain, useSecondaryAxis]);
  const leftAxisTicks = useMemo(
    () => buildTicks(leftAxisDomain, { count: 5, clamp01: primaryMetricIsBounded && (useSecondaryAxis || (showPrimary && !showValLoss)) }),
    [leftAxisDomain, primaryMetricIsBounded, showPrimary, showValLoss, useSecondaryAxis],
  );
  const rightAxisTicks = useMemo(
    () => (rightAxisDomain ? buildTicks(rightAxisDomain, { count: 5 }) : []),
    [rightAxisDomain],
  );
  const xTickValues = useMemo(() => {
    const ticks = buildTicks({ min: 1, max: chartMaxEpoch }, { count: 5 }).map((tick) => Math.max(1, Math.round(tick)));
    return Array.from(new Set(ticks));
  }, [chartMaxEpoch]);

  const seriesLegend = useMemo(
    () => [
      {
        key: primaryMetricKey,
        label: primaryMetricLabel,
        color: primaryColor,
        enabled: showPrimary,
        axis: "left" as const,
      },
      {
        key: "val_loss",
        label: "val loss",
        color: lossColor,
        enabled: showValLoss,
        axis: useSecondaryAxis ? ("right" as const) : ("left" as const),
      },
    ],
    [primaryMetricKey, primaryMetricLabel, showPrimary, showValLoss, useSecondaryAxis],
  );

  const hoveredMetric = useMemo(() => {
    if (hoveredEpoch == null) return null;
    let nearest: ExperimentMetricPoint | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const row of metrics) {
      const epoch = asEpoch(row.epoch);
      if (epoch == null) continue;
      const distance = Math.abs(epoch - hoveredEpoch);
      if (distance < bestDistance) {
        bestDistance = distance;
        nearest = row;
      }
    }
    return nearest;
  }, [hoveredEpoch, metrics]);

  const hoveredEpochValue = hoveredMetric ? asEpoch(hoveredMetric.epoch) : null;
  const hoveredX =
    hoveredEpochValue == null
      ? null
      : chartPadding + (((hoveredEpochValue - 1) / Math.max(1, chartMaxEpoch - 1)) * chartInnerWidth);
  const hoveredSeriesValues = useMemo(() => {
    if (!hoveredMetric) return [];
    return seriesLegend
      .filter((series) => series.enabled)
      .map((series) => ({
        key: series.key,
        label: series.label,
        color: series.color,
        value: metricValueByKey(hoveredMetric, series.key),
      }))
      .filter((row) => row.value != null) as Array<{ key: string; label: string; color: string; value: number }>;
  }, [hoveredMetric, seriesLegend]);

  const hoveredPlotRows = useMemo(() => {
    return hoveredSeriesValues.map((row) => {
      const domain = useSecondaryAxis && row.key === "val_loss" ? lossDomain : leftAxisDomain;
      const range = Math.max(1e-9, domain.max - domain.min);
      const y = chartPadding + (((domain.max - row.value) / range) * chartInnerHeight);
      return { ...row, y };
    });
  }, [chartInnerHeight, chartPadding, hoveredSeriesValues, leftAxisDomain, lossDomain, useSecondaryAxis]);

  const hoverTooltip = useMemo(() => {
    if (hoveredX == null || hoveredEpochValue == null || hoveredPlotRows.length === 0) return null;
    const tooltipWidth = 196;
    const tooltipLineHeight = 15;
    const tooltipHeight = 28 + (hoveredPlotRows.length * tooltipLineHeight);
    let x = hoveredX + 10;
    if (x + tooltipWidth > chartWidth - 6) {
      x = hoveredX - tooltipWidth - 10;
    }
    let y = chartPadding + 10;
    if (y + tooltipHeight > chartHeight - 8) {
      y = chartHeight - tooltipHeight - 8;
    }
    return { x, y, width: tooltipWidth, height: tooltipHeight };
  }, [chartHeight, chartPadding, chartWidth, hoveredEpochValue, hoveredPlotRows.length, hoveredX]);

  const primaryLinePoints = useMemo(
    () =>
      showPrimary
        ? buildLinePoints(metrics, primaryMetricKey, {
            width: chartWidth,
            height: chartHeight,
            padding: chartPadding,
            seriesKeys: chartKeys,
            domain: leftAxisDomain,
            useLog: false,
          })
        : "",
    [chartHeight, chartKeys, chartPadding, chartWidth, leftAxisDomain, metrics, primaryMetricKey, showPrimary],
  );
  const valLossLinePoints = useMemo(
    () =>
      showValLoss
        ? buildLinePoints(metrics, "val_loss", {
            width: chartWidth,
            height: chartHeight,
            padding: chartPadding,
            seriesKeys: chartKeys,
            domain: useSecondaryAxis ? lossDomain : leftAxisDomain,
            useLog: false,
          })
        : "",
    [chartHeight, chartKeys, chartPadding, chartWidth, leftAxisDomain, lossDomain, metrics, showValLoss, useSecondaryAxis],
  );

  function handleChartMouseMove(event: React.MouseEvent<SVGSVGElement>) {
    if (metrics.length === 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width <= 0) return;
    const relativeX = event.clientX - rect.left;
    const svgX = (relativeX / rect.width) * chartWidth;
    const clampedX = Math.max(chartPadding, Math.min(chartPadding + chartInnerWidth, svgX));
    const approximateEpoch = 1 + (((clampedX - chartPadding) / Math.max(1, chartInnerWidth)) * Math.max(1, chartMaxEpoch - 1));

    let nearestEpoch: number | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const row of metrics) {
      const epoch = asEpoch(row.epoch);
      if (epoch == null) continue;
      const distance = Math.abs(epoch - approximateEpoch);
      if (distance < bestDistance) {
        bestDistance = distance;
        nearestEpoch = epoch;
      }
    }
    setHoveredEpoch(nearestEpoch);
  }

  const modelId = savedRecord?.model_id ?? "";
  const backToExperimentsHref = `/projects/${encodeURIComponent(projectId)}/experiments`;

  const loadDetail = useCallback(async () => {
    setIsLoading(true);
    setErrorMessage(null);
    setSaveError(null);
    try {
      const [record, models] = await Promise.all([getExperiment(projectId, experimentId), listProjectModels(projectId)]);
      const resolvedModelName = models.find((model) => model.id === record.model_id)?.name ?? record.model_id;
      setSavedRecord(record);
      setDraftName(record.name);
      setDraftConfig(cloneConfig(record.config_json));
      setMetrics(record.metrics ?? []);
      setCheckpoints(record.checkpoints ?? []);
      setStatus(record.status);
      setLastRunMessage(typeof record.error === "string" && record.error.trim() ? record.error.trim() : null);
      const attempt = typeof record.current_run_attempt === "number" ? record.current_run_attempt : null;
      setActiveAttempt(attempt);
      eventCursorRef.current = 0;
      setModelName(resolvedModelName);
    } catch (error) {
      setErrorMessage(parseApiErrorMessage(error, "Failed to load experiment"));
    } finally {
      setIsLoading(false);
    }
  }, [experimentId, projectId]);

  const loadRuntime = useCallback(async () => {
    try {
      const payload = await getExperimentRuntime(projectId, experimentId);
      setRuntimeInfo(payload);
      setRuntimeError(null);
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        setRuntimeInfo(null);
        setRuntimeError(null);
      } else {
        setRuntimeInfo(null);
        setRuntimeError(parseApiErrorMessage(error, "Failed to load runtime info"));
      }
    }
  }, [experimentId, projectId]);

  const fetchLogsChunk = useCallback(
    async (reset = false) => {
      setIsLogsLoading(true);
      try {
        const chunk = await getExperimentLogs(projectId, experimentId, {
          fromByte: reset ? 0 : logsCursorRef.current,
          maxBytes: 65536,
        });
        setLogsError(null);
        const merged = mergeLogChunk(reset ? "" : logsContentRef.current, chunk, { maxBytes: 200 * 1024, maxLines: 5000 });
        logsContentRef.current = merged.content;
        logsCursorRef.current = merged.cursor;
        setLogsContent(merged.content);
        setLogsCursor(merged.cursor);
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) {
          if (reset) {
            logsContentRef.current = "";
            logsCursorRef.current = 0;
            setLogsCursor(0);
            setLogsContent("");
          }
          setLogsError(null);
        } else {
          setLogsError(parseApiErrorMessage(error, "Failed to load training logs"));
        }
      } finally {
        setIsLogsLoading(false);
      }
    },
    [experimentId, projectId],
  );

  useEffect(() => {
    void loadDetail();
  }, [loadDetail]);

  useEffect(() => {
    void loadRuntime();
  }, [loadRuntime, status]);

  useEffect(() => {
    logsCursorRef.current = 0;
    logsContentRef.current = "";
    setLogsCursor(0);
    setLogsContent("");
    setLogsError(null);
    void fetchLogsChunk(true);
  }, [experimentId, fetchLogsChunk]);

  useEffect(() => {
    if (!logsAutoRefresh || !isRunningLike) return;
    const timer = window.setInterval(() => {
      void fetchLogsChunk(false);
    }, 2000);
    return () => window.clearInterval(timer);
  }, [fetchLogsChunk, isRunningLike, logsAutoRefresh]);

  useEffect(() => {
    setHasUnsavedDrafts(isEditable && isDirty);
  }, [isDirty, isEditable, setHasUnsavedDrafts]);

  useEffect(() => () => setHasUnsavedDrafts(false), [setHasUnsavedDrafts]);

  useEffect(() => {
    if (!toastMessage) return;
    const timeout = window.setTimeout(() => setToastMessage(null), 5000);
    return () => window.clearTimeout(timeout);
  }, [toastMessage]);

  useEffect(() => {
    if (!isRunningLike) return;
    const stop = streamExperimentEvents(
      projectId,
      experimentId,
      {
        fromLine: eventCursorRef.current,
        attempt: activeAttempt ?? undefined,
      },
      {
        onEnvelope: (payload) => {
          if (typeof payload.line === "number" && payload.line > eventCursorRef.current) {
            eventCursorRef.current = payload.line;
          }
          if (typeof payload.attempt === "number") {
            setActiveAttempt(payload.attempt);
          }
        },
        onEvent: (event) => {
        if (event.type === "status") {
          if (event.status) setStatus(event.status);
          if (typeof event.attempt === "number") setActiveAttempt(event.attempt);
          return;
        }
        if (event.type === "metric") {
          setMetrics((current) => mergeMetricPoints(current as any[], [event as any]) as ExperimentMetricPoint[]);
          if (typeof event.attempt === "number") setActiveAttempt(event.attempt);
          return;
        }
        if (event.type === "checkpoint") {
          setCheckpoints((current) => {
            const next = [...current];
            const index = next.findIndex((row) => row.kind === event.kind);
            const row = event as ExperimentCheckpoint;
            if (index >= 0) next[index] = row;
            else next.push(row);
            return next;
          });
          if (typeof event.attempt === "number") setActiveAttempt(event.attempt);
          return;
        }
        if (event.type === "done") {
          if (event.status) setStatus(event.status);
          if (typeof event.attempt === "number") setActiveAttempt(event.attempt);
          if (event.status === "failed") {
            const reason = typeof event.message === "string" && event.message.trim()
              ? event.message.trim()
              : "Unknown trainer error";
            setLastRunMessage(reason);
            setToastTone("error");
            setToastMessage(`Training failed: ${reason}`);
          } else if (event.status === "completed") {
            setLastRunMessage(null);
            setToastTone("success");
            setToastMessage("Training completed");
          }
          void loadDetail();
        }
      },
    });
    return () => stop();
  }, [activeAttempt, experimentId, isRunningLike, loadDetail, projectId]);

  function patchConfig(mutator: (next: Record<string, unknown>) => void) {
    setDraftConfig((current) => {
      if (!current) return current;
      const next = cloneConfig(current);
      mutator(next);
      return next;
    });
  }

  async function handleRefreshLogs() {
    await fetchLogsChunk(false);
  }

  async function handleSave() {
    if (!draftConfig || !savedRecord) return;
    setIsSaving(true);
    setSaveError(null);
    try {
      const updated = await updateExperiment(projectId, experimentId, {
        name: draftName,
        config_json: draftConfig,
      });
      setSavedRecord(updated);
      setDraftName(updated.name);
      setDraftConfig(cloneConfig(updated.config_json));
      setStatus(updated.status);
      setMetrics(updated.metrics ?? []);
      setCheckpoints(updated.checkpoints ?? []);
      setToastTone("success");
      setToastMessage("Experiment saved");
    } catch (error) {
      const message = parseApiErrorMessage(error, "Failed to save experiment");
      setSaveError(message);
      setToastTone("error");
      setToastMessage(`Save failed: ${message}`);
    } finally {
      setIsSaving(false);
    }
  }

  async function handleStart() {
    setIsStarting(true);
    try {
      setLastRunMessage(null);
      const started = await startExperiment(projectId, experimentId);
      if (started.status) setStatus(started.status);
      if (typeof started.attempt === "number") setActiveAttempt(started.attempt);
      eventCursorRef.current = 0;
      setToastTone("success");
      setToastMessage("Training started");
      void loadDetail();
    } catch (error) {
      const message = parseApiErrorMessage(error, "Failed to start experiment");
      setToastTone("error");
      setToastMessage(message);
    } finally {
      setIsStarting(false);
    }
  }

  async function handleCancel() {
    setIsCanceling(true);
    try {
      const canceled = await cancelExperiment(projectId, experimentId);
      if (canceled.status) setStatus(canceled.status);
      if (typeof canceled.attempt === "number") setActiveAttempt(canceled.attempt);
      setToastTone("success");
      setToastMessage(canceled.status === "running" ? "Cancel requested" : "Training canceled");
      void loadDetail();
    } catch (error) {
      const message = parseApiErrorMessage(error, "Failed to cancel experiment");
      setToastTone("error");
      setToastMessage(message);
    } finally {
      setIsCanceling(false);
    }
  }

  async function handlePickCheckpoint(kind: "best_loss" | "best_metric" | "latest") {
    try {
      const updated = await updateExperiment(projectId, experimentId, { selected_checkpoint_kind: kind });
      setSavedRecord(updated);
      setToastTone("success");
      setToastMessage(`Selected ${kind} checkpoint`);
    } catch (error) {
      const message = parseApiErrorMessage(error, "Failed to select checkpoint");
      setToastTone("error");
      setToastMessage(message);
    }
  }

  const optimizer = asRecord(draftConfig?.optimizer);
  const advanced = asRecord(draftConfig?.advanced);
  const datasetVersionId = typeof draftConfig?.dataset_version_id === "string" ? draftConfig.dataset_version_id : "-";
  const optimizerType = typeof optimizer.type === "string" ? optimizer.type : "adam";
  const learningRate = typeof optimizer.lr === "number" ? String(optimizer.lr) : "";
  const epochs = typeof draftConfig?.epochs === "number" ? String(draftConfig.epochs) : "";
  const batchSize = typeof draftConfig?.batch_size === "number" ? String(draftConfig.batch_size) : "";
  const augmentationProfile = typeof draftConfig?.augmentation_profile === "string" ? draftConfig.augmentation_profile : "light";
  const precision = typeof draftConfig?.precision === "string" ? draftConfig.precision : "fp32";
  const seed = typeof advanced.seed === "number" ? String(advanced.seed) : "1337";
  const numWorkers = typeof advanced.num_workers === "number" ? String(advanced.num_workers) : "0";
  const isClassificationTask = task === "classification";
  const classNames = useMemo(
    () => (Array.isArray(evaluation?.classes?.class_names) ? evaluation?.classes?.class_names : []),
    [evaluation?.classes?.class_names],
  );
  const confusionRawMatrix = useMemo(
    () => (Array.isArray(evaluation?.confusion_matrix?.matrix) ? evaluation.confusion_matrix.matrix : []),
    [evaluation?.confusion_matrix?.matrix],
  );
  const normalizedConfusion = useMemo(
    () => normalizeConfusion(confusionRawMatrix, confusionNormalize),
    [confusionNormalize, confusionRawMatrix],
  );
  const confusionMax = useMemo(() => {
    const values = normalizedConfusion.flatMap((row) => (Array.isArray(row) ? row : []));
    if (values.length === 0) return 0;
    return Math.max(...values);
  }, [normalizedConfusion]);

  const parsedTrueFilter = predictionTrueClass === "all" ? undefined : Number.parseInt(predictionTrueClass, 10);
  const parsedPredFilter = predictionPredClass === "all" ? undefined : Number.parseInt(predictionPredClass, 10);
  const explorerRows = useMemo(() => {
    const samples = evaluation?.samples ?? {};
    const bucket = Array.isArray(samples[predictionMode]) ? samples[predictionMode] : [];
    return filterPredictionRows(bucket, {
      mode: predictionMode,
      trueClassIndex: Number.isFinite(parsedTrueFilter) ? parsedTrueFilter : undefined,
      predClassIndex: Number.isFinite(parsedPredFilter) ? parsedPredFilter : undefined,
      limit: predictionLimit,
    }) as ExperimentEvaluationSampleRow[];
  }, [evaluation?.samples, parsedPredFilter, parsedTrueFilter, predictionLimit, predictionMode]);

  const sortedPerClassRows = useMemo(() => {
    const rows = Array.isArray(evaluation?.per_class) ? [...evaluation.per_class] : [];
    if (perClassSort === "f1_asc") rows.sort((a, b) => a.f1 - b.f1);
    if (perClassSort === "f1_desc") rows.sort((a, b) => b.f1 - a.f1);
    if (perClassSort === "precision_desc") rows.sort((a, b) => b.precision - a.precision);
    if (perClassSort === "recall_desc") rows.sort((a, b) => b.recall - a.recall);
    if (perClassSort === "support_desc") rows.sort((a, b) => b.support - a.support);
    return rows;
  }, [evaluation?.per_class, perClassSort]);

  const dashboardSeries = useMemo(() => {
    if (dashboardChartTab === "loss") {
      return [
        { key: "train_loss", label: "train loss", color: "#cc6f36" },
        { key: "val_loss", label: "val loss", color: "#c96262" },
      ];
    }
    if (dashboardChartTab === "accuracy") {
      return [{ key: "val_accuracy", label: "val accuracy", color: "#2f6fca" }];
    }
    return [
      { key: "val_macro_f1", label: "val macro f1", color: "#2f6fca" },
      { key: "val_macro_precision", label: "val macro precision", color: "#2f9d58" },
      { key: "val_macro_recall", label: "val macro recall", color: "#cc6f36" },
    ];
  }, [dashboardChartTab]);

  const dashboardSeriesKeys = useMemo(() => dashboardSeries.map((series) => series.key), [dashboardSeries]);
  const dashboardBounded = dashboardChartTab !== "loss";
  const dashboardHasData = useMemo(
    () =>
      metrics.some((row) =>
        dashboardSeriesKeys.some((key) => {
          const value = metricValueByKey(row, key);
          return value != null;
        }),
      ),
    [dashboardSeriesKeys, metrics],
  );
  const dashboardValues = useMemo(
    () =>
      metrics.flatMap((row) =>
        dashboardSeriesKeys
          .map((key) => metricValueByKey(row, key))
          .filter((value): value is number => value != null),
      ),
    [dashboardSeriesKeys, metrics],
  );
  const dashboardDomain = useMemo(
    () =>
      computeSeriesDomain(dashboardValues, {
        useLog: dashboardLogScale,
        clamp01: dashboardBounded && !dashboardLogScale,
      }),
    [dashboardBounded, dashboardLogScale, dashboardValues],
  );
  const dashboardYTicks = useMemo(
    () =>
      buildTicks(dashboardDomain, {
        useLog: dashboardLogScale,
        count: 5,
        clamp01: dashboardBounded && !dashboardLogScale,
      }),
    [dashboardBounded, dashboardDomain, dashboardLogScale],
  );
  const dashboardXTicks = useMemo(
    () => Array.from(new Set(buildTicks({ min: 1, max: chartMaxEpoch }, { count: 5 }).map((tick) => Math.max(1, Math.round(tick))))),
    [chartMaxEpoch],
  );
  const dashboardLinePoints = useMemo(
    () =>
      dashboardSeries.map((series) => ({
        ...series,
        points: buildLinePoints(metrics, series.key, {
          width: chartWidth,
          height: chartHeight,
          padding: chartPadding,
          seriesKeys: dashboardSeriesKeys,
          domain: dashboardDomain,
          useLog: dashboardLogScale,
        }),
      })),
    [chartHeight, chartPadding, chartWidth, dashboardDomain, dashboardLogScale, dashboardSeries, dashboardSeriesKeys, metrics],
  );

  useEffect(() => {
    if (!isClassificationTask) {
      setEvaluation(null);
      setEvaluationError(null);
      setIsEvaluationLoading(false);
      return;
    }
    let isMounted = true;
    async function loadEvaluation() {
      setIsEvaluationLoading(true);
      setEvaluationError(null);
      try {
        const payload = await getExperimentEvaluation(projectId, experimentId);
        if (!isMounted) return;
        setEvaluation(payload);
      } catch (error) {
        if (!isMounted) return;
        if (error instanceof ApiError && error.status === 404) {
          setEvaluation(null);
          setEvaluationError("Evaluation not available yet. Run must complete at least one validation epoch.");
        } else {
          setEvaluation(null);
          setEvaluationError(parseApiErrorMessage(error, "Failed to load evaluation"));
        }
      } finally {
        if (isMounted) setIsEvaluationLoading(false);
      }
    }
    void loadEvaluation();
    return () => {
      isMounted = false;
    };
  }, [experimentId, isClassificationTask, projectId, status]);

  useEffect(() => {
    if (!cellDrawer || !isClassificationTask) {
      setCellSamples([]);
      setCellSamplesMessage(null);
      return;
    }
    let isMounted = true;
    async function loadCellSamples() {
      const mode = cellDrawer.trueClassIndex === cellDrawer.predClassIndex ? "lowest_confidence_correct" : "misclassified";
      try {
        const response = await listExperimentSamples(projectId, experimentId, {
          mode,
          trueClassIndex: cellDrawer.trueClassIndex,
          predClassIndex: cellDrawer.predClassIndex,
          limit: 100,
        });
        if (!isMounted) return;
        setCellSamples(response.items ?? []);
        setCellSamplesMessage(response.message ?? null);
      } catch {
        if (!isMounted) return;
        const fallbackSource = mode === "misclassified" ? evaluation?.samples?.misclassified : evaluation?.samples?.lowest_confidence_correct;
        const fallbackRows = filterPredictionRows(fallbackSource ?? [], {
          mode,
          trueClassIndex: cellDrawer.trueClassIndex,
          predClassIndex: cellDrawer.predClassIndex,
          limit: 100,
        }) as ExperimentEvaluationSampleRow[];
        setCellSamples(fallbackRows);
        setCellSamplesMessage(fallbackRows.length < 1 ? "No matching samples available for this confusion cell." : null);
      }
    }
    void loadCellSamples();
    return () => {
      isMounted = false;
    };
  }, [cellDrawer, evaluation?.samples?.misclassified, experimentId, isClassificationTask, projectId]);

  return (
    <>
      <main className="workspace-shell project-page-shell">
        <section className="workspace-frame project-content-frame">
          <header className="project-section-header">
            <div className="experiment-header-title">
              <h2>Train Experiment</h2>
              <p>
                Model: <strong>{modelName ?? modelId}</strong>
              </p>
              {activeAttempt ? <p>Run #{activeAttempt}</p> : null}
            </div>
            <Link href={backToExperimentsHref} className="ghost-button">
              Back to Experiments
            </Link>
          </header>

          {isLoading ? (
            <div className="placeholder-card">
              <p>Loading experiment...</p>
            </div>
          ) : null}
          {errorMessage ? <p className="project-field-error">{errorMessage}</p> : null}

          {!isLoading && draftConfig ? (
            <div className="experiment-layout">
              <section className="experiment-left-panel">
                <div className="experiment-card">
                  <label className="project-field">
                    <span>Experiment Name</span>
                    <input
                      type="text"
                      value={draftName}
                      onChange={(event) => setDraftName(event.target.value)}
                      disabled={!isEditable}
                    />
                  </label>
                  <div className="experiment-status-row">
                    <span className={`status-pill status-${status}`}>{status}</span>
                    {runtimeBadge ? <span className="runtime-pill">{runtimeBadge}</span> : null}
                    <span>Updated: {formatDateTime(savedRecord?.updated_at)}</span>
                  </div>
                  {surfacedRunError ? <p className="project-field-error">Last run error: {surfacedRunError}</p> : null}
                </div>

                <div className="experiment-card">
                  <h3>Training Details</h3>
                  <label className="project-field">
                    <span>Training Dataset</span>
                    <input type="text" value={datasetVersionId} readOnly />
                  </label>
                  <label className="project-field">
                    <span>Optimizer</span>
                    <select
                      value={optimizerType}
                      disabled={!isEditable}
                      onChange={(event) =>
                        patchConfig((next) => {
                          const opt = asRecord(next.optimizer);
                          opt.type = event.target.value;
                          next.optimizer = opt;
                        })
                      }
                    >
                      <option value="adam">adam</option>
                      <option value="adamw">adamw</option>
                      <option value="sgd">sgd</option>
                    </select>
                  </label>
                  <label className="project-field">
                    <span>Learning Rate</span>
                    <input
                      type="number"
                      step="0.0001"
                      min="0.0000001"
                      value={learningRate}
                      disabled={!isEditable}
                      onChange={(event) =>
                        patchConfig((next) => {
                          const opt = asRecord(next.optimizer);
                          const parsed = patchNumber(event.target.value);
                          if (parsed !== null) opt.lr = parsed;
                          next.optimizer = opt;
                        })
                      }
                    />
                  </label>
                  <label className="project-field">
                    <span>Epochs</span>
                    <input
                      type="number"
                      min="1"
                      step="1"
                      value={epochs}
                      disabled={!isEditable}
                      onChange={(event) =>
                        patchConfig((next) => {
                          const parsed = Number.parseInt(event.target.value, 10);
                          if (Number.isFinite(parsed)) next.epochs = parsed;
                        })
                      }
                    />
                  </label>
                  <label className="project-field">
                    <span>Batch Size</span>
                    <input
                      type="number"
                      min="1"
                      step="1"
                      value={batchSize}
                      disabled={!isEditable}
                      onChange={(event) =>
                        patchConfig((next) => {
                          const parsed = Number.parseInt(event.target.value, 10);
                          if (Number.isFinite(parsed)) next.batch_size = parsed;
                        })
                      }
                    />
                  </label>
                  <label className="project-field">
                    <span>Augmentation</span>
                    <select
                      value={augmentationProfile}
                      disabled={!isEditable}
                      onChange={(event) =>
                        patchConfig((next) => {
                          next.augmentation_profile = event.target.value;
                        })
                      }
                    >
                      <option value="none">none</option>
                      <option value="light">light</option>
                      <option value="medium">medium</option>
                      <option value="heavy">heavy</option>
                    </select>
                  </label>

                  <button type="button" className="ghost-button experiment-advanced-toggle" onClick={() => setShowAdvanced((v) => !v)}>
                    {showAdvanced ? "Hide Advanced Parameters" : "Advanced Parameters"}
                  </button>
                  {showAdvanced ? (
                    <div className="experiment-advanced-fields">
                      <label className="project-field">
                        <span>Seed</span>
                        <input
                          type="number"
                          step="1"
                          value={seed}
                          disabled={!isEditable}
                          onChange={(event) =>
                            patchConfig((next) => {
                              const adv = asRecord(next.advanced);
                              const parsed = Number.parseInt(event.target.value, 10);
                              if (Number.isFinite(parsed)) adv.seed = parsed;
                              next.advanced = adv;
                            })
                          }
                        />
                      </label>
                      <label className="project-field">
                        <span>Num Workers</span>
                        <input
                          type="number"
                          min="0"
                          step="1"
                          value={numWorkers}
                          disabled={!isEditable}
                          onChange={(event) =>
                            patchConfig((next) => {
                              const adv = asRecord(next.advanced);
                              const parsed = Number.parseInt(event.target.value, 10);
                              if (Number.isFinite(parsed)) adv.num_workers = parsed;
                              next.advanced = adv;
                            })
                          }
                        />
                      </label>
                      <label className="project-field">
                        <span>Precision</span>
                        <select
                          value={precision}
                          disabled={!isEditable}
                          onChange={(event) =>
                            patchConfig((next) => {
                              next.precision = event.target.value;
                            })
                          }
                        >
                          <option value="fp32">fp32</option>
                          <option value="amp">amp</option>
                        </select>
                      </label>
                    </div>
                  ) : null}
                </div>
              </section>

              <section className="experiment-right-panel">
                <div className="experiment-card">
                  <h3>Runtime & Logs</h3>
                  <div className="experiment-runtime-grid">
                    <span>Device selected</span>
                    <strong>{runtimeInfo ? (runtimeBadge ?? runtimeInfo.device_selected.toUpperCase()) : "-"}</strong>
                    <span>CUDA available</span>
                    <strong>{asYesNo(runtimeInfo?.cuda_available)}</strong>
                    <span>MPS available</span>
                    <strong>{asYesNo(runtimeInfo?.mps_available)}</strong>
                    <span>AMP enabled</span>
                    <strong>{asYesNo(runtimeInfo?.amp_enabled)}</strong>
                    <span>torch</span>
                    <strong>{runtimeInfo?.torch_version ?? "-"}</strong>
                    <span>torchvision</span>
                    <strong>{runtimeInfo?.torchvision_version ?? "-"}</strong>
                    <span>num_workers</span>
                    <strong>{runtimeInfo?.num_workers ?? "-"}</strong>
                    <span>pin_memory</span>
                    <strong>{asYesNo(runtimeInfo?.pin_memory)}</strong>
                  </div>
                  {runtimeError ? <p className="project-field-error">{runtimeError}</p> : null}
                  <div className="experiment-logs-toolbar">
                    <button type="button" className="ghost-button" onClick={() => void handleRefreshLogs()} disabled={isLogsLoading}>
                      {isLogsLoading ? "Refreshing..." : "Refresh logs"}
                    </button>
                    <label className="model-builder-checkbox">
                      <input
                        type="checkbox"
                        checked={logsAutoRefresh}
                        onChange={(event) => setLogsAutoRefresh(event.target.checked)}
                        disabled={!isRunningLike}
                      />
                      <span>Auto-refresh (2s)</span>
                    </label>
                    <button type="button" className="ghost-button" onClick={() => setIsLogsExpanded((value) => !value)}>
                      {isLogsExpanded ? "Collapse logs" : "Expand logs"}
                    </button>
                  </div>
                  {logsError ? <p className="project-field-error">{logsError}</p> : null}
                  {isLogsExpanded ? (
                    <pre className="experiment-log-viewer">{logsContent || "No training logs available yet."}</pre>
                  ) : null}
                  <p className="experiment-log-cursor">Cursor: {logsCursor}</p>
                </div>

                <div className="experiment-card">
                  <h3>Checkpoints</h3>
                  <div className="experiment-checkpoint-grid">
                    <div className="experiment-checkpoint-row">
                      <strong>best_metric</strong>
                      <span>{formatCheckpoint(checkpointIndex.best_metric)}</span>
                      <button
                        type="button"
                        className="ghost-button"
                        disabled={checkpointIndex.best_metric?.epoch == null}
                        onClick={() => void handlePickCheckpoint("best_metric")}
                      >
                        Pick
                      </button>
                    </div>
                    <div className="experiment-checkpoint-row">
                      <strong>best_loss</strong>
                      <span>{formatCheckpoint(checkpointIndex.best_loss)}</span>
                      <button
                        type="button"
                        className="ghost-button"
                        disabled={checkpointIndex.best_loss?.epoch == null}
                        onClick={() => void handlePickCheckpoint("best_loss")}
                      >
                        Pick
                      </button>
                    </div>
                    <div className="experiment-checkpoint-row">
                      <strong>latest</strong>
                      <span>{formatCheckpoint(checkpointIndex.latest)}</span>
                      <button
                        type="button"
                        className="ghost-button"
                        disabled={checkpointIndex.latest?.epoch == null}
                        onClick={() => void handlePickCheckpoint("latest")}
                      >
                        Pick
                      </button>
                    </div>
                  </div>
                </div>

                <div className="experiment-card">
                  <h3>Metrics</h3>
                  <div className="experiment-series-toggle-row">
                    <label className="model-builder-checkbox">
                      <input type="checkbox" checked={showPrimary} onChange={(event) => setShowPrimary(event.target.checked)} />
                      <span>{primaryMetricLabel}</span>
                    </label>
                    <label className="model-builder-checkbox">
                      <input type="checkbox" checked={showValLoss} onChange={(event) => setShowValLoss(event.target.checked)} />
                      <span>val loss</span>
                    </label>
                  </div>
                  <div className="experiment-chart-legend" aria-label="Metric legend">
                    {seriesLegend.map((series) => (
                      <span key={series.key} className={`experiment-legend-item${series.enabled ? "" : " is-muted"}`}>
                        <span className="experiment-legend-swatch" style={{ background: series.color }} aria-hidden />
                        <span>{series.label}{series.enabled && useSecondaryAxis ? ` (${series.axis === "right" ? "R" : "L"})` : ""}</span>
                      </span>
                    ))}
                  </div>
                  <div className="experiment-chart-wrap">
                    {metrics.length === 0 ? (
                      <p className="labels-empty">No metrics yet. Start training to stream live values.</p>
                    ) : (
                      <svg
                        viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                        role="img"
                        aria-label="Experiment metrics chart"
                        onMouseMove={handleChartMouseMove}
                        onMouseLeave={() => setHoveredEpoch(null)}
                      >
                        <rect x="0" y="0" width={chartWidth} height={chartHeight} fill="#f8fbff" stroke="#d0d9e7" />
                        <line
                          x1={chartPadding}
                          y1={chartPadding}
                          x2={chartPadding}
                          y2={chartPadding + chartInnerHeight}
                          stroke="#9db0ca"
                          strokeWidth="1.25"
                        />
                        <line
                          x1={chartPadding}
                          y1={chartPadding + chartInnerHeight}
                          x2={chartPadding + chartInnerWidth}
                          y2={chartPadding + chartInnerHeight}
                          stroke="#9db0ca"
                          strokeWidth="1.25"
                        />
                        {leftAxisTicks.map((tickValue) => {
                          const range = Math.max(1e-9, leftAxisDomain.max - leftAxisDomain.min);
                          const ratio = (leftAxisDomain.max - tickValue) / range;
                          const y = chartPadding + (ratio * chartInnerHeight);
                          return (
                            <g key={`y:${tickValue.toFixed(6)}`}>
                              <line x1={chartPadding} y1={y} x2={chartPadding + chartInnerWidth} y2={y} stroke="#e1e8f2" strokeWidth="1" />
                              <text className="axis-tick" x={chartPadding - 6} y={y + 4} textAnchor="end">
                                {formatTick(tickValue, {
                                  bounded: primaryMetricIsBounded && (useSecondaryAxis || (showPrimary && !showValLoss)),
                                })}
                              </text>
                            </g>
                          );
                        })}
                        {useSecondaryAxis ? (
                          <>
                            <line
                              x1={chartPadding + chartInnerWidth}
                              y1={chartPadding}
                              x2={chartPadding + chartInnerWidth}
                              y2={chartPadding + chartInnerHeight}
                              stroke="#9db0ca"
                              strokeWidth="1.25"
                            />
                            {rightAxisTicks.map((tickValue) => {
                              const range = Math.max(1e-9, (rightAxisDomain?.max ?? 1) - (rightAxisDomain?.min ?? 0));
                              const ratio = ((rightAxisDomain?.max ?? 1) - tickValue) / range;
                              const y = chartPadding + (ratio * chartInnerHeight);
                              return (
                                <g key={`y-right:${tickValue.toFixed(6)}`}>
                                  <text className="axis-tick axis-tick-right" x={chartPadding + chartInnerWidth + 6} y={y + 4}>
                                    {formatTick(tickValue)}
                                  </text>
                                </g>
                              );
                            })}
                          </>
                        ) : null}
                        {xTickValues.map((tickEpoch) => {
                          const ratio = (tickEpoch - 1) / Math.max(1, chartMaxEpoch - 1);
                          const x = chartPadding + (ratio * chartInnerWidth);
                          return (
                            <g key={`x:${tickEpoch}`}>
                              <line x1={x} y1={chartPadding} x2={x} y2={chartPadding + chartInnerHeight} stroke="#edf1f7" strokeWidth="1" />
                              <text className="axis-tick" x={x} y={chartPadding + chartInnerHeight + 16} textAnchor="middle">
                                {tickEpoch}
                              </text>
                            </g>
                          );
                        })}
                        {showPrimary && primaryLinePoints ? (
                          <polyline fill="none" stroke={primaryColor} strokeWidth="2.25" points={primaryLinePoints} />
                        ) : null}
                        {showValLoss && valLossLinePoints ? (
                          <polyline fill="none" stroke={lossColor} strokeWidth="2.25" points={valLossLinePoints} />
                        ) : null}
                        {hoveredX != null ? (
                          <line
                            x1={hoveredX}
                            y1={chartPadding}
                            x2={hoveredX}
                            y2={chartPadding + chartInnerHeight}
                            stroke="#8da2c1"
                            strokeWidth="1"
                            strokeDasharray="4 3"
                          />
                        ) : null}
                        {hoveredX != null
                          ? hoveredPlotRows.map((row) => (
                              <circle key={row.key} cx={hoveredX} cy={row.y} r="3.5" fill="#ffffff" stroke={row.color} strokeWidth="2" />
                            ))
                          : null}
                        {hoverTooltip && hoveredEpochValue != null ? (
                          <g className="experiment-chart-tooltip">
                            <rect
                              x={hoverTooltip.x}
                              y={hoverTooltip.y}
                              width={hoverTooltip.width}
                              height={hoverTooltip.height}
                              rx="8"
                              ry="8"
                              fill="#f6f9ff"
                              stroke="#bdcbe0"
                            />
                            <text x={hoverTooltip.x + 10} y={hoverTooltip.y + 16} fontSize="12" fill="#304765" fontWeight="700">
                              Epoch {hoveredEpochValue}
                            </text>
                            {hoveredPlotRows.map((row, index) => (
                              <text
                                key={`${row.key}:tooltip`}
                                x={hoverTooltip.x + 10}
                                y={hoverTooltip.y + 32 + (index * 15)}
                                fontSize="12"
                                fill={row.color}
                              >
                                {row.label}: {row.value.toFixed(4)}
                              </text>
                            ))}
                          </g>
                        ) : null}
                        <text className="axis-label" x={chartPadding + (chartInnerWidth / 2)} y={chartHeight - 4} textAnchor="middle">
                          epoch
                        </text>
                        <text
                          className="axis-label"
                          x="16"
                          y={chartPadding + (chartInnerHeight / 2)}
                          transform={`rotate(-90 16 ${chartPadding + (chartInnerHeight / 2)})`}
                          textAnchor="middle"
                        >
                          {useSecondaryAxis ? primaryMetricLabel : showPrimary && !showValLoss ? primaryMetricLabel : "metric value"}
                        </text>
                        {useSecondaryAxis ? (
                          <text
                            className="axis-label"
                            x={chartWidth - 10}
                            y={chartPadding + (chartInnerHeight / 2)}
                            transform={`rotate(90 ${chartWidth - 10} ${chartPadding + (chartInnerHeight / 2)})`}
                            textAnchor="middle"
                          >
                            val loss
                          </text>
                        ) : null}
                      </svg>
                    )}
                  </div>
                </div>
              </section>
            </div>
          ) : null}

          {!isLoading ? (
            <section className="experiment-dashboard-section">
              <header className="project-section-header">
                <h3>Dashboard</h3>
                {evaluation?.attempt ? <span className="status-pill">Evaluation Run #{evaluation.attempt}</span> : null}
              </header>

              {!isClassificationTask ? (
                <div className="placeholder-card">
                  <p>Dashboard not supported yet for this task.</p>
                </div>
              ) : null}

              {isClassificationTask && isEvaluationLoading ? (
                <div className="placeholder-card">
                  <p>Loading evaluation dashboard...</p>
                </div>
              ) : null}

              {isClassificationTask && !isEvaluationLoading && evaluationError ? (
                <div className="placeholder-card">
                  <p>{evaluationError}</p>
                </div>
              ) : null}

              {isClassificationTask && !isEvaluationLoading && !evaluationError && evaluation ? (
                <>
                  <div className="experiment-card">
                    <div className="experiment-analytics-header">
                      <h4>Metrics Trends</h4>
                      <div className="experiment-analytics-controls">
                        <div className="experiment-tab-row">
                          <button
                            type="button"
                            className={`ghost-button ${dashboardChartTab === "loss" ? "active-toggle" : ""}`}
                            onClick={() => setDashboardChartTab("loss")}
                          >
                            Loss
                          </button>
                          <button
                            type="button"
                            className={`ghost-button ${dashboardChartTab === "accuracy" ? "active-toggle" : ""}`}
                            onClick={() => setDashboardChartTab("accuracy")}
                          >
                            Accuracy
                          </button>
                          <button
                            type="button"
                            className={`ghost-button ${dashboardChartTab === "prf" ? "active-toggle" : ""}`}
                            onClick={() => setDashboardChartTab("prf")}
                          >
                            F1 / Precision / Recall
                          </button>
                        </div>
                        <label className="model-builder-checkbox">
                          <input
                            type="checkbox"
                            checked={dashboardLogScale}
                            onChange={(event) => setDashboardLogScale(event.target.checked)}
                          />
                          <span>Log scale</span>
                        </label>
                      </div>
                    </div>
                    <div className="experiment-chart-legend">
                      {dashboardLinePoints.map((series) => (
                        <span key={series.key} className="experiment-legend-item">
                          <span className="experiment-legend-swatch" style={{ background: series.color }} />
                          <span>{series.label}</span>
                        </span>
                      ))}
                    </div>
                    <div className="experiment-chart-wrap">
                      {!dashboardHasData ? (
                        <p className="labels-empty">Metrics for this tab are not available yet.</p>
                      ) : (
                        <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} role="img" aria-label="Dashboard metrics chart">
                          <rect x="0" y="0" width={chartWidth} height={chartHeight} fill="#f8fbff" stroke="#d0d9e7" />
                          <line x1={chartPadding} y1={chartPadding} x2={chartPadding} y2={chartPadding + chartInnerHeight} stroke="#9db0ca" />
                          <line
                            x1={chartPadding}
                            y1={chartPadding + chartInnerHeight}
                            x2={chartPadding + chartInnerWidth}
                            y2={chartPadding + chartInnerHeight}
                            stroke="#9db0ca"
                          />
                          {dashboardYTicks.map((tickValue) => {
                            const range = Math.max(1e-9, dashboardDomain.max - dashboardDomain.min);
                            const ratio = (dashboardDomain.max - tickValue) / range;
                            const y = chartPadding + (ratio * chartInnerHeight);
                            return (
                              <g key={`dashboard-y:${tickValue.toFixed(8)}`}>
                                <line x1={chartPadding} y1={y} x2={chartPadding + chartInnerWidth} y2={y} stroke="#e1e8f2" strokeWidth="1" />
                                <text className="axis-tick" x={chartPadding - 6} y={y + 4} textAnchor="end">
                                  {formatTick(tickValue, { useLog: dashboardLogScale, bounded: dashboardBounded && !dashboardLogScale })}
                                </text>
                              </g>
                            );
                          })}
                          {dashboardXTicks.map((tickEpoch) => {
                            const ratio = (tickEpoch - 1) / Math.max(1, chartMaxEpoch - 1);
                            const x = chartPadding + (ratio * chartInnerWidth);
                            return (
                              <g key={`dashboard-x:${tickEpoch}`}>
                                <line x1={x} y1={chartPadding} x2={x} y2={chartPadding + chartInnerHeight} stroke="#edf1f7" strokeWidth="1" />
                                <text className="axis-tick" x={x} y={chartPadding + chartInnerHeight + 16} textAnchor="middle">
                                  {tickEpoch}
                                </text>
                              </g>
                            );
                          })}
                          {dashboardLinePoints.map((series) =>
                            series.points ? (
                              <polyline key={series.key} fill="none" stroke={series.color} strokeWidth="2.1" points={series.points} />
                            ) : null,
                          )}
                          <text className="axis-label" x={chartPadding + (chartInnerWidth / 2)} y={chartHeight - 4} textAnchor="middle">
                            epoch
                          </text>
                          <text
                            className="axis-label"
                            x="16"
                            y={chartPadding + (chartInnerHeight / 2)}
                            transform={`rotate(-90 16 ${chartPadding + (chartInnerHeight / 2)})`}
                            textAnchor="middle"
                          >
                            {dashboardChartTab === "loss" ? "Loss" : "Metric value"}{dashboardLogScale ? " (log10)" : ""}
                          </text>
                        </svg>
                      )}
                    </div>
                  </div>

                  <div className="experiment-card">
                    <div className="experiment-analytics-header">
                      <h4>Confusion Matrix (Validation)</h4>
                      <label className="project-field">
                        <span>Normalize</span>
                        <select
                          value={confusionNormalize}
                          onChange={(event) => setConfusionNormalize(event.target.value as "none" | "by_true" | "by_pred")}
                        >
                          <option value="none">none</option>
                          <option value="by_true">by_true</option>
                          <option value="by_pred">by_pred</option>
                        </select>
                      </label>
                    </div>
                    <div className="confusion-matrix-wrap">
                      {normalizedConfusion.length === 0 ? (
                        <p className="labels-empty">Confusion matrix unavailable.</p>
                      ) : (
                        <div className="confusion-grid">
                          {normalizedConfusion.map((row, trueIndex) => (
                            <div key={`row-${trueIndex}`} className="confusion-row">
                              <span className="confusion-axis-label" title={classNames[trueIndex] ?? `class_${trueIndex}`}>
                                {classNames[trueIndex] ?? `c${trueIndex}`}
                              </span>
                              {row.map((value, predIndex) => {
                                const rawValue = confusionRawMatrix?.[trueIndex]?.[predIndex] ?? 0;
                                const normalizedValue = Number.isFinite(value) ? value : 0;
                                const intensity = confusionMax > 0 ? Math.max(0.08, normalizedValue / confusionMax) : 0.08;
                                return (
                                  <button
                                    type="button"
                                    key={`cell-${trueIndex}-${predIndex}`}
                                    className="confusion-cell"
                                    style={{ backgroundColor: `rgba(47,111,202,${intensity})` }}
                                    title={`true=${classNames[trueIndex] ?? trueIndex}, pred=${classNames[predIndex] ?? predIndex}, count=${rawValue}, normalized=${normalizedValue.toFixed(4)}`}
                                    onClick={() => setCellDrawer({ trueClassIndex: trueIndex, predClassIndex: predIndex })}
                                  >
                                    {confusionNormalize === "none" ? rawValue : normalizedValue.toFixed(2)}
                                  </button>
                                );
                              })}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="experiment-card">
                    <div className="experiment-analytics-header">
                      <h4>Per-class Metrics</h4>
                      <label className="project-field">
                        <span>Sort</span>
                        <select
                          value={perClassSort}
                          onChange={(event) =>
                            setPerClassSort(
                              event.target.value as "f1_desc" | "f1_asc" | "precision_desc" | "recall_desc" | "support_desc",
                            )
                          }
                        >
                          <option value="f1_desc">f1 desc</option>
                          <option value="f1_asc">f1 asc</option>
                          <option value="precision_desc">precision desc</option>
                          <option value="recall_desc">recall desc</option>
                          <option value="support_desc">support desc</option>
                        </select>
                      </label>
                    </div>
                    <div className="models-table-wrap">
                      <table className="models-table">
                        <thead>
                          <tr>
                            <th>Class</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1</th>
                            <th>Support</th>
                          </tr>
                        </thead>
                        <tbody>
                          {sortedPerClassRows.map((row) => (
                            <tr key={`per-class-${row.class_index}`}>
                              <td>{row.name}</td>
                              <td>{row.precision.toFixed(4)}</td>
                              <td>{row.recall.toFixed(4)}</td>
                              <td>{row.f1.toFixed(4)}</td>
                              <td>{row.support}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div className="experiment-card">
                    <div className="experiment-analytics-header">
                      <h4>Prediction Explorer</h4>
                      <div className="experiment-analytics-controls">
                        <label className="project-field">
                          <span>Mode</span>
                          <select
                            value={predictionMode}
                            onChange={(event) =>
                              setPredictionMode(event.target.value as "misclassified" | "lowest_confidence_correct" | "highest_confidence_wrong")
                            }
                          >
                            <option value="misclassified">Misclassified</option>
                            <option value="lowest_confidence_correct">Lowest confidence correct</option>
                            <option value="highest_confidence_wrong">Highest confidence wrong</option>
                          </select>
                        </label>
                        <label className="project-field">
                          <span>True class</span>
                          <select value={predictionTrueClass} onChange={(event) => setPredictionTrueClass(event.target.value)}>
                            <option value="all">All</option>
                            {classNames.map((className, index) => (
                              <option key={`true-${index}`} value={String(index)}>
                                {className}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="project-field">
                          <span>Pred class</span>
                          <select value={predictionPredClass} onChange={(event) => setPredictionPredClass(event.target.value)}>
                            <option value="all">All</option>
                            {classNames.map((className, index) => (
                              <option key={`pred-${index}`} value={String(index)}>
                                {className}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="project-field">
                          <span>Limit</span>
                          <select value={predictionLimit} onChange={(event) => setPredictionLimit(Number.parseInt(event.target.value, 10))}>
                            <option value={25}>25</option>
                            <option value={50}>50</option>
                            <option value={100}>100</option>
                          </select>
                        </label>
                      </div>
                    </div>
                    <div className="experiment-sample-grid">
                      {explorerRows.length === 0 ? (
                        <p className="labels-empty">No samples match the selected filters.</p>
                      ) : (
                        explorerRows.map((row, index) => (
                          <button
                            type="button"
                            className="experiment-sample-tile"
                            key={`${row.asset_id}-${index}`}
                            onClick={() => setSelectedSampleImage(row)}
                          >
                            <img src={`/api/v1/assets/${encodeURIComponent(row.asset_id)}/content`} alt={row.asset_id} loading="lazy" />
                            <span>True: {classNames[row.true_class_index] ?? row.true_class_index}</span>
                            <span>Pred: {classNames[row.pred_class_index] ?? row.pred_class_index}</span>
                            <span>Conf: {row.confidence.toFixed(4)}</span>
                          </button>
                        ))
                      )}
                    </div>
                  </div>
                </>
              ) : null}
            </section>
          ) : null}

          {saveError ? <p className="project-field-error">{saveError}</p> : null}
          {!validation.isValid ? (
            <ul className="status-errors">
              {validation.issues.map((issue) => (
                <li key={issue}>{issue}</li>
              ))}
            </ul>
          ) : null}

          {!isLoading ? (
            <footer className="model-builder-footer experiment-actions-row">
              <button
                type="button"
                className="ghost-button"
                disabled={!isEditable || !isDirty || !validation.isValid || isSaving || !draftConfig}
                onClick={() => void handleSave()}
              >
                {isSaving ? "Saving..." : "Save"}
              </button>
              {status === "running" || status === "queued" ? (
                <button type="button" className="ghost-button" disabled={isCanceling} onClick={() => void handleCancel()}>
                  {isCanceling ? "Canceling..." : status === "queued" ? "Cancel Queue" : "Cancel"}
                </button>
              ) : (
                <button
                  type="button"
                  className="primary-button"
                  disabled={!isEditable || isStarting || !validation.isValid || !draftConfig}
                  onClick={() => void handleStart()}
                >
                  {isStarting ? "Starting..." : "Start Training"}
                </button>
              )}
            </footer>
          ) : null}
        </section>
      </main>

      {cellDrawer ? (
        <div className="experiment-modal-backdrop" role="dialog" aria-modal="true" aria-label="Confusion cell samples">
          <div className="experiment-modal">
            <header className="project-section-header">
              <h3>
                Cell Samples: true {classNames[cellDrawer.trueClassIndex] ?? cellDrawer.trueClassIndex} / pred{" "}
                {classNames[cellDrawer.predClassIndex] ?? cellDrawer.predClassIndex}
              </h3>
              <button type="button" className="ghost-button" onClick={() => setCellDrawer(null)}>
                Close
              </button>
            </header>
            {cellSamplesMessage ? <p className="project-field-error">{cellSamplesMessage}</p> : null}
            <div className="experiment-sample-grid">
              {cellSamples.length < 1 ? (
                <p className="labels-empty">No samples available for this confusion cell.</p>
              ) : (
                cellSamples.map((row, index) => (
                  <button
                    type="button"
                    className="experiment-sample-tile"
                    key={`${row.asset_id}-${index}`}
                    onClick={() => setSelectedSampleImage(row)}
                  >
                    <img src={`/api/v1/assets/${encodeURIComponent(row.asset_id)}/content`} alt={row.asset_id} loading="lazy" />
                    <span>True: {classNames[row.true_class_index] ?? row.true_class_index}</span>
                    <span>Pred: {classNames[row.pred_class_index] ?? row.pred_class_index}</span>
                    <span>Conf: {row.confidence.toFixed(4)}</span>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>
      ) : null}

      {selectedSampleImage ? (
        <div className="experiment-modal-backdrop" role="dialog" aria-modal="true" aria-label="Sample image preview">
          <div className="experiment-modal experiment-image-modal">
            <header className="project-section-header">
              <h3>Sample Preview</h3>
              <button type="button" className="ghost-button" onClick={() => setSelectedSampleImage(null)}>
                Close
              </button>
            </header>
            <img
              className="experiment-sample-preview-image"
              src={`/api/v1/assets/${encodeURIComponent(selectedSampleImage.asset_id)}/content`}
              alt={selectedSampleImage.asset_id}
            />
            <p>
              True: {classNames[selectedSampleImage.true_class_index] ?? selectedSampleImage.true_class_index} | Pred:{" "}
              {classNames[selectedSampleImage.pred_class_index] ?? selectedSampleImage.pred_class_index} | Confidence:{" "}
              {selectedSampleImage.confidence.toFixed(4)}
            </p>
          </div>
        </div>
      ) : null}

      {toastMessage ? (
        <div className={`status-toast ${toastTone === "error" ? "is-error" : "is-success"}`} role="status" aria-live="polite">
          <span>{toastMessage}</span>
          <button type="button" aria-label="Dismiss message" onClick={() => setToastMessage(null)}>
            x
          </button>
        </div>
      ) : null}
    </>
  );
}
