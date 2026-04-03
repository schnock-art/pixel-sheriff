"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useProjectNavigationGuard } from "../../../../../components/workspace/ProjectNavigationContext";
import {
  ApiError,
  cancelExperiment,
  createDeployment,
  deleteExperiment,
  getExperiment,
  getExperimentEvaluation,
  getExperimentLogs,
  getExperimentOnnx,
  getExperimentRuntime,
  getExperimentVariants,
  listExperimentSamples,
  listDeployments,
  listDatasetVersions,
  listProjectModels,
  resolveAssetUri,
  startExperiment,
  streamExperimentEvents,
  triggerExperimentFp16,
  triggerExperimentPtq,
  triggerExperimentQat,
  updateExperiment,
  type ExperimentCheckpoint,
  type ExperimentEvaluationPayload,
  type ExperimentEvaluationSampleRow,
  type ExperimentMetricPoint,
  type ExperimentVariantSummary,
  type ExperimentVariantsPayload,
  type ExperimentOnnxPayload,
  type ExperimentRuntimePayload,
  type ExperimentStatus,
  type ModelVariantKey,
  type ProjectExperimentRecord,
} from "../../../../../lib/api";
import {
  buildLinePoints,
  buildTicks,
  buildExperimentMetricChartModel,
  buildExperimentMetricHoverModel,
  computeSeriesDomain,
  findNearestMetricEpoch,
  formatTick,
  indexCheckpointsByKind,
  mergeMetricPoints,
  metricKeyForTask,
  metricValueByKey,
} from "../../../../../lib/workspace/experimentMetrics";
import {
  appendQatDashboardSeries,
  dashboardSeriesForTask,
  dashboardTabsForTask,
  filterPredictionRows,
  normalizeConfusion,
} from "../../../../../lib/workspace/experimentDashboard";
import { buildDatasetVersionOptions } from "../../../../../lib/workspace/experimentDatasetSelection";
import {
  asRecord,
  asYesNo,
  cloneConfig,
  configValidation,
  formatCheckpoint,
  formatDateTime,
  formatDurationSeconds,
  formatEtaClock,
  patchNumber,
} from "../../../../../lib/workspace/experimentDetail";
import { onnxClassNamesText, onnxInputShapeText, onnxStatusLabel, onnxValidationText } from "../../../../../lib/workspace/experimentOnnx";
import { mergeLogChunk, runtimeBadgeLabel } from "../../../../../lib/workspace/experimentRuntime";
import { describeQatSupport, describeQatVariant } from "../../../../../lib/workspace/experimentVariants";
import { deploymentTaskForExperiment } from "../../../../../lib/workspace/deployHelpers.js";
import {
  addAugmentationStep,
  createAugmentationStep,
  moveAugmentationStep,
  readAugmentationProfile,
  readAugmentationSteps,
  removeAugmentationStep,
  setAugmentationProfile,
  updateAugmentationStep,
} from "../../../../../lib/workspace/augmentationConfig";

interface ExperimentDetailPageProps {
  params: {
    projectId: string;
    experimentId: string;
  };
}

type DashboardChartTab = "loss" | "accuracy" | "prf" | "map" | "quality" | "counts" | "runtime";
type PerClassSortKey =
  | "f1_desc"
  | "f1_asc"
  | "precision_desc"
  | "recall_desc"
  | "support_desc"
  | "ap50_desc"
  | "ap75_desc"
  | "map_50_95_desc"
  | "fp_desc"
  | "fn_desc";

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

function asFiniteNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function formatMetricValue(value: unknown, digits = 4): string {
  const numeric = asFiniteNumber(value);
  return numeric == null ? "-" : numeric.toFixed(digits);
}

function formatCountValue(value: unknown): string {
  const numeric = asFiniteNumber(value);
  return numeric == null ? "-" : String(Math.round(numeric));
}

function metricLabelForKey(key: string): string {
  if (key === "val_map") return "val mAP@50";
  if (key === "val_map_50_95") return "val mAP@50:95";
  if (key === "val_iou") return "val IoU";
  return key.replace("val_", "val ").replace(/_/g, " ");
}

function formatBytes(value: unknown): string {
  const numeric = asFiniteNumber(value);
  if (numeric == null || numeric < 0) return "-";
  if (numeric < 1024) return `${Math.round(numeric)} B`;
  if (numeric < 1024 * 1024) return `${(numeric / 1024).toFixed(1)} KB`;
  return `${(numeric / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDelta(value: unknown, baseline: unknown, digits = 4): string {
  const numeric = asFiniteNumber(value);
  const baselineNumeric = asFiniteNumber(baseline);
  if (numeric == null || baselineNumeric == null) return "-";
  const delta = numeric - baselineNumeric;
  const sign = delta > 0 ? "+" : "";
  return `${sign}${delta.toFixed(digits)}`;
}

function variantMetricKeysForTask(task: string): string[] {
  if (task === "detection") return ["mAP50", "mAP50_95", "precision", "recall"];
  return ["accuracy", "macro_f1", "macro_precision", "macro_recall"];
}

export default function ExperimentDetailPage({ params }: ExperimentDetailPageProps) {
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);
  const experimentId = useMemo(() => decodeURIComponent(params.experimentId), [params.experimentId]);
  const router = useRouter();
  const { setHasUnsavedDrafts } = useProjectNavigationGuard();

  const [savedRecord, setSavedRecord] = useState<ProjectExperimentRecord | null>(null);
  const [draftName, setDraftName] = useState("");
  const [draftConfig, setDraftConfig] = useState<Record<string, unknown> | null>(null);
  const [metrics, setMetrics] = useState<ExperimentMetricPoint[]>([]);
  const [checkpoints, setCheckpoints] = useState<ExperimentCheckpoint[]>([]);
  const [status, setStatus] = useState<ExperimentStatus>("draft");
  const [modelName, setModelName] = useState<string | null>(null);
  const [datasetVersionOptions, setDatasetVersionOptions] = useState<Array<{ id: string; name: string }>>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [showStartChoiceModal, setShowStartChoiceModal] = useState(false);
  const [isCanceling, setIsCanceling] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isDeploying, setIsDeploying] = useState(false);
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
  const [onnxInfo, setOnnxInfo] = useState<ExperimentOnnxPayload | null>(null);
  const [onnxError, setOnnxError] = useState<string | null>(null);
  const [isOnnxLoading, setIsOnnxLoading] = useState(false);
  const [variantsInfo, setVariantsInfo] = useState<ExperimentVariantsPayload | null>(null);
  const [variantsError, setVariantsError] = useState<string | null>(null);
  const [isVariantsLoading, setIsVariantsLoading] = useState(false);
  const [selectedVariantKey, setSelectedVariantKey] = useState<ModelVariantKey | null>(null);
  const [variantSplit, setVariantSplit] = useState<"val" | "test">("val");
  const [variantBenchmarkDevice, setVariantBenchmarkDevice] = useState<"cpu" | "cuda">("cpu");
  const [isTriggeringFp16, setIsTriggeringFp16] = useState(false);
  const [isTriggeringPtq, setIsTriggeringPtq] = useState(false);
  const [isTriggeringQat, setIsTriggeringQat] = useState(false);
  const [ptqCalibrationMaxSamples, setPtqCalibrationMaxSamples] = useState("256");
  const [qatCalibrationMaxSamples, setQatCalibrationMaxSamples] = useState("256");
  const [qatEpochsOverride, setQatEpochsOverride] = useState("");
  const [qatLearningRateOverride, setQatLearningRateOverride] = useState("");
  const [logsContent, setLogsContent] = useState("");
  const [logsCursor, setLogsCursor] = useState(0);
  const [logsAttempt, setLogsAttempt] = useState<number | null>(null);
  const [logsError, setLogsError] = useState<string | null>(null);
  const [isLogsLoading, setIsLogsLoading] = useState(false);
  const [isLogsExpanded, setIsLogsExpanded] = useState(true);
  const [logsAutoRefresh, setLogsAutoRefresh] = useState(true);
  const [dashboardChartTab, setDashboardChartTab] = useState<DashboardChartTab>("loss");
  const [dashboardEnabledSeries, setDashboardEnabledSeries] = useState<Record<string, boolean>>({});
  const [dashboardLogScale, setDashboardLogScale] = useState(false);
  const [confusionNormalize, setConfusionNormalize] = useState<"none" | "by_true" | "by_pred">("none");
  const [perClassSort, setPerClassSort] = useState<PerClassSortKey>("f1_desc");
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
  const activeAttemptRef = useRef<number | null>(null);

  const isEditable = status === "draft" || status === "failed" || status === "canceled";
  const isRunningLike = status === "running" || status === "queued";
  const task = (typeof draftConfig?.task === "string" ? draftConfig.task : "classification") as string;
  const runtimeBadge = useMemo(() => runtimeBadgeLabel(runtimeInfo), [runtimeInfo]);
  const onnxStatus = onnxStatusLabel(onnxInfo, status);
  const onnxInputShape = onnxInputShapeText(onnxInfo);
  const onnxClassSummary = onnxClassNamesText(onnxInfo);
  const onnxValidationSummary = onnxValidationText(onnxInfo);
  const preferredVariantKey = variantsInfo?.preferred_variant_key ?? onnxInfo?.preferred_variant_key ?? null;
  const availableVariantKeys = useMemo<ModelVariantKey[]>(
    () =>
      (variantsInfo?.variants ? Object.keys(variantsInfo.variants) : onnxInfo?.available_variants ?? []).filter((value): value is ModelVariantKey =>
        value === "fp32" || value === "fp16" || value === "ptq_int8" || value === "qat_int8",
      ),
    [onnxInfo?.available_variants, variantsInfo?.variants],
  );
  const hasActiveVariantJob = useMemo(
    () =>
      Object.values(variantsInfo?.variants ?? {}).some(
        (variant) => variant && (variant.status === "queued" || variant.status === "running"),
      ),
    [variantsInfo?.variants],
  );
  const primaryMetricKey = metricKeyForTask(task);
  const primaryMetricLabel = metricLabelForKey(primaryMetricKey);
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
  const variantRows = variantsInfo?.variants ?? {};
  const baselineVariant = variantRows.fp32 ?? null;
  const fp16Variant = variantRows.fp16 ?? null;
  const ptqVariant = variantRows.ptq_int8 ?? null;
  const qatVariant = variantRows.qat_int8 ?? null;
  const qatSupportSummary = useMemo(() => describeQatSupport(variantsInfo?.support ?? null), [variantsInfo?.support]);

  const {
    chartWidth,
    chartHeight,
    chartPadding,
    chartInnerWidth,
    chartInnerHeight,
    chartMaxEpoch,
    primaryMetricIsBounded,
    useSecondaryAxis,
    leftAxisDomain,
    rightAxisDomain,
    leftAxisTicks,
    rightAxisTicks,
    xTickValues,
    seriesLegend,
    primaryLinePoints,
    valLossLinePoints,
  } = useMemo(
    () =>
      buildExperimentMetricChartModel(metrics, {
        primaryMetricKey,
        primaryMetricLabel,
        showPrimary,
        showValLoss,
        chartWidth: 760,
        chartHeight: 280,
        chartPadding: 44,
        primaryColor,
        lossColor,
      }),
    [lossColor, metrics, primaryColor, primaryMetricKey, primaryMetricLabel, showPrimary, showValLoss],
  );

  const { hoveredEpochValue, hoveredX, hoveredPlotRows, hoverTooltip } = useMemo(
    () =>
      buildExperimentMetricHoverModel(metrics, {
        hoveredEpoch,
        seriesLegend,
        chartWidth,
        chartHeight,
        chartPadding,
        chartInnerWidth,
        chartInnerHeight,
        chartMaxEpoch,
        leftAxisDomain,
        lossDomain: rightAxisDomain ?? leftAxisDomain,
        useSecondaryAxis,
      }),
    [
      chartHeight,
      chartInnerHeight,
      chartInnerWidth,
      chartMaxEpoch,
      chartPadding,
      chartWidth,
      hoveredEpoch,
      leftAxisDomain,
      metrics,
      rightAxisDomain,
      seriesLegend,
      useSecondaryAxis,
    ],
  );

  function handleChartMouseMove(event: React.MouseEvent<SVGSVGElement>) {
    if (metrics.length === 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width <= 0) return;
    const relativeX = event.clientX - rect.left;
    const svgX = (relativeX / rect.width) * chartWidth;
    const clampedX = Math.max(chartPadding, Math.min(chartPadding + chartInnerWidth, svgX));
    const approximateEpoch = 1 + (((clampedX - chartPadding) / Math.max(1, chartInnerWidth)) * Math.max(1, chartMaxEpoch - 1));
    setHoveredEpoch(findNearestMetricEpoch(metrics, approximateEpoch));
  }

  const modelId = savedRecord?.model_id ?? "";
  const backToExperimentsHref = `/projects/${encodeURIComponent(projectId)}/experiments`;
  const modelHref = modelId ? `/projects/${encodeURIComponent(projectId)}/models/${encodeURIComponent(modelId)}` : null;

  const loadDetail = useCallback(async () => {
    setIsLoading(true);
    setErrorMessage(null);
    setSaveError(null);
    try {
      const [record, models, datasetVersions] = await Promise.all([
        getExperiment(projectId, experimentId),
        listProjectModels(projectId),
        listDatasetVersions(projectId),
      ]);
      const resolvedModelName = models.find((model) => model.id === record.model_id)?.name ?? record.model_id;
      const configDatasetVersionId =
        typeof record.config_json?.dataset_version_id === "string" ? record.config_json.dataset_version_id : "";
      setDatasetVersionOptions(buildDatasetVersionOptions(datasetVersions.items ?? [], configDatasetVersionId));
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

  const loadOnnx = useCallback(async (options: { silent?: boolean } = {}) => {
    if (!options.silent) setIsOnnxLoading(true);
    try {
      const payload = await getExperimentOnnx(projectId, experimentId, {
        variant: selectedVariantKey ?? "preferred",
      });
      setOnnxInfo(payload);
      setOnnxError(null);
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        setOnnxInfo(null);
        setOnnxError(null);
      } else {
        setOnnxInfo(null);
        setOnnxError(parseApiErrorMessage(error, "Failed to load ONNX export"));
      }
    } finally {
      if (!options.silent) setIsOnnxLoading(false);
    }
  }, [experimentId, projectId, selectedVariantKey]);

  const loadVariants = useCallback(async (options: { silent?: boolean } = {}) => {
    if (!options.silent) setIsVariantsLoading(true);
    try {
      const payload = await getExperimentVariants(projectId, experimentId);
      setVariantsInfo(payload);
      setVariantsError(null);
      setSelectedVariantKey((current) => {
        if (current && payload.variants?.[current]) return current;
        return payload.preferred_variant_key ?? current ?? null;
      });
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        setVariantsInfo(null);
        setVariantsError(null);
      } else {
        setVariantsInfo(null);
        setVariantsError(parseApiErrorMessage(error, "Failed to load model variants"));
      }
    } finally {
      if (!options.silent) setIsVariantsLoading(false);
    }
  }, [experimentId, projectId]);

  const fetchLogsChunk = useCallback(
    async (reset = false) => {
      const requestedAttempt = activeAttempt;
      setIsLogsLoading(true);
      try {
        const chunk = await getExperimentLogs(projectId, experimentId, {
          attempt: requestedAttempt ?? undefined,
          fromByte: reset ? 0 : logsCursorRef.current,
          maxBytes: 65536,
        });
        if ((activeAttemptRef.current ?? null) !== (requestedAttempt ?? null)) {
          return;
        }
        setLogsError(null);
        const merged = mergeLogChunk(reset ? "" : logsContentRef.current, chunk, { maxBytes: 200 * 1024, maxLines: 5000 });
        logsContentRef.current = merged.content;
        logsCursorRef.current = merged.cursor;
        setLogsContent(merged.content);
        setLogsCursor(merged.cursor);
        setLogsAttempt(chunk.attempt);
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) {
          if ((activeAttemptRef.current ?? null) !== (requestedAttempt ?? null)) {
            return;
          }
          if (reset) {
            logsContentRef.current = "";
            logsCursorRef.current = 0;
            setLogsCursor(0);
            setLogsContent("");
            setLogsAttempt(requestedAttempt ?? null);
          }
          setLogsError(null);
        } else {
          setLogsError(parseApiErrorMessage(error, "Failed to load training logs"));
        }
      } finally {
        setIsLogsLoading(false);
      }
    },
    [activeAttempt, experimentId, projectId],
  );

  useEffect(() => {
    void loadDetail();
  }, [loadDetail]);

  useEffect(() => {
    void loadRuntime();
  }, [loadRuntime, status]);

  useEffect(() => {
    void loadOnnx();
  }, [loadOnnx, status]);

  useEffect(() => {
    void loadVariants();
  }, [loadVariants, status]);

  useEffect(() => {
    activeAttemptRef.current = activeAttempt;
  }, [activeAttempt]);

  useEffect(() => {
    logsCursorRef.current = 0;
    logsContentRef.current = "";
    setLogsCursor(0);
    setLogsContent("");
    setLogsAttempt(activeAttempt);
    setLogsError(null);
    void fetchLogsChunk(true);
  }, [activeAttempt, experimentId, fetchLogsChunk]);

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
        if (event.type === "onnx_export") {
          if (typeof event.attempt === "number") setActiveAttempt(event.attempt);
          void loadOnnx();
          void loadVariants();
          return;
        }
        if (event.type === "variant_status") {
          if (typeof event.attempt === "number") setActiveAttempt(event.attempt);
          void loadVariants();
          void loadOnnx();
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
          void loadVariants();
        }
      },
    });
    return () => stop();
  }, [activeAttempt, experimentId, isRunningLike, loadDetail, loadOnnx, loadVariants, projectId]);

  useEffect(() => {
    if (!hasActiveVariantJob) return;
    let cancelled = false;

    async function refreshVariantArtifacts() {
      await Promise.all([
        loadVariants({ silent: true }),
        loadOnnx({ silent: true }),
      ]);
    }

    void refreshVariantArtifacts();
    const timer = window.setInterval(() => {
      if (cancelled) return;
      void refreshVariantArtifacts();
    }, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [hasActiveVariantJob, loadOnnx, loadVariants]);

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

  async function saveExperimentDraft(): Promise<boolean> {
    if (!draftConfig || !savedRecord) return false;
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
      return true;
    } catch (error) {
      const message = parseApiErrorMessage(error, "Failed to save experiment");
      setSaveError(message);
      setToastTone("error");
      setToastMessage(`Save failed: ${message}`);
      return false;
    } finally {
      setIsSaving(false);
    }
  }

  async function handleSave() {
    await saveExperimentDraft();
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

  function handleStartClick() {
    if (isDirty) {
      setShowStartChoiceModal(true);
      return;
    }
    void handleStart();
  }

  async function handleSaveAndStart() {
    const saved = await saveExperimentDraft();
    if (!saved) return;
    setShowStartChoiceModal(false);
    await handleStart();
  }

  async function handleDeployModel() {
    if (!savedRecord || !onnxInfo?.attempt) return;
    setIsDeploying(true);
    try {
      const deploymentList = await listDeployments(projectId);
      const deploymentNameBase = (savedRecord.name || draftName || `deploy_${experimentId.slice(0, 8)}`).trim();
      await createDeployment(projectId, {
        name: `${deploymentNameBase} run ${onnxInfo.attempt} ${onnxInfo.variant_key ?? "preferred"}`,
        task: deploymentTaskForExperiment(savedRecord.task ?? task),
        device_preference: "auto",
        source: {
          experiment_id: experimentId,
          attempt: onnxInfo.attempt,
          checkpoint_kind: "best_metric",
          variant_key: onnxInfo.variant_key ?? "preferred",
        },
        is_active: deploymentList.items.filter((item) => item.status === "available").length === 0,
      });
      setToastTone("success");
      setToastMessage("Deployment created");
      router.push(`/projects/${encodeURIComponent(projectId)}/deploy`);
    } catch (error) {
      const message = parseApiErrorMessage(error, "Failed to deploy model");
      setToastTone("error");
      setToastMessage(message);
    } finally {
      setIsDeploying(false);
    }
  }

  async function handleTriggerPtq() {
    setIsTriggeringPtq(true);
    try {
      const calibrationMaxSamples = Number.parseInt(ptqCalibrationMaxSamples, 10);
      await triggerExperimentPtq(projectId, experimentId, {
        calibration_max_samples: Number.isFinite(calibrationMaxSamples) && calibrationMaxSamples >= 1 ? calibrationMaxSamples : 256,
      });
      setToastTone("success");
      setToastMessage("PTQ queued");
      void loadVariants();
    } catch (error) {
      const message = parseApiErrorMessage(error, "Failed to start PTQ");
      setToastTone("error");
      setToastMessage(message);
    } finally {
      setIsTriggeringPtq(false);
    }
  }

  async function handleTriggerFp16() {
    setIsTriggeringFp16(true);
    try {
      await triggerExperimentFp16(projectId, experimentId);
      setToastTone("success");
      setToastMessage("FP16 queued");
      void loadVariants();
    } catch (error) {
      const message = parseApiErrorMessage(error, "Failed to start FP16 export");
      setToastTone("error");
      setToastMessage(message);
    } finally {
      setIsTriggeringFp16(false);
    }
  }

  async function handleTriggerQat() {
    setIsTriggeringQat(true);
    try {
      const calibrationMaxSamples = Number.parseInt(qatCalibrationMaxSamples, 10);
      await triggerExperimentQat(projectId, experimentId, {
        epochs_override: patchNumber(qatEpochsOverride) ?? undefined,
        learning_rate_override: patchNumber(qatLearningRateOverride) ?? undefined,
        calibration_max_samples: Number.isFinite(calibrationMaxSamples) && calibrationMaxSamples >= 1 ? calibrationMaxSamples : 256,
      });
      setToastTone("success");
      setToastMessage("QAT queued");
      void loadVariants();
    } catch (error) {
      const message = parseApiErrorMessage(error, "Failed to start QAT");
      setToastTone("error");
      setToastMessage(message);
    } finally {
      setIsTriggeringQat(false);
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

  async function handleDelete() {
    const confirmed = window.confirm(
      'Delete this experiment and all artifacts?\n\nThis removes runs, logs, checkpoints, evaluation, runtime, predictions, and ONNX artifacts.',
    );
    if (!confirmed) return;

    setIsDeleting(true);
    try {
      await deleteExperiment(projectId, experimentId);
      router.replace(`/projects/${encodeURIComponent(projectId)}/experiments`);
    } catch (error) {
      const message = parseApiErrorMessage(error, "Failed to delete experiment");
      setToastTone("error");
      setToastMessage(message);
    } finally {
      setIsDeleting(false);
    }
  }

  const optimizer = asRecord(draftConfig?.optimizer);
  const advanced = asRecord(draftConfig?.advanced);
  const runtimeConfig = asRecord(draftConfig?.runtime);
  const datasetVersionId = typeof draftConfig?.dataset_version_id === "string" ? draftConfig.dataset_version_id : "-";
  const optimizerType = typeof optimizer.type === "string" ? optimizer.type : "adam";
  const learningRate = typeof optimizer.lr === "number" ? String(optimizer.lr) : "";
  const weightDecay = typeof optimizer.weight_decay === "number" ? String(optimizer.weight_decay) : "0";
  const momentum = typeof optimizer.momentum === "number" ? String(optimizer.momentum) : "0.9";
  const scheduler = asRecord(draftConfig?.scheduler);
  const schedulerParams = asRecord(scheduler.params);
  const schedulerType = typeof scheduler.type === "string" ? scheduler.type : "none";
  const schedulerStepSize = typeof schedulerParams.step_size === "number" ? String(schedulerParams.step_size) : "10";
  const schedulerGamma = typeof schedulerParams.gamma === "number" ? String(schedulerParams.gamma) : "0.1";
  const epochs = typeof draftConfig?.epochs === "number" ? String(draftConfig.epochs) : "";
  const batchSize = typeof draftConfig?.batch_size === "number" ? String(draftConfig.batch_size) : "";
  const augmentationProfile = readAugmentationProfile(draftConfig ?? {}, task);
  const augmentationSteps = readAugmentationSteps(draftConfig ?? {});
  const precision = typeof draftConfig?.precision === "string" ? draftConfig.precision : "fp32";
  const seed = typeof advanced.seed === "number" ? String(advanced.seed) : "1337";
  const gradClipNorm = typeof advanced.grad_clip_norm === "number" ? String(advanced.grad_clip_norm) : "";
  const evaluationConfig = asRecord(draftConfig?.evaluation);
  const evalInterval = typeof evaluationConfig.eval_interval_epochs === "number" ? String(evaluationConfig.eval_interval_epochs) : "1";
  const resumeConfig = asRecord(draftConfig?.resume);
  const resumeEnabled = Boolean(resumeConfig.enabled);
  const resumeCheckpointKind = typeof resumeConfig.checkpoint_kind === "string" ? resumeConfig.checkpoint_kind : "latest";
  const numWorkers =
    typeof runtimeConfig.num_workers === "number"
      ? String(runtimeConfig.num_workers)
      : typeof advanced.num_workers === "number"
        ? String(advanced.num_workers)
        : "0";
  const pinMemoryMode = typeof runtimeConfig.pin_memory === "boolean" ? (runtimeConfig.pin_memory ? "true" : "false") : "auto";
  const persistentWorkersMode =
    typeof runtimeConfig.persistent_workers === "boolean" ? (runtimeConfig.persistent_workers ? "true" : "false") : "auto";
  const prefetchFactor = typeof runtimeConfig.prefetch_factor === "number" ? String(runtimeConfig.prefetch_factor) : "2";
  const cacheResizedImages = typeof runtimeConfig.cache_resized_images === "boolean" ? runtimeConfig.cache_resized_images : true;
  const maxCachedImages = typeof runtimeConfig.max_cached_images === "number" ? String(runtimeConfig.max_cached_images) : "1024";
  const latestMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const latestEpochSeconds = typeof latestMetric?.epoch_seconds === "number" ? latestMetric.epoch_seconds : null;
  const latestEtaSeconds = typeof latestMetric?.eta_seconds === "number" ? latestMetric.eta_seconds : null;
  const latestEtaClock = formatEtaClock(latestEtaSeconds);
  const isClassificationTask = task === "classification";
  const isDetectionTask = task === "detection";
  const dashboardSupported = isClassificationTask || isDetectionTask;
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
    if (perClassSort === "ap50_desc") rows.sort((a, b) => (asFiniteNumber(b.ap50) ?? -1) - (asFiniteNumber(a.ap50) ?? -1));
    if (perClassSort === "ap75_desc") rows.sort((a, b) => (asFiniteNumber(b.ap75) ?? -1) - (asFiniteNumber(a.ap75) ?? -1));
    if (perClassSort === "map_50_95_desc") rows.sort((a, b) => (asFiniteNumber(b.map_50_95) ?? -1) - (asFiniteNumber(a.map_50_95) ?? -1));
    if (perClassSort === "fp_desc") rows.sort((a, b) => (asFiniteNumber(b.fp) ?? -1) - (asFiniteNumber(a.fp) ?? -1));
    if (perClassSort === "fn_desc") rows.sort((a, b) => (asFiniteNumber(b.fn) ?? -1) - (asFiniteNumber(a.fn) ?? -1));
    return rows;
  }, [evaluation?.per_class, perClassSort]);

  const dashboardTabs = useMemo(() => dashboardTabsForTask(task), [task]);
  const qatMetrics = useMemo(
    () => (Array.isArray(qatVariant?.metrics) ? qatVariant.metrics : []),
    [qatVariant?.metrics],
  );
  const dashboardAvailableSeries = useMemo(
    () => appendQatDashboardSeries(dashboardSeriesForTask(task, dashboardChartTab), qatMetrics),
    [dashboardChartTab, qatMetrics, task],
  );
  const dashboardSeries = useMemo(
    () => dashboardAvailableSeries.filter((series) => dashboardEnabledSeries[series.key] !== false),
    [dashboardAvailableSeries, dashboardEnabledSeries],
  );
  const dashboardBounded =
    dashboardChartTab === "accuracy" ||
    dashboardChartTab === "prf" ||
    dashboardChartTab === "map" ||
    dashboardChartTab === "quality";
  const dashboardHasVisibleSeries = dashboardSeries.length > 0;
  const dashboardEpochMax = useMemo(() => {
    const epochs = dashboardSeries.flatMap((series) =>
      (series.source === "qat" ? qatMetrics : metrics)
        .map((row) => Number.parseInt(String(row?.epoch), 10))
        .filter((epoch) => Number.isFinite(epoch) && epoch >= 1),
    );
    if (epochs.length < 1) return 1;
    return Math.max(...epochs);
  }, [dashboardSeries, metrics, qatMetrics]);
  const dashboardHasData = useMemo(
    () =>
      dashboardSeries.length > 0 &&
      dashboardSeries.some((series) =>
        (series.source === "qat" ? qatMetrics : metrics).some((row) => {
          const value = metricValueByKey(row, series.metricKey ?? series.key);
          return value != null;
        }),
      ),
    [dashboardSeries, metrics, qatMetrics],
  );
  const dashboardAxisLabel =
    dashboardChartTab === "loss"
      ? "Loss"
      : dashboardChartTab === "runtime"
        ? "Seconds"
        : dashboardChartTab === "counts"
          ? "Count"
          : "Metric value";
  const dashboardValues = useMemo(
    () =>
      dashboardSeries.flatMap((series) =>
        (series.source === "qat" ? qatMetrics : metrics)
          .map((row) => metricValueByKey(row, series.metricKey ?? series.key))
          .filter((value): value is number => value != null),
      ),
    [dashboardSeries, metrics, qatMetrics],
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
    () => Array.from(new Set(buildTicks({ min: 1, max: dashboardEpochMax }, { count: 5 }).map((tick) => Math.max(1, Math.round(tick))))),
    [dashboardEpochMax],
  );
  const dashboardLinePoints = useMemo(
    () =>
      dashboardSeries.map((series) => ({
        ...series,
        points: buildLinePoints(series.source === "qat" ? qatMetrics : metrics, series.metricKey ?? series.key, {
          width: chartWidth,
          height: chartHeight,
          padding: chartPadding,
          domain: dashboardDomain,
          useLog: dashboardLogScale,
          maxEpoch: dashboardEpochMax,
        }),
      })),
    [chartHeight, chartPadding, chartWidth, dashboardDomain, dashboardEpochMax, dashboardLogScale, dashboardSeries, metrics, qatMetrics],
  );

  useEffect(() => {
    if (dashboardTabs.some((tab) => tab.key === dashboardChartTab)) return;
    const nextTab = dashboardTabs[0]?.key;
    if (
      nextTab === "loss" ||
      nextTab === "accuracy" ||
      nextTab === "prf" ||
      nextTab === "map" ||
      nextTab === "quality" ||
      nextTab === "counts" ||
      nextTab === "runtime"
    ) {
      setDashboardChartTab(nextTab);
    }
  }, [dashboardChartTab, dashboardTabs]);

  useEffect(() => {
    if (!dashboardSupported) {
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
  }, [dashboardSupported, experimentId, projectId, status]);

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

  const detectionOverall = isDetectionTask ? (evaluation?.overall ?? null) : null;
  const detectionOverallMap50 = asFiniteNumber(detectionOverall?.mAP50);
  const detectionOverallMap50_95 = asFiniteNumber(detectionOverall?.mAP50_95);
  const detectionSizeBucketRows = useMemo(() => {
    if (!detectionOverall?.size_buckets || typeof detectionOverall.size_buckets !== "object") return [];
    return Object.entries(detectionOverall.size_buckets).map(([name, value]) => ({
      name,
      groundTruthCount: asFiniteNumber(value?.ground_truth_count),
      predictionCount: asFiniteNumber(value?.prediction_count),
      ap50: asFiniteNumber(value?.ap50),
      map50_95: asFiniteNumber(value?.map_50_95),
      precision: asFiniteNumber(value?.precision),
      recall: asFiniteNumber(value?.recall),
    }));
  }, [detectionOverall?.size_buckets]);

  function renderVariantComparisonTable(rows: Array<ExperimentVariantSummary | null>, tableId: string) {
    const metricKeys = variantMetricKeysForTask(task);
    const baselineOverall = baselineVariant?.evaluation?.[variantSplit]?.overall ?? null;
    const visibleRows = rows.filter((row): row is ExperimentVariantSummary => Boolean(row));
    if (!visibleRows.length) {
      return <p className="experiment-log-cursor">No model variants available yet.</p>;
    }
    return (
      <div className="table-card">
        <table className="project-table" data-testid={tableId}>
          <thead>
            <tr>
              <th>Variant</th>
              <th>Status</th>
              {metricKeys.map((key) => (
                <th key={key}>{key.replace(/_/g, " ")}</th>
              ))}
              <th>Size</th>
              <th>Mean latency</th>
              <th>Throughput</th>
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row) => {
              const splitSummary = row.evaluation?.[variantSplit];
              const overall = splitSummary?.overall ?? null;
              const benchmarkSummary =
                row.benchmarks && typeof row.benchmarks === "object"
                  ? (row.benchmarks[variantBenchmarkDevice] ?? null)
                  : typeof row.benchmark === "object" && row.benchmark
                    ? row.benchmark
                    : null;
              const qatWarning = describeQatVariant(row);
              const meanLatency = asFiniteNumber(benchmarkSummary?.mean_latency_ms);
              const throughput = asFiniteNumber(benchmarkSummary?.throughput_items_per_second);
              const benchmarkMessage = typeof benchmarkSummary?.message === "string" ? benchmarkSummary.message : null;
              return (
                <tr key={row.variant_key}>
                  <td>
                    <strong>{row.label}</strong>
                    {row.preferred ? <div className="experiment-log-cursor">Preferred export</div> : null}
                    {qatWarning ? <div className="experiment-log-cursor">{qatWarning}</div> : null}
                  </td>
                  <td>{row.status}</td>
                  {metricKeys.map((key) => (
                    <td key={key}>
                      {formatMetricValue(overall?.[key], 4)}
                      {baselineOverall && row.variant_key !== "fp32" ? (
                        <div className="experiment-log-cursor">Delta {formatDelta(overall?.[key], baselineOverall?.[key])}</div>
                      ) : null}
                    </td>
                  ))}
                  <td>{formatBytes(row.onnx?.size_bytes)}</td>
                  <td>
                    {formatMetricValue(meanLatency, 2)}
                    {meanLatency != null ? " ms" : ""}
                    {benchmarkSummary?.status === "unavailable" && benchmarkMessage ? (
                      <div className="experiment-log-cursor">{benchmarkMessage}</div>
                    ) : null}
                  </td>
                  <td>{formatMetricValue(throughput, 2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }

  return (
    <>
      <main className="workspace-shell project-page-shell" data-testid="experiment-detail-page">
        <section className="workspace-frame project-content-frame">
          <header className="project-section-header">
            <div className="experiment-header-title">
              <h2>Train Experiment</h2>
              <p>
                Model:{" "}
                {modelHref ? (
                  <Link href={modelHref}>
                    <strong>{modelName ?? modelId}</strong>
                  </Link>
                ) : (
                  <strong>{modelName ?? modelId}</strong>
                )}
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
                    <select
                      value={datasetVersionId === "-" ? "" : datasetVersionId}
                      disabled={!isEditable || datasetVersionOptions.length === 0}
                      onChange={(event) =>
                        patchConfig((next) => {
                          next.dataset_version_id = event.target.value;
                        })
                      }
                    >
                      {datasetVersionOptions.length === 0 ? <option value="">No dataset versions</option> : null}
                      {datasetVersionOptions.map((row) => (
                        <option key={row.id} value={row.id}>
                          {row.name}
                        </option>
                      ))}
                    </select>
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
                    <span>Weight Decay</span>
                    <input
                      type="number"
                      step="0.0001"
                      min="0"
                      value={weightDecay}
                      disabled={!isEditable}
                      onChange={(event) =>
                        patchConfig((next) => {
                          const opt = asRecord(next.optimizer);
                          const parsed = patchNumber(event.target.value);
                          if (parsed !== null && parsed >= 0) opt.weight_decay = parsed;
                          next.optimizer = opt;
                        })
                      }
                    />
                  </label>
                  {optimizerType === "sgd" ? (
                    <label className="project-field">
                      <span>Momentum</span>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        value={momentum}
                        disabled={!isEditable}
                        onChange={(event) =>
                          patchConfig((next) => {
                            const opt = asRecord(next.optimizer);
                            const parsed = patchNumber(event.target.value);
                            if (parsed !== null && parsed >= 0) opt.momentum = parsed;
                            next.optimizer = opt;
                          })
                        }
                      />
                    </label>
                  ) : null}
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
                  <label className="project-field">
                    <span>Scheduler</span>
                    <select
                      value={schedulerType}
                      disabled={!isEditable}
                      onChange={(event) =>
                        patchConfig((next) => {
                          const nextScheduler = asRecord(next.scheduler);
                          nextScheduler.type = event.target.value;
                          if (event.target.value === "step") {
                            nextScheduler.params = { step_size: 10, gamma: 0.1 };
                          } else {
                            nextScheduler.params = {};
                          }
                          next.scheduler = nextScheduler;
                        })
                      }
                    >
                      <option value="none">none</option>
                      <option value="step">step</option>
                      <option value="cosine">cosine</option>
                    </select>
                  </label>
                  {schedulerType === "step" ? (
                    <>
                      <label className="project-field">
                        <span>Step Size</span>
                        <input
                          type="number"
                          min="1"
                          step="1"
                          value={schedulerStepSize}
                          disabled={!isEditable}
                          onChange={(event) =>
                            patchConfig((next) => {
                              const nextScheduler = asRecord(next.scheduler);
                              const nextParams = asRecord(nextScheduler.params);
                              const parsed = Number.parseInt(event.target.value, 10);
                              if (Number.isFinite(parsed) && parsed >= 1) nextParams.step_size = parsed;
                              nextScheduler.params = nextParams;
                              next.scheduler = nextScheduler;
                            })
                          }
                        />
                      </label>
                      <label className="project-field">
                        <span>Gamma</span>
                        <input
                          type="number"
                          min="0.0001"
                          step="0.01"
                          value={schedulerGamma}
                          disabled={!isEditable}
                          onChange={(event) =>
                            patchConfig((next) => {
                              const nextScheduler = asRecord(next.scheduler);
                              const nextParams = asRecord(nextScheduler.params);
                              const parsed = patchNumber(event.target.value);
                              if (parsed !== null && parsed > 0) nextParams.gamma = parsed;
                              nextScheduler.params = nextParams;
                              next.scheduler = nextScheduler;
                            })
                          }
                        />
                      </label>
                    </>
                  ) : null}
                  <label className="project-field">
                    <span>Augmentation</span>
                    <select
                      value={augmentationProfile}
                      disabled={!isEditable}
                      onChange={(event) =>
                        patchConfig((next) => {
                          setAugmentationProfile(next, event.target.value);
                        })
                      }
                    >
                      <option value="none">none</option>
                      <option value="light">light</option>
                      <option value="medium">medium</option>
                      <option value="heavy">heavy</option>
                      <option value="custom">custom</option>
                    </select>
                  </label>
                  {augmentationProfile === "custom" ? (
                    <div className="project-field" style={{ gap: 10 }}>
                      <span>Custom Augmentation Steps</span>
                      <div style={{ display: "grid", gap: 10 }}>
                        {augmentationSteps.length === 0 ? <p className="labels-empty">Add at least one step to use custom augmentation.</p> : null}
                        {augmentationSteps.map((step, index) => (
                          <div
                            key={`augmentation-step-${index}`}
                            style={{
                              display: "grid",
                              gap: 8,
                              padding: 10,
                              border: "1px solid var(--border-subtle, #d5dce8)",
                              borderRadius: 10,
                              background: "var(--panel-muted, #f8fbff)",
                            }}
                          >
                            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "flex-end" }}>
                              <label className="project-field" style={{ flex: "1 1 180px", marginBottom: 0 }}>
                                <span>Transform</span>
                                <select
                                  value={step.type}
                                  disabled={!isEditable}
                                  onChange={(event) =>
                                    patchConfig((next) => {
                                      updateAugmentationStep(next, index, createAugmentationStep(event.target.value));
                                    })
                                  }
                                >
                                  <option value="horizontal_flip">horizontal_flip</option>
                                  <option value="vertical_flip">vertical_flip</option>
                                  <option value="color_jitter">color_jitter</option>
                                  <option value="rotate">rotate</option>
                                </select>
                              </label>
                              <label className="project-field" style={{ width: 120, marginBottom: 0 }}>
                                <span>Probability</span>
                                <input
                                  type="number"
                                  min="0"
                                  max="1"
                                  step="0.05"
                                  value={String(step.p)}
                                  disabled={!isEditable}
                                  onChange={(event) =>
                                    patchConfig((next) => {
                                      const parsed = Number(event.target.value);
                                      updateAugmentationStep(next, index, {
                                        p: Number.isFinite(parsed) ? Math.max(0, Math.min(1, parsed)) : step.p,
                                      });
                                    })
                                  }
                                />
                              </label>
                            </div>

                            {step.type === "color_jitter" ? (
                              <div style={{ display: "grid", gap: 8, gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))" }}>
                                {(["brightness", "contrast", "saturation", "hue"] as const).map((key) => (
                                  <label key={key} className="project-field" style={{ marginBottom: 0 }}>
                                    <span>{key}</span>
                                    <input
                                      type="number"
                                      min="0"
                                      max={key === "hue" ? "0.5" : undefined}
                                      step="0.01"
                                      value={String(typeof step.params?.[key] === "number" ? step.params[key] : 0)}
                                      disabled={!isEditable}
                                      onChange={(event) =>
                                        patchConfig((next) => {
                                          const parsed = Number(event.target.value);
                                          const nextParams = {
                                            ...(step.params ?? {}),
                                            [key]: Number.isFinite(parsed) ? Math.max(0, parsed) : 0,
                                          };
                                          if (key === "hue" && nextParams.hue > 0.5) nextParams.hue = 0.5;
                                          updateAugmentationStep(next, index, { params: nextParams });
                                        })
                                      }
                                    />
                                  </label>
                                ))}
                              </div>
                            ) : null}

                            {step.type === "rotate" ? (
                              <div style={{ display: "grid", gap: 8, gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))" }}>
                                <label className="project-field" style={{ marginBottom: 0 }}>
                                  <span>Min Degrees</span>
                                  <input
                                    type="number"
                                    step="0.5"
                                    value={String(typeof step.params?.min_degrees === "number" ? step.params.min_degrees : -8)}
                                    disabled={!isEditable}
                                    onChange={(event) =>
                                      patchConfig((next) => {
                                        const parsed = Number(event.target.value);
                                        updateAugmentationStep(next, index, {
                                          params: {
                                            ...(step.params ?? {}),
                                            min_degrees: Number.isFinite(parsed) ? parsed : -8,
                                          },
                                        });
                                      })
                                    }
                                  />
                                </label>
                                <label className="project-field" style={{ marginBottom: 0 }}>
                                  <span>Max Degrees</span>
                                  <input
                                    type="number"
                                    step="0.5"
                                    value={String(typeof step.params?.max_degrees === "number" ? step.params.max_degrees : 8)}
                                    disabled={!isEditable}
                                    onChange={(event) =>
                                      patchConfig((next) => {
                                        const parsed = Number(event.target.value);
                                        updateAugmentationStep(next, index, {
                                          params: {
                                            ...(step.params ?? {}),
                                            max_degrees: Number.isFinite(parsed) ? parsed : 8,
                                          },
                                        });
                                      })
                                    }
                                  />
                                </label>
                              </div>
                            ) : null}

                            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                              <button
                                type="button"
                                className="ghost-button"
                                disabled={!isEditable || index < 1}
                                onClick={() =>
                                  patchConfig((next) => {
                                    moveAugmentationStep(next, index, "up");
                                  })
                                }
                              >
                                Move Up
                              </button>
                              <button
                                type="button"
                                className="ghost-button"
                                disabled={!isEditable || index >= augmentationSteps.length - 1}
                                onClick={() =>
                                  patchConfig((next) => {
                                    moveAugmentationStep(next, index, "down");
                                  })
                                }
                              >
                                Move Down
                              </button>
                              <button
                                type="button"
                                className="ghost-button"
                                disabled={!isEditable}
                                onClick={() =>
                                  patchConfig((next) => {
                                    removeAugmentationStep(next, index);
                                  })
                                }
                              >
                                Remove
                              </button>
                            </div>
                          </div>
                        ))}
                        <div>
                          <button
                            type="button"
                            className="ghost-button"
                            disabled={!isEditable}
                            onClick={() =>
                              patchConfig((next) => {
                                addAugmentationStep(next, "horizontal_flip");
                              })
                            }
                          >
                            Add Step
                          </button>
                        </div>
                      </div>
                    </div>
                  ) : null}

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
                        <span>Gradient Clip Norm</span>
                        <input
                          type="number"
                          min="0"
                          step="0.1"
                          value={gradClipNorm}
                          disabled={!isEditable}
                          onChange={(event) =>
                            patchConfig((next) => {
                              const adv = asRecord(next.advanced);
                              adv.grad_clip_norm = patchNumber(event.target.value) ?? null;
                              next.advanced = adv;
                            })
                          }
                        />
                      </label>
                      <label className="project-field">
                        <span>Eval Interval</span>
                        <input
                          type="number"
                          min="1"
                          step="1"
                          value={evalInterval}
                          disabled={!isEditable}
                          onChange={(event) =>
                            patchConfig((next) => {
                              const evalCfg = asRecord(next.evaluation);
                              const parsed = Number.parseInt(event.target.value, 10);
                              if (Number.isFinite(parsed) && parsed >= 1) evalCfg.eval_interval_epochs = parsed;
                              next.evaluation = evalCfg;
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
                              const runtime = asRecord(next.runtime);
                              const adv = asRecord(next.advanced);
                              const parsed = Number.parseInt(event.target.value, 10);
                              if (Number.isFinite(parsed)) {
                                runtime.num_workers = parsed;
                                adv.num_workers = parsed;
                              }
                              next.runtime = runtime;
                              next.advanced = adv;
                            })
                          }
                        />
                      </label>
                      <label className="model-builder-checkbox">
                        <input
                          type="checkbox"
                          checked={resumeEnabled}
                          disabled={!isEditable}
                          onChange={(event) =>
                            patchConfig((next) => {
                              const resume = asRecord(next.resume);
                              resume.enabled = event.target.checked;
                              next.resume = resume;
                            })
                          }
                        />
                        <span>Resume from checkpoint</span>
                      </label>
                      <label className="project-field">
                        <span>Resume Checkpoint</span>
                        <select
                          value={resumeCheckpointKind}
                          disabled={!isEditable || !resumeEnabled}
                          onChange={(event) =>
                            patchConfig((next) => {
                              const resume = asRecord(next.resume);
                              resume.checkpoint_kind = event.target.value;
                              next.resume = resume;
                            })
                          }
                        >
                          <option value="latest">latest</option>
                          <option value="best_loss">best_loss</option>
                          <option value="best_metric">best_metric</option>
                        </select>
                      </label>
                      <label className="project-field">
                        <span>Pin Memory</span>
                        <select
                          value={pinMemoryMode}
                          disabled={!isEditable}
                          onChange={(event) =>
                            patchConfig((next) => {
                              const runtime = asRecord(next.runtime);
                              if (event.target.value === "auto") {
                                delete runtime.pin_memory;
                              } else {
                                runtime.pin_memory = event.target.value === "true";
                              }
                              next.runtime = runtime;
                            })
                          }
                        >
                          <option value="auto">auto</option>
                          <option value="true">true</option>
                          <option value="false">false</option>
                        </select>
                      </label>
                      <label className="project-field">
                        <span>Persistent Workers</span>
                        <select
                          value={persistentWorkersMode}
                          disabled={!isEditable}
                          onChange={(event) =>
                            patchConfig((next) => {
                              const runtime = asRecord(next.runtime);
                              if (event.target.value === "auto") {
                                delete runtime.persistent_workers;
                              } else {
                                runtime.persistent_workers = event.target.value === "true";
                              }
                              next.runtime = runtime;
                            })
                          }
                        >
                          <option value="auto">auto</option>
                          <option value="true">true</option>
                          <option value="false">false</option>
                        </select>
                      </label>
                      <label className="project-field">
                        <span>Prefetch Factor</span>
                        <input
                          type="number"
                          min="1"
                          step="1"
                          value={prefetchFactor}
                          disabled={!isEditable}
                          onChange={(event) =>
                            patchConfig((next) => {
                              const runtime = asRecord(next.runtime);
                              const parsed = Number.parseInt(event.target.value, 10);
                              if (Number.isFinite(parsed) && parsed >= 1) runtime.prefetch_factor = parsed;
                              next.runtime = runtime;
                            })
                          }
                        />
                      </label>
                      <label className="project-field">
                        <span>Max Cached Images</span>
                        <input
                          type="number"
                          min="0"
                          step="1"
                          value={maxCachedImages}
                          disabled={!isEditable}
                          onChange={(event) =>
                            patchConfig((next) => {
                              const runtime = asRecord(next.runtime);
                              const parsed = Number.parseInt(event.target.value, 10);
                              if (Number.isFinite(parsed) && parsed >= 0) runtime.max_cached_images = parsed;
                              next.runtime = runtime;
                            })
                          }
                        />
                      </label>
                      <label className="model-builder-checkbox">
                        <input
                          type="checkbox"
                          checked={cacheResizedImages}
                          disabled={!isEditable}
                          onChange={(event) =>
                            patchConfig((next) => {
                              const runtime = asRecord(next.runtime);
                              runtime.cache_resized_images = event.target.checked;
                              next.runtime = runtime;
                            })
                          }
                        />
                        <span>Cache resized images in memory</span>
                      </label>
                    </div>
                  ) : null}
                </div>
              </section>

              <section className="experiment-right-panel">
                <div className="experiment-card" data-testid="experiment-card-runtime-logs">
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
                    <span>persistent_workers</span>
                    <strong>{asYesNo(runtimeInfo?.persistent_workers)}</strong>
                    <span>prefetch_factor</span>
                    <strong>{runtimeInfo?.prefetch_factor ?? "-"}</strong>
                    <span>cache_resized_images</span>
                    <strong>{asYesNo(runtimeInfo?.cache_resized_images)}</strong>
                    <span>max_cached_images</span>
                    <strong>{runtimeInfo?.max_cached_images ?? "-"}</strong>
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
                  <p className="experiment-log-cursor">
                    {logsAttempt ? `Run #${logsAttempt} • ` : ""}Cursor: {logsCursor}
                  </p>
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

                <div className="experiment-card" data-testid="experiment-card-onnx">
                  <h3>Exported Models</h3>
                  <div className="experiment-status-row">
                    <span className="status-pill">{onnxStatus}</span>
                    {onnxInfo?.attempt ? <span>Run #{onnxInfo.attempt}</span> : null}
                    {preferredVariantKey ? <span>Preferred: {preferredVariantKey}</span> : null}
                  </div>
                  <div className="experiment-onnx-grid">
                    <span>Selected variant</span>
                    <strong>{onnxInfo?.variant_key ?? preferredVariantKey ?? "-"}</strong>
                    <span>Input shape</span>
                    <strong>{onnxInputShape}</strong>
                    <span>Class order</span>
                    <strong title={onnxClassSummary}>{onnxClassSummary}</strong>
                    <span>Validation</span>
                    <strong>{onnxValidationSummary}</strong>
                  </div>
                  <div className="project-field">
                    <label htmlFor="experiment-variant-select">Export variant</label>
                    <select
                      id="experiment-variant-select"
                      value={selectedVariantKey ?? ""}
                      onChange={(event) => setSelectedVariantKey((event.target.value || null) as ModelVariantKey | null)}
                      disabled={availableVariantKeys.length < 1}
                    >
                      {availableVariantKeys.length < 1 ? <option value="">No variants available</option> : null}
                      {availableVariantKeys.map((variantKey) => (
                        <option key={variantKey} value={variantKey}>
                          {variantKey}
                          {preferredVariantKey === variantKey ? " (preferred)" : ""}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="experiment-logs-toolbar">
                    {onnxInfo?.model_onnx_url ? (
                      <a className="ghost-button" href={resolveAssetUri(onnxInfo.model_onnx_url)}>
                        Download ONNX
                      </a>
                    ) : (
                      <button type="button" className="ghost-button" disabled>
                        Download ONNX
                      </button>
                    )}
                    {onnxInfo?.metadata_url ? (
                      <a className="ghost-button" href={resolveAssetUri(onnxInfo.metadata_url)}>
                        Download Metadata
                      </a>
                    ) : (
                      <button type="button" className="ghost-button" disabled>
                        Download Metadata
                      </button>
                    )}
                    <button type="button" className="ghost-button" onClick={() => void loadOnnx()} disabled={isOnnxLoading}>
                      {isOnnxLoading ? "Refreshing..." : "Refresh ONNX"}
                    </button>
                    {status === "completed" && onnxInfo?.status === "exported" && onnxInfo.attempt ? (
                      <button
                        type="button"
                        className="primary-button"
                        onClick={() => void handleDeployModel()}
                        disabled={isDeploying}
                        data-testid="experiment-deploy-model-button"
                      >
                        {isDeploying ? "Deploying..." : "Deploy Model"}
                      </button>
                    ) : null}
                  </div>
                  {onnxError ? <p className="project-field-error">{onnxError}</p> : null}
                  {variantsError ? <p className="project-field-error">{variantsError}</p> : null}
                  {onnxInfo?.status === "failed" && onnxInfo.error ? <p className="project-field-error">{onnxInfo.error}</p> : null}
                  {!onnxInfo && !onnxError && status === "completed" ? (
                    <p className="experiment-log-cursor">ONNX export is not available yet.</p>
                  ) : null}
                </div>

                <div className="experiment-card" data-testid="experiment-card-variants">
                  <h3>Model Variants</h3>
                  <p className="experiment-log-cursor">
                    Compare FP32, FP16, PTQ INT8, and QAT INT8 on the {variantSplit.toUpperCase()} split, and switch latency/throughput views between {variantBenchmarkDevice.toUpperCase()} benchmarks.
                  </p>
                  <div className="experiment-logs-toolbar">
                    <button
                      type="button"
                      className={`ghost-button${variantSplit === "val" ? " active" : ""}`}
                      onClick={() => setVariantSplit("val")}
                    >
                      Val
                    </button>
                    <button
                      type="button"
                      className={`ghost-button${variantSplit === "test" ? " active" : ""}`}
                      onClick={() => setVariantSplit("test")}
                    >
                      Test
                    </button>
                    <button
                      type="button"
                      className={`ghost-button${variantBenchmarkDevice === "cpu" ? " active" : ""}`}
                      onClick={() => setVariantBenchmarkDevice("cpu")}
                    >
                      CPU Bench
                    </button>
                    <button
                      type="button"
                      className={`ghost-button${variantBenchmarkDevice === "cuda" ? " active" : ""}`}
                      onClick={() => setVariantBenchmarkDevice("cuda")}
                    >
                      CUDA Bench
                    </button>
                    <button type="button" className="ghost-button" onClick={() => void loadVariants()} disabled={isVariantsLoading}>
                      {isVariantsLoading ? "Refreshing..." : "Refresh variants"}
                    </button>
                  </div>

                  <div style={{ display: "grid", gap: 12 }}>
                    <div style={{ display: "grid", gap: 8, gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
                      <div className="project-field" style={{ marginBottom: 0 }}>
                        <span>FP16 Export</span>
                        <button
                          type="button"
                          className="primary-button"
                          disabled={!variantsInfo?.support.fp16_supported || isTriggeringFp16 || fp16Variant?.status === "running"}
                          onClick={() => void handleTriggerFp16()}
                        >
                          {isTriggeringFp16 || fp16Variant?.status === "running" ? "Running FP16..." : "Trigger FP16"}
                        </button>
                      </div>
                      <label className="project-field" style={{ marginBottom: 0 }}>
                        <span>PTQ Calibration Samples</span>
                        <input
                          type="number"
                          min="1"
                          step="1"
                          value={ptqCalibrationMaxSamples}
                          onChange={(event) => setPtqCalibrationMaxSamples(event.target.value)}
                          disabled={!variantsInfo?.support.ptq_supported || isTriggeringPtq}
                        />
                      </label>
                      <div className="project-field" style={{ marginBottom: 0 }}>
                        <span>PTQ to INT8</span>
                        <button
                          type="button"
                          className="primary-button"
                          disabled={!variantsInfo?.support.ptq_supported || isTriggeringPtq || ptqVariant?.status === "running"}
                          onClick={() => void handleTriggerPtq()}
                        >
                          {isTriggeringPtq || ptqVariant?.status === "running" ? "Running PTQ..." : "Trigger PTQ"}
                        </button>
                      </div>
                      <label className="project-field" style={{ marginBottom: 0 }}>
                        <span>QAT Epoch Override</span>
                        <input
                          type="number"
                          min="1"
                          step="1"
                          value={qatEpochsOverride}
                          onChange={(event) => setQatEpochsOverride(event.target.value)}
                          disabled={!variantsInfo?.support.qat_supported || isTriggeringQat}
                          placeholder="auto"
                        />
                      </label>
                      <label className="project-field" style={{ marginBottom: 0 }}>
                        <span>QAT LR Override</span>
                        <input
                          type="number"
                          min="0.0000001"
                          step="0.0001"
                          value={qatLearningRateOverride}
                          onChange={(event) => setQatLearningRateOverride(event.target.value)}
                          disabled={!variantsInfo?.support.qat_supported || isTriggeringQat}
                          placeholder="auto"
                        />
                      </label>
                      <label className="project-field" style={{ marginBottom: 0 }}>
                        <span>QAT Calibration Samples</span>
                        <input
                          type="number"
                          min="1"
                          step="1"
                          value={qatCalibrationMaxSamples}
                          onChange={(event) => setQatCalibrationMaxSamples(event.target.value)}
                          disabled={!variantsInfo?.support.qat_supported || isTriggeringQat}
                        />
                      </label>
                      <div className="project-field" style={{ marginBottom: 0 }}>
                        <span>QAT to INT8</span>
                        <button
                          type="button"
                          className="primary-button"
                          disabled={!variantsInfo?.support.qat_supported || isTriggeringQat || qatVariant?.status === "running"}
                          onClick={() => void handleTriggerQat()}
                        >
                          {isTriggeringQat || qatVariant?.status === "running" ? "Running QAT..." : "Trigger QAT"}
                        </button>
                      </div>
                    </div>
                    <span className="experiment-log-cursor">
                      Source checkpoint: {savedRecord?.artifacts_json?.selected_checkpoint_kind ? String(savedRecord.artifacts_json.selected_checkpoint_kind) : "best_metric"}
                    </span>
                  </div>

                  {!variantsInfo?.support.fp16_supported ? (
                    <p className="project-field-error">{variantsInfo?.support.fp16_reason ?? "FP16 is not supported for this task."}</p>
                  ) : null}
                  {!variantsInfo?.support.ptq_supported ? <p className="project-field-error">PTQ is not supported for this task.</p> : null}
                  {qatSupportSummary.state === "unsupported" ? (
                    <p className="project-field-error">{qatSupportSummary.message}</p>
                  ) : null}
                  {qatSupportSummary.state === "experimental" ? (
                    <p className="experiment-log-cursor">
                      {qatSupportSummary.message}
                      {qatSupportSummary.mode ? ` (${qatSupportSummary.mode})` : ""}
                    </p>
                  ) : null}
                  {fp16Variant?.error ? <p className="project-field-error">{fp16Variant.error}</p> : null}
                  {ptqVariant?.error ? <p className="project-field-error">{ptqVariant.error}</p> : null}
                  {qatVariant?.error ? <p className="project-field-error">{qatVariant.error}</p> : null}
                  {renderVariantComparisonTable([baselineVariant, fp16Variant, ptqVariant, qatVariant], "experiment-variant-table")}
                </div>

                <div className="experiment-card">
                  <h3>Danger Zone</h3>
                  <p className="labels-empty">Deleting this experiment removes all persisted run artifacts and cannot be undone.</p>
                  <button
                    type="button"
                    className="ghost-button danger-button"
                    disabled={isDeleting || status === "queued" || status === "running"}
                    onClick={() => void handleDelete()}
                  >
                    {isDeleting ? "Deleting..." : "Delete Experiment"}
                  </button>
                </div>

                <div className="experiment-card">
                  <h3>Metrics</h3>
                  <p className="experiment-log-cursor">
                    Last epoch time: {formatDurationSeconds(latestEpochSeconds)} | ETA: {formatDurationSeconds(latestEtaSeconds)} (finishes ~
                    {latestEtaClock})
                  </p>
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
            <section className="experiment-dashboard-section" data-testid="experiment-dashboard-section">
              <header className="project-section-header">
                <h3>Dashboard</h3>
                {evaluation?.attempt ? <span className="status-pill">Evaluation Run #{evaluation.attempt}</span> : null}
              </header>

              {!dashboardSupported ? (
                <div className="placeholder-card">
                  <p>Dashboard not supported yet for this task.</p>
                </div>
              ) : null}

              {dashboardSupported && isEvaluationLoading ? (
                <div className="placeholder-card">
                  <p>Loading evaluation dashboard...</p>
                </div>
              ) : null}

              {dashboardSupported && !isEvaluationLoading && evaluationError ? (
                <div className="placeholder-card">
                  <p>{evaluationError}</p>
                </div>
              ) : null}

              {dashboardSupported && !isEvaluationLoading && !evaluationError ? (
                <>
                  <div className="experiment-card">
                    <div className="experiment-analytics-header">
                      <h4>Metrics Trends</h4>
                      <div className="experiment-analytics-controls">
                        <div className="experiment-tab-row">
                          {dashboardTabs.map((tab) => (
                            <button
                              key={tab.key}
                              type="button"
                              className={`ghost-button ${dashboardChartTab === tab.key ? "active-toggle" : ""}`}
                              onClick={() => setDashboardChartTab(tab.key as DashboardChartTab)}
                            >
                              {tab.label}
                            </button>
                          ))}
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
                    <p className="experiment-log-cursor">
                      Last epoch time: {formatDurationSeconds(latestEpochSeconds)} | ETA: {formatDurationSeconds(latestEtaSeconds)} (finishes ~
                      {latestEtaClock})
                    </p>
                    {qatMetrics.length > 0 ? (
                      <p className="experiment-log-cursor">Dashed QAT lines use the same metric tabs and restart from QAT epoch 1.</p>
                    ) : null}
                    {dashboardAvailableSeries.length > 0 ? (
                      <div className="experiment-series-toggle-row">
                        {dashboardAvailableSeries.map((series) => {
                          const enabled = dashboardEnabledSeries[series.key] !== false;
                          return (
                            <label key={`dashboard-toggle-${series.key}`} className="model-builder-checkbox">
                              <input
                                type="checkbox"
                                checked={enabled}
                                onChange={(event) =>
                                  setDashboardEnabledSeries((current) => ({
                                    ...current,
                                    [series.key]: event.target.checked,
                                  }))
                                }
                              />
                              <span>{series.label}</span>
                            </label>
                          );
                        })}
                      </div>
                    ) : null}
                    <div className="experiment-chart-legend">
                      {dashboardAvailableSeries.map((series) => (
                        <span
                          key={series.key}
                          className={`experiment-legend-item${dashboardEnabledSeries[series.key] === false ? " is-muted" : ""}`}
                        >
                          <span className="experiment-legend-swatch" style={{ background: series.color }} />
                          <span>{series.label}</span>
                        </span>
                      ))}
                    </div>
                    <div className="experiment-chart-wrap">
                      {!dashboardHasVisibleSeries ? (
                        <p className="labels-empty">Enable at least one metric to render this chart.</p>
                      ) : !dashboardHasData ? (
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
                            const ratio = (tickEpoch - 1) / Math.max(1, dashboardEpochMax - 1);
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
                              <polyline
                                key={series.key}
                                fill="none"
                                stroke={series.color}
                                strokeWidth="2.1"
                                points={series.points}
                                strokeDasharray={series.strokeDasharray ?? undefined}
                              />
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
                            {dashboardAxisLabel}{dashboardLogScale ? " (log10)" : ""}
                          </text>
                        </svg>
                      )}
                    </div>
                  </div>

                  {isDetectionTask ? (
                    <>
                      <div className="experiment-card">
                        <div className="experiment-analytics-header">
                          <h4>Detection Summary</h4>
                        </div>
                        <div className="experiment-runtime-grid">
                          <span>Task</span>
                          <strong>{task}</strong>
                          <span>Best checkpoint</span>
                          <strong>{formatCheckpoint(checkpointIndex.best_metric)}</strong>
                          <span>Latest checkpoint</span>
                          <strong>{formatCheckpoint(checkpointIndex.latest)}</strong>
                          <span>Validation split</span>
                          <strong>{evaluation?.split ?? "val"}</strong>
                          <span>Classes</span>
                          <strong>{classNames.length > 0 ? classNames.join(", ") : "-"}</strong>
                          <span>Validation mAP@50</span>
                          <strong>{formatMetricValue(detectionOverallMap50)}</strong>
                          <span>Validation mAP@50:95</span>
                          <strong>{formatMetricValue(detectionOverallMap50_95)}</strong>
                          <span>Validation precision</span>
                          <strong>{formatMetricValue(detectionOverall?.precision)}</strong>
                          <span>Validation recall</span>
                          <strong>{formatMetricValue(detectionOverall?.recall)}</strong>
                          <span>Matched mean IoU</span>
                          <strong>{formatMetricValue(detectionOverall?.matched_mean_iou)}</strong>
                          <span>True positives</span>
                          <strong>{formatCountValue(detectionOverall?.tp)}</strong>
                          <span>False positives</span>
                          <strong>{formatCountValue(detectionOverall?.fp)}</strong>
                          <span>False negatives</span>
                          <strong>{formatCountValue(detectionOverall?.fn)}</strong>
                          <span>Duplicate FP</span>
                          <strong>{formatCountValue(detectionOverall?.duplicate_fp)}</strong>
                          <span>AP small</span>
                          <strong>{formatMetricValue(detectionOverall?.ap_small)}</strong>
                          <span>AP medium</span>
                          <strong>{formatMetricValue(detectionOverall?.ap_medium)}</strong>
                          <span>AP large</span>
                          <strong>{formatMetricValue(detectionOverall?.ap_large)}</strong>
                        </div>
                      </div>

                      <div className="experiment-card">
                        <div className="experiment-analytics-header">
                          <h4>Per-class Metrics</h4>
                          <label className="project-field">
                            <span>Sort</span>
                            <select value={perClassSort} onChange={(event) => setPerClassSort(event.target.value as PerClassSortKey)}>
                              <option value="ap50_desc">AP50 desc</option>
                              <option value="map_50_95_desc">mAP@50:95 desc</option>
                              <option value="ap75_desc">AP75 desc</option>
                              <option value="precision_desc">precision desc</option>
                              <option value="recall_desc">recall desc</option>
                              <option value="fp_desc">FP desc</option>
                              <option value="fn_desc">FN desc</option>
                              <option value="support_desc">support desc</option>
                            </select>
                          </label>
                        </div>
                        <div className="models-table-wrap">
                          <table className="models-table">
                            <thead>
                              <tr>
                                <th>Class</th>
                                <th>AP50</th>
                                <th>AP75</th>
                                <th>mAP@50:95</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>TP</th>
                                <th>FP</th>
                                <th>FN</th>
                                <th>Dup FP</th>
                                <th>Mean IoU</th>
                                <th>Support</th>
                              </tr>
                            </thead>
                            <tbody>
                              {sortedPerClassRows.map((row) => (
                                <tr key={`det-per-class-${row.class_index}`}>
                                  <td>{row.name}</td>
                                  <td>{formatMetricValue(row.ap50)}</td>
                                  <td>{formatMetricValue(row.ap75)}</td>
                                  <td>{formatMetricValue(row.map_50_95)}</td>
                                  <td>{formatMetricValue(row.precision)}</td>
                                  <td>{formatMetricValue(row.recall)}</td>
                                  <td>{formatCountValue(row.tp)}</td>
                                  <td>{formatCountValue(row.fp)}</td>
                                  <td>{formatCountValue(row.fn)}</td>
                                  <td>{formatCountValue(row.duplicate_fp)}</td>
                                  <td>{formatMetricValue(row.matched_mean_iou)}</td>
                                  <td>{formatCountValue(row.support)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>

                      {detectionSizeBucketRows.length > 0 ? (
                        <div className="experiment-card">
                          <div className="experiment-analytics-header">
                            <h4>Size Buckets</h4>
                          </div>
                          <div className="models-table-wrap">
                            <table className="models-table">
                              <thead>
                                <tr>
                                  <th>Bucket</th>
                                  <th>GT</th>
                                  <th>Predictions</th>
                                  <th>AP50</th>
                                  <th>mAP@50:95</th>
                                  <th>Precision</th>
                                  <th>Recall</th>
                                </tr>
                              </thead>
                              <tbody>
                                {detectionSizeBucketRows.map((row) => (
                                  <tr key={`size-bucket-${row.name}`}>
                                    <td>{row.name}</td>
                                    <td>{formatCountValue(row.groundTruthCount)}</td>
                                    <td>{formatCountValue(row.predictionCount)}</td>
                                    <td>{formatMetricValue(row.ap50)}</td>
                                    <td>{formatMetricValue(row.map50_95)}</td>
                                    <td>{formatMetricValue(row.precision)}</td>
                                    <td>{formatMetricValue(row.recall)}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      ) : null}
                    </>
                  ) : null}

                  {isClassificationTask && evaluation ? (
                  <>
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
                          onChange={(event) => setPerClassSort(event.target.value as PerClassSortKey)}
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
            <footer className="model-builder-footer experiment-actions-row" data-testid="experiment-actions-footer">
              <button
                type="button"
                className="ghost-button"
                disabled={!isEditable || !isDirty || !validation.isValid || isSaving || isDeleting || !draftConfig}
                onClick={() => void handleSave()}
              >
                {isSaving ? "Saving..." : "Save"}
              </button>
              {status === "running" || status === "queued" ? (
                <button type="button" className="ghost-button" disabled={isCanceling || isDeleting} onClick={() => void handleCancel()}>
                  {isCanceling ? "Canceling..." : status === "queued" ? "Cancel Queue" : "Cancel"}
                </button>
              ) : (
                <button
                  type="button"
                  className="primary-button"
                  disabled={!isEditable || isStarting || isDeleting || !validation.isValid || !draftConfig}
                  onClick={handleStartClick}
                  data-testid="experiment-start-button"
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

      {showStartChoiceModal ? (
        <div className="project-modal-backdrop" role="presentation">
          <div className="project-modal" role="dialog" aria-modal="true" aria-label="Choose start action">
            <h3>Unsaved Experiment Changes</h3>
            <p className="import-selection-summary">
              Training now will use the last saved experiment config unless you save first.
            </p>
            <div className="project-modal-actions">
              <button type="button" className="ghost-button" onClick={() => setShowStartChoiceModal(false)}>
                Close
              </button>
              <button
                type="button"
                className="ghost-button"
                disabled={isStarting || isSaving}
                onClick={() => {
                  setShowStartChoiceModal(false);
                  void handleStart();
                }}
              >
                Start saved version
              </button>
              <button
                type="button"
                className="primary-button"
                disabled={isStarting || isSaving}
                onClick={() => void handleSaveAndStart()}
              >
                {isSaving ? "Saving..." : isStarting ? "Starting..." : "Save and run"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
