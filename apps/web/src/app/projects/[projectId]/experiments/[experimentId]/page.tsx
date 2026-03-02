"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useProjectNavigationGuard } from "../../../../../components/workspace/ProjectNavigationContext";
import {
  ApiError,
  cancelExperiment,
  getExperiment,
  listProjectModels,
  startExperiment,
  streamExperimentEvents,
  updateExperiment,
  type ExperimentCheckpoint,
  type ExperimentMetricPoint,
  type ExperimentStatus,
  type ProjectExperimentRecord,
} from "../../../../../lib/api";
import {
  buildLinePoints,
  indexCheckpointsByKind,
  mergeMetricPoints,
  metricDomain,
  metricKeyForTask,
} from "../../../../../lib/workspace/experimentMetrics";

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
  if (key === "val_loss") return asMetricValue(row.val_loss);
  if (key === "val_accuracy") return asMetricValue(row.val_accuracy);
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
  const eventCursorRef = useRef(0);

  const isEditable = status === "draft" || status === "failed" || status === "canceled";
  const task = (typeof draftConfig?.task === "string" ? draftConfig.task : "classification") as string;
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
  const chartYDomain = useMemo(() => metricDomain(metrics, chartKeys), [chartKeys, metrics]);
  const yTickValues = useMemo(() => {
    const ticks: number[] = [];
    const stepCount = 4;
    const range = Math.max(1e-9, chartYDomain.max - chartYDomain.min);
    for (let index = 0; index <= stepCount; index += 1) {
      const ratio = index / stepCount;
      ticks.push(chartYDomain.max - (range * ratio));
    }
    return ticks;
  }, [chartYDomain.max, chartYDomain.min]);
  const xTickValues = useMemo(() => {
    const ticks: number[] = [];
    const stepCount = 4;
    for (let index = 0; index <= stepCount; index += 1) {
      const value = 1 + ((chartMaxEpoch - 1) * (index / stepCount));
      ticks.push(Math.max(1, Math.round(value)));
    }
    return Array.from(new Set(ticks));
  }, [chartMaxEpoch]);

  const seriesLegend = useMemo(
    () => [
      { key: primaryMetricKey, label: primaryMetricLabel, color: primaryColor, enabled: showPrimary },
      { key: "val_loss", label: "val loss", color: lossColor, enabled: showValLoss },
    ],
    [primaryMetricKey, primaryMetricLabel, showPrimary, showValLoss],
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
    const range = Math.max(1e-9, chartYDomain.max - chartYDomain.min);
    return hoveredSeriesValues.map((row) => {
      const y = chartPadding + (((chartYDomain.max - row.value) / range) * chartInnerHeight);
      return { ...row, y };
    });
  }, [chartInnerHeight, chartPadding, chartYDomain.max, chartYDomain.min, hoveredSeriesValues]);

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
          })
        : "",
    [chartHeight, chartKeys, chartPadding, chartWidth, metrics, primaryMetricKey, showPrimary],
  );
  const valLossLinePoints = useMemo(
    () =>
      showValLoss
        ? buildLinePoints(metrics, "val_loss", {
            width: chartWidth,
            height: chartHeight,
            padding: chartPadding,
            seriesKeys: chartKeys,
          })
        : "",
    [chartHeight, chartKeys, chartPadding, chartWidth, metrics, showValLoss],
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
  const backToModelHref = modelId
    ? `/projects/${encodeURIComponent(projectId)}/models/${encodeURIComponent(modelId)}`
    : `/projects/${encodeURIComponent(projectId)}/models`;

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

  useEffect(() => {
    void loadDetail();
  }, [loadDetail]);

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
    if (status !== "running" && status !== "queued") return;
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
  }, [activeAttempt, experimentId, loadDetail, projectId, status]);

  function patchConfig(mutator: (next: Record<string, unknown>) => void) {
    setDraftConfig((current) => {
      if (!current) return current;
      const next = cloneConfig(current);
      mutator(next);
      return next;
    });
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
            <Link href={backToModelHref} className="ghost-button">
              Back to Model
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
                        <span>{series.label}</span>
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
                        {yTickValues.map((tickValue) => {
                          const range = Math.max(1e-9, chartYDomain.max - chartYDomain.min);
                          const ratio = (chartYDomain.max - tickValue) / range;
                          const y = chartPadding + (ratio * chartInnerHeight);
                          return (
                            <g key={`y:${tickValue.toFixed(6)}`}>
                              <line x1={chartPadding} y1={y} x2={chartPadding + chartInnerWidth} y2={y} stroke="#e1e8f2" strokeWidth="1" />
                              <text x={chartPadding - 6} y={y + 4} textAnchor="end" fontSize="11" fill="#4f6482">
                                {tickValue.toFixed(2)}
                              </text>
                            </g>
                          );
                        })}
                        {xTickValues.map((tickEpoch) => {
                          const ratio = (tickEpoch - 1) / Math.max(1, chartMaxEpoch - 1);
                          const x = chartPadding + (ratio * chartInnerWidth);
                          return (
                            <g key={`x:${tickEpoch}`}>
                              <line x1={x} y1={chartPadding} x2={x} y2={chartPadding + chartInnerHeight} stroke="#edf1f7" strokeWidth="1" />
                              <text x={x} y={chartPadding + chartInnerHeight + 16} textAnchor="middle" fontSize="11" fill="#4f6482">
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
                        <text x={chartPadding + (chartInnerWidth / 2)} y={chartHeight - 4} textAnchor="middle" fontSize="12" fill="#334a6a">
                          epoch
                        </text>
                        <text
                          x="16"
                          y={chartPadding + (chartInnerHeight / 2)}
                          transform={`rotate(-90 16 ${chartPadding + (chartInnerHeight / 2)})`}
                          textAnchor="middle"
                          fontSize="12"
                          fill="#334a6a"
                        >
                          metric value
                        </text>
                      </svg>
                    )}
                  </div>
                </div>
              </section>
            </div>
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
