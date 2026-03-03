"use client";

import Link from "next/link";
import { useMemo, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import {
  ApiError,
  getExperimentAnalytics,
  type ExperimentAnalyticsItem,
} from "../../../../lib/api";
import {
  bestRunByMetric,
  buildAnalyticsSummary,
  defaultSelectedRunIds,
  filterAnalyticsItems,
  scatterPoints,
  seriesPoints,
} from "../../../../lib/workspace/experimentAnalytics";
import {
  buildTicks,
  computeSeriesDomain,
  formatTick,
  isBoundedMetricKey,
} from "../../../../lib/workspace/experimentMetrics";
import { runtimeBadgeLabel } from "../../../../lib/workspace/experimentRuntime";

interface ExperimentsPageProps {
  params: {
    projectId: string;
  };
}

function formatDateTime(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "-";
  return parsed.toLocaleString();
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

function chartYValue(value: number, useLog: boolean): number {
  if (!useLog) return value;
  return Math.log10(Math.max(1e-9, value));
}

export default function ExperimentsPage({ params }: ExperimentsPageProps) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);
  const queryModelId = searchParams.get("modelId") ?? "";

  const [items, setItems] = useState<ExperimentAnalyticsItem[]>([]);
  const [availableSeries, setAvailableSeries] = useState<string[]>([]);
  const [selectedModelId, setSelectedModelId] = useState(queryModelId);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedMetric, setSelectedMetric] = useState("val_accuracy");
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [showFailed, setShowFailed] = useState(false);
  const [useLogScale, setUseLogScale] = useState(false);
  const [xParam, setXParam] = useState("learning_rate");
  const [yParam, setYParam] = useState("best_val_accuracy");
  const [hoverEpoch, setHoverEpoch] = useState<number | null>(null);
  const [hoveredScatterPointId, setHoveredScatterPointId] = useState<string | null>(null);

  useEffect(() => {
    setSelectedModelId(queryModelId);
  }, [queryModelId]);

  useEffect(() => {
    let isMounted = true;
    async function loadPageData() {
      setIsLoading(true);
      setErrorMessage(null);
      try {
        const response = await getExperimentAnalytics(projectId, { maxPoints: 200 });
        if (!isMounted) return;
        setItems(response.items ?? []);
        setAvailableSeries(response.available_series ?? []);
        if (response.available_series?.length) {
          if (!response.available_series.includes(selectedMetric)) {
            if (response.available_series.includes("val_accuracy")) setSelectedMetric("val_accuracy");
            else setSelectedMetric(response.available_series[0]);
          }
        }
      } catch (error) {
        if (!isMounted) return;
        setErrorMessage(parseApiErrorMessage(error, "Failed to load experiments analytics"));
        setItems([]);
        setAvailableSeries([]);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }

    void loadPageData();
    return () => {
      isMounted = false;
    };
  }, [projectId, selectedMetric]);

  useEffect(() => {
    if (selectedRunIds.length > 0 || items.length === 0) return;
    setSelectedRunIds(defaultSelectedRunIds(items, 3));
  }, [items, selectedRunIds.length]);

  const visibleItems = useMemo(
    () => filterAnalyticsItems(items, { modelId: selectedModelId, showFailed }),
    [items, selectedModelId, showFailed],
  );
  const selectedItems = useMemo(
    () => visibleItems.filter((item) => selectedRunIds.includes(item.experiment_id)),
    [selectedRunIds, visibleItems],
  );
  const summary = useMemo(() => buildAnalyticsSummary(visibleItems), [visibleItems]);
  const bestRunId = useMemo(
    () => bestRunByMetric(visibleItems, selectedMetric),
    [selectedMetric, visibleItems],
  );

  const modelOptions = useMemo(() => {
    const unique = new Map<string, string>();
    for (const item of items) {
      if (!item?.model_id) continue;
      if (!unique.has(item.model_id)) unique.set(item.model_id, item.model_name || item.model_id);
    }
    return Array.from(unique.entries()).map(([id, name]) => ({ id, name })).sort((a, b) => a.name.localeCompare(b.name));
  }, [items]);

  const chartRuns = useMemo(
    () =>
      selectedItems.map((item, index) => ({
        ...item,
        color: ["#2f6fca", "#cc6f36", "#2f9d58", "#9a4dcb", "#c64b65", "#1f7f8e"][index % 6],
        points: seriesPoints(item, selectedMetric),
      })),
    [selectedItems, selectedMetric],
  );

  const chartEpochMax = useMemo(() => {
    const epochs = chartRuns.flatMap((run) => run.points.map((point) => point.epoch));
    if (epochs.length === 0) return 1;
    return Math.max(...epochs);
  }, [chartRuns]);
  const chartMetricIsBounded = useMemo(() => isBoundedMetricKey(selectedMetric), [selectedMetric]);
  const chartDomain = useMemo(() => {
    const values = chartRuns.flatMap((run) => run.points.map((point) => point.value));
    return computeSeriesDomain(values, {
      useLog: useLogScale,
      clamp01: chartMetricIsBounded && !useLogScale,
    });
  }, [chartMetricIsBounded, chartRuns, useLogScale]);
  const [chartMin, chartMax] = [chartDomain.min, chartDomain.max];
  const chartYTicks = useMemo(
    () =>
      buildTicks(chartDomain, {
        useLog: useLogScale,
        count: 5,
        clamp01: chartMetricIsBounded && !useLogScale,
      }),
    [chartDomain, chartMetricIsBounded, useLogScale],
  );
  const chartXTicks = useMemo(
    () => buildTicks({ min: 1, max: chartEpochMax }, { count: 5 }).map((tick) => Math.max(1, Math.round(tick))),
    [chartEpochMax],
  );

  const tooltipRows = useMemo(() => {
    if (hoverEpoch == null) return [];
    return chartRuns
      .map((run) => {
        let bestPoint = null;
        let bestDistance = Number.POSITIVE_INFINITY;
        for (const point of run.points) {
          const distance = Math.abs(point.epoch - hoverEpoch);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestPoint = point;
          }
        }
        if (!bestPoint) return null;
        return {
          id: run.experiment_id,
          name: run.name,
          color: run.color,
          epoch: bestPoint.epoch,
          value: bestPoint.value,
        };
      })
      .filter((row) => row != null);
  }, [chartRuns, hoverEpoch]);

  const scatter = useMemo(() => scatterPoints(visibleItems, xParam, yParam), [visibleItems, xParam, yParam]);
  const scatterYIsBounded = yParam === "best_val_accuracy" || yParam === "final_val_accuracy";
  const scatterXDomain = useMemo(() => {
    if (xParam === "augmentation") return { min: 0, max: 3 };
    return computeSeriesDomain(scatter.map((row) => row.x), { useLog: false, clamp01: false });
  }, [scatter, xParam]);
  const scatterYDomain = useMemo(
    () =>
      computeSeriesDomain(scatter.map((row) => row.y), {
        useLog: false,
        clamp01: scatterYIsBounded,
      }),
    [scatter, scatterYIsBounded],
  );
  const scatterXTicks = useMemo(() => {
    if (xParam === "augmentation") return [0, 1, 2, 3];
    return buildTicks(scatterXDomain, { count: 5 });
  }, [scatterXDomain, xParam]);
  const scatterYTicks = useMemo(
    () => buildTicks(scatterYDomain, { count: 5, clamp01: scatterYIsBounded }),
    [scatterYDomain, scatterYIsBounded],
  );
  const hoveredScatterPoint = useMemo(
    () => scatter.find((row) => row.experimentId === hoveredScatterPointId) ?? null,
    [hoveredScatterPointId, scatter],
  );

  const createHref = selectedModelId
    ? `/projects/${encodeURIComponent(projectId)}/experiments/new?modelId=${encodeURIComponent(selectedModelId)}`
    : `/projects/${encodeURIComponent(projectId)}/experiments/new`;

  return (
    <main className="workspace-shell project-page-shell">
      <section className="workspace-frame project-content-frame placeholder-page">
        <header className="project-section-header">
          <h2>Experiments</h2>
          <Link className="primary-button" href={createHref}>
            + New Experiment
          </Link>
        </header>

        <div className="experiments-list-toolbar experiments-analytics-toolbar">
          <label className="project-field">
            <span>Filter by model</span>
            <select value={selectedModelId} onChange={(event) => setSelectedModelId(event.target.value)}>
              <option value="">All models</option>
              {modelOptions.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </label>
          <label className="model-builder-checkbox">
            <input type="checkbox" checked={showFailed} onChange={(event) => setShowFailed(event.target.checked)} />
            <span>Show failed</span>
          </label>
        </div>

        {errorMessage ? <p className="project-field-error">{errorMessage}</p> : null}
        {isLoading ? (
          <div className="placeholder-card">
            <p>Loading experiments...</p>
          </div>
        ) : null}

        {!isLoading ? (
          <section className="experiment-analytics-section">
            <h3>Analytics</h3>
            <div className="experiment-analytics-summary">
              <article className="placeholder-card">
                <h4>Best Accuracy</h4>
                <p>{summary.bestAccuracy == null ? "-" : summary.bestAccuracy.toFixed(4)}</p>
              </article>
              <article className="placeholder-card">
                <h4>Lowest Val Loss</h4>
                <p>{summary.lowestValLoss == null ? "-" : summary.lowestValLoss.toFixed(4)}</p>
              </article>
              <article className="placeholder-card">
                <h4>Total Runs</h4>
                <p>{summary.totalRuns}</p>
              </article>
              <article className="placeholder-card">
                <h4>Failures</h4>
                <p>{summary.failures}</p>
              </article>
            </div>

            <div className="experiment-card">
              <div className="experiment-analytics-header">
                <h4>Multi-run Comparison</h4>
                <div className="experiment-analytics-controls">
                  <label className="project-field">
                    <span>Metric</span>
                    <select value={selectedMetric} onChange={(event) => setSelectedMetric(event.target.value)}>
                      {availableSeries.map((seriesName) => (
                        <option key={seriesName} value={seriesName}>
                          {seriesName}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="model-builder-checkbox">
                    <input type="checkbox" checked={useLogScale} onChange={(event) => setUseLogScale(event.target.checked)} />
                    <span>Log scale</span>
                  </label>
                </div>
              </div>

              <div className="experiment-run-selector-grid">
                {visibleItems.map((item) => (
                  <label key={item.experiment_id} className="model-builder-checkbox">
                    <input
                      type="checkbox"
                      checked={selectedRunIds.includes(item.experiment_id)}
                      onChange={(event) =>
                        setSelectedRunIds((current) => {
                          if (event.target.checked) return Array.from(new Set([...current, item.experiment_id]));
                          return current.filter((id) => id !== item.experiment_id);
                        })
                      }
                    />
                    <span>
                      {item.name}
                      {bestRunId === item.experiment_id ? " (best)" : ""}
                    </span>
                  </label>
                ))}
              </div>

              <div className="experiment-chart-wrap">
                {chartRuns.length === 0 ? (
                  <p className="labels-empty">Select runs with available metrics to compare.</p>
                ) : (
                  <svg
                    viewBox="0 0 820 300"
                    role="img"
                    aria-label="Multi-run metrics chart"
                    onMouseMove={(event) => {
                      const rect = event.currentTarget.getBoundingClientRect();
                      const x = ((event.clientX - rect.left) / Math.max(1, rect.width)) * 820;
                      const clamped = Math.max(50, Math.min(790, x));
                      const epoch = 1 + (((clamped - 50) / 740) * Math.max(1, chartEpochMax - 1));
                      setHoverEpoch(Math.max(1, Math.round(epoch)));
                    }}
                    onMouseLeave={() => setHoverEpoch(null)}
                  >
                    <rect x="0" y="0" width="820" height="300" fill="#f8fbff" stroke="#d0d9e7" />
                    <line x1="50" y1="24" x2="50" y2="260" stroke="#9db0ca" />
                    <line x1="50" y1="260" x2="790" y2="260" stroke="#9db0ca" />
                    {chartYTicks.map((tick) => {
                      const y = 24 + (((chartMax - tick) / Math.max(1e-9, chartMax - chartMin)) * 236);
                      return (
                        <g key={`y-${tick.toFixed(8)}`}>
                          <line x1="50" y1={y} x2="790" y2={y} stroke="#e1e8f2" strokeWidth="1" />
                          <text className="axis-tick" x="44" y={y + 4} textAnchor="end">
                            {formatTick(tick, { useLog: useLogScale, bounded: chartMetricIsBounded && !useLogScale })}
                          </text>
                        </g>
                      );
                    })}
                    {Array.from(new Set(chartXTicks)).map((tickEpoch) => {
                      const x = 50 + (((tickEpoch - 1) / Math.max(1, chartEpochMax - 1)) * 740);
                      return (
                        <g key={`x-${tickEpoch}`}>
                          <line x1={x} y1="24" x2={x} y2="260" stroke="#edf1f7" strokeWidth="1" />
                          <text className="axis-tick" x={x} y="276" textAnchor="middle">
                            {tickEpoch}
                          </text>
                        </g>
                      );
                    })}
                    {chartRuns.map((run) => {
                      const points = run.points
                        .map((point) => {
                          const x = 50 + (((point.epoch - 1) / Math.max(1, chartEpochMax - 1)) * 740);
                          const yValue = chartYValue(point.value, useLogScale);
                          const y = 24 + (((chartMax - yValue) / Math.max(1e-9, chartMax - chartMin)) * 236);
                          return `${x.toFixed(2)},${y.toFixed(2)}`;
                        })
                        .join(" ");
                      return <polyline key={run.experiment_id} fill="none" stroke={run.color} strokeWidth="2.1" points={points} />;
                    })}
                    {hoverEpoch != null ? (
                      <line
                        x1={50 + (((hoverEpoch - 1) / Math.max(1, chartEpochMax - 1)) * 740)}
                        y1="24"
                        x2={50 + (((hoverEpoch - 1) / Math.max(1, chartEpochMax - 1)) * 740)}
                        y2="260"
                        stroke="#8da2c1"
                        strokeDasharray="4 3"
                      />
                    ) : null}
                    {tooltipRows.length > 0 ? (
                      <g>
                        <rect x="560" y="32" width="248" height={34 + tooltipRows.length * 15} fill="#f6f9ff" stroke="#bdcbe0" rx="8" />
                        <text x="572" y="50" fontSize="12" fill="#304765" fontWeight="700">
                          Epoch {hoverEpoch}
                        </text>
                        {tooltipRows.map((row, index) => (
                          <text key={row.id} x="572" y={66 + index * 15} fontSize="12" fill={row.color}>
                            {row.name}: {row.value.toFixed(4)}
                          </text>
                        ))}
                      </g>
                    ) : null}
                    <text className="axis-label" x="420" y="294" textAnchor="middle">
                      Epoch
                    </text>
                    <text className="axis-label" x="16" y="142" transform="rotate(-90 16 142)" textAnchor="middle">
                      {selectedMetric}{useLogScale ? " (log10)" : ""}
                    </text>
                  </svg>
                )}
              </div>
            </div>

            <div className="experiment-card">
              <div className="experiment-analytics-header">
                <h4>Hyperparameter Scatter</h4>
                <div className="experiment-analytics-controls">
                  <label className="project-field">
                    <span>X</span>
                    <select value={xParam} onChange={(event) => setXParam(event.target.value)}>
                      <option value="learning_rate">learning_rate</option>
                      <option value="batch_size">batch_size</option>
                      <option value="augmentation">augmentation</option>
                      <option value="epochs">epochs</option>
                    </select>
                  </label>
                  <label className="project-field">
                    <span>Y</span>
                    <select value={yParam} onChange={(event) => setYParam(event.target.value)}>
                      <option value="best_val_accuracy">best_val_accuracy</option>
                      <option value="best_val_loss">best_val_loss</option>
                      <option value="final_val_accuracy">final_val_accuracy</option>
                    </select>
                  </label>
                </div>
              </div>
              <div className="experiment-chart-wrap">
                {scatter.length === 0 ? (
                  <p className="labels-empty">No runs with required values for this scatter view.</p>
                ) : (
                  <svg
                    viewBox="0 0 820 280"
                    role="img"
                    aria-label="Hyperparameter scatter chart"
                    onMouseLeave={() => setHoveredScatterPointId(null)}
                  >
                    <rect x="0" y="0" width="820" height="280" fill="#f8fbff" stroke="#d0d9e7" />
                    <line x1="50" y1="24" x2="50" y2="240" stroke="#9db0ca" />
                    <line x1="50" y1="240" x2="790" y2="240" stroke="#9db0ca" />
                    {scatterYTicks.map((tick) => {
                      const y = 24 + (((scatterYDomain.max - tick) / Math.max(1e-9, scatterYDomain.max - scatterYDomain.min)) * 216);
                      return (
                        <g key={`scatter-y-${tick.toFixed(8)}`}>
                          <line x1="50" y1={y} x2="790" y2={y} stroke="#e1e8f2" strokeWidth="1" />
                          <text className="axis-tick" x="44" y={y + 4} textAnchor="end">
                            {formatTick(tick, { bounded: scatterYIsBounded })}
                          </text>
                        </g>
                      );
                    })}
                    {scatterXTicks.map((tick) => {
                      const x = 50 + (((tick - scatterXDomain.min) / Math.max(1e-9, scatterXDomain.max - scatterXDomain.min)) * 740);
                      const label =
                        xParam === "augmentation"
                          ? (["none", "light", "medium", "heavy"][Math.max(0, Math.min(3, Math.round(tick)))] ?? String(tick))
                          : formatTick(tick);
                      return (
                        <g key={`scatter-x-${tick.toFixed(8)}`}>
                          <line x1={x} y1="24" x2={x} y2="240" stroke="#edf1f7" strokeWidth="1" />
                          <text className="axis-tick" x={x} y="256" textAnchor="middle">
                            {label}
                          </text>
                        </g>
                      );
                    })}
                    {scatter.map((point) => {
                      const x = 50 + (((point.x - scatterXDomain.min) / Math.max(1e-9, scatterXDomain.max - scatterXDomain.min)) * 740);
                      const y = 24 + (((scatterYDomain.max - point.y) / Math.max(1e-9, scatterYDomain.max - scatterYDomain.min)) * 216);
                      const isHovered = hoveredScatterPointId === point.experimentId;
                      return (
                        <g key={point.experimentId}>
                          <circle
                            cx={x}
                            cy={y}
                            r={isHovered ? "6.5" : "5"}
                            fill={point.status === "failed" ? "#9ea9ba" : "#2f6fca"}
                            stroke={isHovered ? "#1f4f95" : "none"}
                            strokeWidth={isHovered ? "1.5" : "0"}
                            onMouseEnter={() => setHoveredScatterPointId(point.experimentId)}
                            onMouseLeave={() => setHoveredScatterPointId((current) => (current === point.experimentId ? null : current))}
                            onClick={() =>
                              router.push(
                                `/projects/${encodeURIComponent(projectId)}/experiments/${encodeURIComponent(point.experimentId)}`,
                              )
                            }
                            style={{ cursor: "pointer" }}
                          />
                          <title>{point.name}</title>
                        </g>
                      );
                    })}
                    {hoveredScatterPoint ? (() => {
                      const hx = 50 + (((hoveredScatterPoint.x - scatterXDomain.min) / Math.max(1e-9, scatterXDomain.max - scatterXDomain.min)) * 740);
                      const hy = 24 + (((scatterYDomain.max - hoveredScatterPoint.y) / Math.max(1e-9, scatterYDomain.max - scatterYDomain.min)) * 216);
                      const tooltipWidth = 228;
                      const tooltipHeight = 56;
                      const tooltipX = Math.min(790 - tooltipWidth, Math.max(54, hx + 10));
                      const tooltipY = Math.min(236 - tooltipHeight, Math.max(28, hy - tooltipHeight - 8));
                      return (
                        <g>
                          <rect x={tooltipX} y={tooltipY} width={tooltipWidth} height={tooltipHeight} rx="8" fill="#f6f9ff" stroke="#bdcbe0" />
                          <text x={tooltipX + 10} y={tooltipY + 18} fontSize="12" fill="#304765" fontWeight="700">
                            {hoveredScatterPoint.name}
                          </text>
                          <text x={tooltipX + 10} y={tooltipY + 34} fontSize="11" fill="#3d5779">
                            {xParam}: {formatTick(hoveredScatterPoint.x)}
                          </text>
                          <text x={tooltipX + 10} y={tooltipY + 48} fontSize="11" fill="#3d5779">
                            {yParam}: {formatTick(hoveredScatterPoint.y, { bounded: scatterYIsBounded })}
                          </text>
                        </g>
                      );
                    })() : null}
                    <text className="axis-label" x="420" y="274" textAnchor="middle">
                      {xParam}
                    </text>
                    <text className="axis-label" x="16" y="132" transform="rotate(-90 16 132)" textAnchor="middle">
                      {yParam}
                    </text>
                  </svg>
                )}
              </div>
            </div>
          </section>
        ) : null}

        {!isLoading && visibleItems.length === 0 ? (
          <div className="placeholder-card">
            <h3>No experiments yet</h3>
            <p>Create an experiment draft and start training to stream metrics and checkpoints.</p>
          </div>
        ) : null}

        {!isLoading && visibleItems.length > 0 ? (
          <div className="models-table-wrap">
            <table className="models-table experiments-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Model</th>
                  <th>Status</th>
                  <th>Best Metric</th>
                  <th>Updated</th>
                </tr>
              </thead>
              <tbody>
                {visibleItems.map((experiment) => {
                  const metricName = experiment.best?.metric_name ?? null;
                  const metricValue = typeof experiment.best?.metric_value === "number" ? experiment.best.metric_value : null;
                  const runtimeBadge = runtimeBadgeLabel(experiment.runtime ?? null);
                  const bestLabel =
                    metricName && metricValue != null
                      ? `${metricName}: ${metricValue.toFixed(4)}${experiment.best?.epoch ? ` (ep ${experiment.best.epoch})` : ""}`
                      : "-";
                  return (
                    <tr key={experiment.experiment_id}>
                      <td>
                        <Link
                          href={`/projects/${encodeURIComponent(projectId)}/experiments/${encodeURIComponent(experiment.experiment_id)}`}
                        >
                          {experiment.name}
                        </Link>
                      </td>
                      <td>{experiment.model_name}</td>
                      <td>
                        <span className={`status-pill status-${experiment.status}`}>{experiment.status}</span>
                        {runtimeBadge ? <span className="runtime-pill">{runtimeBadge}</span> : null}
                      </td>
                      <td>{bestLabel}</td>
                      <td>{formatDateTime(experiment.updated_at)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>
    </main>
  );
}
