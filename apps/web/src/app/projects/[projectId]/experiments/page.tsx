"use client";

import Link from "next/link";
import { useMemo, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";

import {
  ApiError,
  listExperiments,
  listProjectModels,
  type ProjectExperimentSummary,
  type ProjectModelSummary,
} from "../../../../lib/api";

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

function renderBestMetric(experiment: ProjectExperimentSummary): string {
  const summary = experiment.summary_json;
  if (!summary.best_metric_name || typeof summary.best_metric_value !== "number") return "-";
  const value = summary.best_metric_value.toFixed(4);
  if (typeof summary.best_epoch === "number") {
    return `${summary.best_metric_name}: ${value} (ep ${summary.best_epoch})`;
  }
  return `${summary.best_metric_name}: ${value}`;
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

export default function ExperimentsPage({ params }: ExperimentsPageProps) {
  const searchParams = useSearchParams();
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);
  const queryModelId = searchParams.get("modelId") ?? "";

  const [models, setModels] = useState<ProjectModelSummary[]>([]);
  const [experiments, setExperiments] = useState<ProjectExperimentSummary[]>([]);
  const [selectedModelId, setSelectedModelId] = useState(queryModelId);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    setSelectedModelId(queryModelId);
  }, [queryModelId]);

  useEffect(() => {
    let isMounted = true;
    async function loadPageData() {
      setIsLoading(true);
      setErrorMessage(null);
      try {
        const [modelRows, experimentRows] = await Promise.all([
          listProjectModels(projectId),
          listExperiments(projectId, { modelId: selectedModelId || undefined }),
        ]);
        if (!isMounted) return;
        setModels(modelRows);
        setExperiments(experimentRows.items);
      } catch (error) {
        if (!isMounted) return;
        setErrorMessage(parseApiErrorMessage(error, "Failed to load experiments"));
        setModels([]);
        setExperiments([]);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }

    void loadPageData();
    return () => {
      isMounted = false;
    };
  }, [projectId, selectedModelId]);

  const modelsById = useMemo(() => new Map(models.map((model) => [model.id, model])), [models]);
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

        <div className="experiments-list-toolbar">
          <label className="project-field">
            <span>Filter by model</span>
            <select value={selectedModelId} onChange={(event) => setSelectedModelId(event.target.value)}>
              <option value="">All models</option>
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </label>
        </div>

        {errorMessage ? <p className="project-field-error">{errorMessage}</p> : null}

        {isLoading ? (
          <div className="placeholder-card">
            <p>Loading experiments...</p>
          </div>
        ) : null}

        {!isLoading && experiments.length === 0 ? (
          <div className="placeholder-card">
            <h3>No experiments yet</h3>
            <p>Create an experiment draft and start training to stream metrics and checkpoints.</p>
          </div>
        ) : null}

        {!isLoading && experiments.length > 0 ? (
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
                {experiments.map((experiment) => {
                  const modelName = modelsById.get(experiment.model_id)?.name ?? experiment.model_id;
                  return (
                    <tr key={experiment.id}>
                      <td>
                        <Link href={`/projects/${encodeURIComponent(projectId)}/experiments/${encodeURIComponent(experiment.id)}`}>
                          {experiment.name}
                        </Link>
                      </td>
                      <td>{modelName}</td>
                      <td>
                        <span className={`status-pill status-${experiment.status}`}>{experiment.status}</span>
                      </td>
                      <td>{renderBestMetric(experiment)}</td>
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

