"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import {
  ApiError,
  createExperiment,
  listDatasetVersions,
  listProjectModels,
  type ProjectModelSummary,
} from "../../../../../lib/api";

interface NewExperimentPageProps {
  params: {
    projectId: string;
  };
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

export default function NewExperimentPage({ params }: NewExperimentPageProps) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);
  const modelIdFromQuery = searchParams.get("modelId") ?? "";
  const startedRef = useRef(false);

  const [models, setModels] = useState<ProjectModelSummary[]>([]);
  const [selectedModelId, setSelectedModelId] = useState(modelIdFromQuery);
  const [datasetVersionOptions, setDatasetVersionOptions] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedDatasetVersionId, setSelectedDatasetVersionId] = useState("");
  const [name, setName] = useState("");
  const [isLoadingModels, setIsLoadingModels] = useState(!modelIdFromQuery);
  const [isCreating, setIsCreating] = useState(Boolean(modelIdFromQuery));
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    if (modelIdFromQuery) {
      setSelectedModelId(modelIdFromQuery);
      return;
    }
    let isMounted = true;
    async function loadModels() {
      setIsLoadingModels(true);
      try {
        const [rows, datasetVersions] = await Promise.all([listProjectModels(projectId), listDatasetVersions(projectId)]);
        if (!isMounted) return;
        setModels(rows);
        if (!selectedModelId && rows.length > 0) {
          setSelectedModelId(rows[0].id);
        }
        const versionRows = (datasetVersions.items ?? []).map((item) => {
          const version = item.version as Record<string, unknown>;
          const id = typeof version.dataset_version_id === "string" ? version.dataset_version_id : "";
          const name = typeof version.name === "string" ? version.name : id;
          return { id, name };
        });
        setDatasetVersionOptions(versionRows.filter((item) => item.id));
        const activeId = datasetVersions.active_dataset_version_id ?? versionRows[0]?.id ?? "";
        setSelectedDatasetVersionId(activeId);
      } catch (error) {
        if (!isMounted) return;
        setErrorMessage(parseApiErrorMessage(error, "Failed to load models"));
      } finally {
        if (isMounted) setIsLoadingModels(false);
      }
    }
    void loadModels();
    return () => {
      isMounted = false;
    };
  }, [modelIdFromQuery, projectId, selectedModelId]);

  useEffect(() => {
    if (!modelIdFromQuery) return;
    if (startedRef.current) return;
    startedRef.current = true;
    let isMounted = true;

    async function createAndRedirect() {
      setErrorMessage(null);
      setIsCreating(true);
      try {
        const created = await createExperiment(projectId, { model_id: modelIdFromQuery });
        if (!isMounted) return;
        router.replace(`/projects/${encodeURIComponent(projectId)}/experiments/${encodeURIComponent(created.id)}`);
      } catch (error) {
        if (!isMounted) return;
        setErrorMessage(parseApiErrorMessage(error, "Failed to create experiment"));
      } finally {
        if (isMounted) setIsCreating(false);
      }
    }

    void createAndRedirect();

    return () => {
      isMounted = false;
    };
  }, [modelIdFromQuery, projectId, router]);

  async function handleCreateFromPicker() {
    if (!selectedModelId) return;
    setErrorMessage(null);
    setIsCreating(true);
    try {
      const created = await createExperiment(projectId, {
        model_id: selectedModelId,
        dataset_version_id: selectedDatasetVersionId || undefined,
        name: name.trim() || undefined,
      });
      router.replace(`/projects/${encodeURIComponent(projectId)}/experiments/${encodeURIComponent(created.id)}`);
    } catch (error) {
      setErrorMessage(parseApiErrorMessage(error, "Failed to create experiment"));
      setIsCreating(false);
    }
  }

  return (
    <main className="workspace-shell project-page-shell">
      <section className="workspace-frame project-content-frame placeholder-page">
        <header className="project-section-header">
          <h2>Train Experiment</h2>
        </header>
        <div className="placeholder-card experiment-new-card">
          {modelIdFromQuery ? <p>{errorMessage ?? (isCreating ? "Creating experiment draft..." : "Experiment created.")}</p> : null}

          {!modelIdFromQuery ? (
            <>
              <label className="project-field">
                <span>Model</span>
                <select value={selectedModelId} onChange={(event) => setSelectedModelId(event.target.value)} disabled={isLoadingModels}>
                  {models.length === 0 ? <option value="">No models available</option> : null}
                  {models.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="project-field">
                <span>Experiment Name (optional)</span>
                <input
                  type="text"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="training_run_1"
                />
              </label>
              <label className="project-field">
                <span>Dataset Version</span>
                <select
                  value={selectedDatasetVersionId}
                  onChange={(event) => setSelectedDatasetVersionId(event.target.value)}
                  disabled={datasetVersionOptions.length === 0}
                >
                  {datasetVersionOptions.length === 0 ? <option value="">No dataset versions</option> : null}
                  {datasetVersionOptions.map((row) => (
                    <option key={row.id} value={row.id}>
                      {row.name}
                    </option>
                  ))}
                </select>
              </label>
              {errorMessage ? <p className="project-field-error">{errorMessage}</p> : null}
              <div className="project-modal-actions">
                <button
                  type="button"
                  className="primary-button"
                  disabled={!selectedModelId || isCreating || isLoadingModels || !selectedDatasetVersionId}
                  onClick={() => void handleCreateFromPicker()}
                >
                  {isCreating ? "Creating..." : "Create Experiment"}
                </button>
              </div>
            </>
          ) : null}
        </div>
      </section>
    </main>
  );
}
