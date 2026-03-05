"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { ApiError, createProjectModel, listDatasetVersions } from "../../../../../lib/api";

interface NewModelPageProps {
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

export default function NewModelPage({ params }: NewModelPageProps) {
  const router = useRouter();
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);

  const [datasetVersionOptions, setDatasetVersionOptions] = useState<Array<{ id: string; name: string; task: string }>>([]);
  const [selectedTask, setSelectedTask] = useState("");
  const [selectedVersionId, setSelectedVersionId] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    async function loadVersions() {
      setIsLoading(true);
      try {
        const data = await listDatasetVersions(projectId);
        if (!isMounted) return;
        const rows = (data.items ?? []).map((item) => {
          const version = item.version as Record<string, unknown>;
          const id = typeof version.dataset_version_id === "string" ? version.dataset_version_id : "";
          const name = typeof version.name === "string" ? version.name : id;
          const task = typeof version.task === "string" ? version.task : "";
          return { id, name, task };
        }).filter((item) => item.id);
        setDatasetVersionOptions(rows);

        // Set initial selected version (active or first available)
        const activeId = data.active_dataset_version_id ?? "";
        const activeVersion = rows.find((r) => r.id === activeId);
        const defaultVersion = activeVersion ?? rows[0];
        setSelectedTask(defaultVersion?.task ?? "");
        setSelectedVersionId(defaultVersion?.id ?? "");
      } catch (error) {
        if (!isMounted) return;
        setErrorMessage(parseApiErrorMessage(error, "Failed to load dataset versions"));
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }
    void loadVersions();
    return () => {
      isMounted = false;
    };
  }, [projectId]);

  // Unique tasks derived from loaded versions
  const taskOptions = useMemo(() => {
    const seen = new Set<string>();
    const tasks: string[] = [];
    for (const row of datasetVersionOptions) {
      if (row.task && !seen.has(row.task)) {
        seen.add(row.task);
        tasks.push(row.task);
      }
    }
    return tasks;
  }, [datasetVersionOptions]);

  // Versions filtered by selected task
  const filteredVersions = useMemo(
    () => datasetVersionOptions.filter((r) => !selectedTask || r.task === selectedTask),
    [datasetVersionOptions, selectedTask],
  );

  function handleTaskChange(task: string) {
    setSelectedTask(task);
    // Reset selected version to first in filtered list
    const first = datasetVersionOptions.find((r) => r.task === task);
    setSelectedVersionId(first?.id ?? "");
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!selectedVersionId || isCreating) return;
    setErrorMessage(null);
    setIsCreating(true);
    try {
      const created = await createProjectModel(projectId, { dataset_version_id: selectedVersionId });
      router.replace(`/projects/${encodeURIComponent(projectId)}/models/${encodeURIComponent(created.id)}`);
    } catch (error) {
      setErrorMessage(parseApiErrorMessage(error, "Failed to create model"));
      setIsCreating(false);
    }
  }

  return (
    <main className="workspace-shell project-page-shell">
      <section className="workspace-frame project-content-frame placeholder-page">
        <header className="project-section-header">
          <h2>New Model</h2>
        </header>
        <div className="placeholder-card">
          {isLoading ? (
            <p>Loading dataset versions...</p>
          ) : errorMessage && datasetVersionOptions.length === 0 ? (
            <p className="project-field-error">{errorMessage}</p>
          ) : datasetVersionOptions.length === 0 ? (
            <p>No dataset versions found. Create a dataset version before adding a model.</p>
          ) : (
            <form onSubmit={(e) => void handleSubmit(e)}>
              <label className="project-field">
                <span>Task</span>
                <select value={selectedTask} onChange={(e) => handleTaskChange(e.target.value)}>
                  {taskOptions.map((task) => (
                    <option key={task} value={task}>
                      {task}
                    </option>
                  ))}
                </select>
              </label>
              <label className="project-field">
                <span>Dataset Version</span>
                <select
                  value={selectedVersionId}
                  onChange={(e) => setSelectedVersionId(e.target.value)}
                  disabled={filteredVersions.length === 0}
                >
                  {filteredVersions.length === 0 ? (
                    <option value="">No versions for this task</option>
                  ) : null}
                  {filteredVersions.map((row) => (
                    <option key={row.id} value={row.id}>
                      {row.task ? `${row.name} (${row.task})` : row.name}
                    </option>
                  ))}
                </select>
              </label>
              {errorMessage ? <p className="project-field-error">{errorMessage}</p> : null}
              <div className="project-modal-actions">
                <button
                  type="submit"
                  className="primary-button"
                  disabled={!selectedVersionId || isCreating}
                >
                  {isCreating ? "Creating..." : "Create Model"}
                </button>
              </div>
            </form>
          )}
        </div>
      </section>
    </main>
  );
}
