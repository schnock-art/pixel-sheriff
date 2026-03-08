"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { ProjectSectionLayout } from "../../../../../components/workspace/project-shell/ProjectSectionLayout";
import { useProjectShell } from "../../../../../components/workspace/project-shell/ProjectShellContext";
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
  const searchParams = useSearchParams();
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);
  const { selectedTaskId } = useProjectShell();
  const taskIdFromQuery = searchParams.get("taskId");
  const datasetVersionIdFromQuery = searchParams.get("datasetVersionId");
  const preferredTaskId = taskIdFromQuery || selectedTaskId;

  const [datasetVersionOptions, setDatasetVersionOptions] = useState<Array<{ id: string; name: string; task: string; taskId: string | null }>>([]);
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
        const rows = (data.items ?? [])
          .map((item) => {
            const version = item.version as Record<string, unknown>;
            const id = typeof version.dataset_version_id === "string" ? version.dataset_version_id : "";
            const name = typeof version.name === "string" ? version.name : id;
            const task = typeof version.task === "string" ? version.task : "";
            const taskId = typeof version.task_id === "string" ? version.task_id : null;
            return { id, name, task, taskId };
          })
          .filter((item) => item.id);
        setDatasetVersionOptions(rows);

        const rowsForPreferredTask = preferredTaskId ? rows.filter((row) => row.taskId === preferredTaskId) : rows;
        const requestedVersion = datasetVersionIdFromQuery ? rows.find((row) => row.id === datasetVersionIdFromQuery) ?? null : null;
        const activeId = data.active_dataset_version_id ?? "";
        const activeForTask = rowsForPreferredTask.find((row) => row.id === activeId) ?? null;
        const fallbackForTask = rowsForPreferredTask[0] ?? rows[0] ?? null;
        const chosen = requestedVersion ?? activeForTask ?? fallbackForTask;

        setSelectedTask(chosen?.taskId ?? preferredTaskId ?? "");
        setSelectedVersionId(chosen?.id ?? "");
        setErrorMessage(null);
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
  }, [datasetVersionIdFromQuery, preferredTaskId, projectId]);

  const taskOptions = useMemo(() => {
    const seen = new Set<string>();
    const options: Array<{ taskId: string; taskLabel: string }> = [];
    for (const row of datasetVersionOptions) {
      if (!row.taskId || seen.has(row.taskId)) continue;
      seen.add(row.taskId);
      options.push({ taskId: row.taskId, taskLabel: row.task || row.taskId });
    }
    return options;
  }, [datasetVersionOptions]);

  const filteredVersions = useMemo(
    () => datasetVersionOptions.filter((row) => !selectedTask || row.taskId === selectedTask),
    [datasetVersionOptions, selectedTask],
  );

  function handleTaskChange(taskId: string) {
    setSelectedTask(taskId);
    const first = datasetVersionOptions.find((row) => row.taskId === taskId);
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
    <ProjectSectionLayout
      title="New Model"
      description="Start from a dataset version, confirm the task context, and continue into the model builder with the source dataset already attached."
    >
      <div className="placeholder-card model-entry-card">
        {isLoading ? (
          <p>Loading dataset versions...</p>
        ) : errorMessage && datasetVersionOptions.length === 0 ? (
          <p className="project-field-error">{errorMessage}</p>
        ) : datasetVersionOptions.length === 0 ? (
          <p>No dataset versions found. Create a dataset version before adding a model.</p>
        ) : (
          <form className="model-entry-form" onSubmit={(event) => void handleSubmit(event)} data-testid="new-model-form">
            <div className="model-entry-intro">
              <h3>Model source</h3>
              <p>Select the dataset version you want this model to track. The next screen opens the full builder with architecture and export settings.</p>
            </div>
            <label className="project-field">
              <span>Task</span>
              <select data-testid="new-model-task-select" value={selectedTask} onChange={(event) => handleTaskChange(event.target.value)}>
                {taskOptions.map((task) => (
                  <option key={task.taskId} value={task.taskId}>
                    {task.taskLabel}
                  </option>
                ))}
              </select>
            </label>
            <label className="project-field">
              <span>Dataset Version</span>
              <select
                data-testid="new-model-version-select"
                value={selectedVersionId}
                onChange={(event) => setSelectedVersionId(event.target.value)}
                disabled={filteredVersions.length === 0}
              >
                {filteredVersions.length === 0 ? <option value="">No versions for this task</option> : null}
                {filteredVersions.map((row) => (
                  <option key={row.id} value={row.id}>
                    {row.task ? `${row.name} (${row.task})` : row.name}
                  </option>
                ))}
              </select>
            </label>
            {errorMessage ? <p className="project-field-error">{errorMessage}</p> : null}
            <div className="project-modal-actions">
              <button type="submit" className="primary-button" disabled={!selectedVersionId || isCreating} data-testid="new-model-submit">
                {isCreating ? "Creating..." : "Continue to Builder"}
              </button>
            </div>
          </form>
        )}
      </div>
    </ProjectSectionLayout>
  );
}
