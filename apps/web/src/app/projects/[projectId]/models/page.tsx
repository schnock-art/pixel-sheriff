"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

import { ApiError, createProjectModel, listProjectModels, type ProjectModelSummary } from "../../../../lib/api";

interface ModelsPageProps {
  params: {
    projectId: string;
  };
}

function formatDate(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "-";
  return parsed.toLocaleDateString();
}

function formatTaskLabel(task: string, taskId: string | null | undefined): string {
  const normalized = (task || "classification").trim() || "classification";
  const shortId = typeof taskId === "string" && taskId.trim() ? taskId.slice(0, 8) : "";
  return shortId ? `${normalized} • ${shortId}` : normalized;
}

export default function ModelsPage({ params }: ModelsPageProps) {
  const router = useRouter();
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);

  const [models, setModels] = useState<ProjectModelSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    async function loadModels() {
      setIsLoading(true);
      setErrorMessage(null);
      try {
        const rows = await listProjectModels(projectId);
        if (isMounted) setModels(rows);
      } catch (error) {
        if (!isMounted) return;
        if (error instanceof ApiError && error.responseBody) {
          setErrorMessage(error.responseBody);
        } else {
          setErrorMessage(error instanceof Error ? error.message : "Failed to load models");
        }
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }

    void loadModels();

    return () => {
      isMounted = false;
    };
  }, [projectId]);

  async function handleCreateModel() {
    setIsCreating(true);
    setErrorMessage(null);
    try {
      const created = await createProjectModel(projectId, {});
      router.push(`/projects/${encodeURIComponent(projectId)}/models/${encodeURIComponent(created.id)}`);
    } catch (error) {
      if (error instanceof ApiError && error.responseBody) {
        setErrorMessage(error.responseBody);
      } else {
        setErrorMessage(error instanceof Error ? error.message : "Failed to create model");
      }
      setIsCreating(false);
    }
  }

  return (
    <main className="workspace-shell project-page-shell">
      <section className="workspace-frame project-content-frame placeholder-page">
        <header className="project-section-header">
          <h2>Models</h2>
          <button type="button" className="primary-button" onClick={() => void handleCreateModel()} disabled={isCreating || isLoading}>
            {isCreating ? "Creating..." : "+ New Model"}
          </button>
        </header>

        {errorMessage ? <p className="project-field-error">{errorMessage}</p> : null}

        {isLoading ? (
          <div className="placeholder-card">
            <p>Loading models...</p>
          </div>
        ) : null}

        {!isLoading && models.length === 0 ? (
          <div className="placeholder-card">
            <h3>No models yet</h3>
            <p>Create a model to configure architecture, train experiments, and export deployment artifacts.</p>
          </div>
        ) : null}

        {!isLoading && models.length > 0 ? (
          <div className="models-table-wrap">
            <table className="models-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Task</th>
                  <th>Backbone</th>
                  <th>Classes</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => (
                  <tr key={model.id}>
                    <td>
                      <Link href={`/projects/${encodeURIComponent(projectId)}/models/${encodeURIComponent(model.id)}`}>
                        {model.name}
                      </Link>
                    </td>
                    <td>{formatTaskLabel(model.task, model.task_id)}</td>
                    <td>{model.backbone_name}</td>
                    <td>{model.num_classes}</td>
                    <td>{formatDate(model.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>
    </main>
  );
}
