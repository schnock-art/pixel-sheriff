"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { ModelTable, type ModelTableRowView } from "../../../../components/workspace/models/ModelTable";
import { ProjectSectionLayout } from "../../../../components/workspace/project-shell/ProjectSectionLayout";
import { useProjectShell } from "../../../../components/workspace/project-shell/ProjectShellContext";
import {
  ApiError,
  getProjectModel,
  listDatasetVersions,
  listExperiments,
  listProjectModels,
} from "../../../../lib/api";
import { deriveModelDatasetVersion, deriveModelStatus } from "../../../../lib/workspace/modelList";
import { buildModelCreateHref } from "../../../../lib/workspace/projectRouting";

interface ModelsPageProps {
  params: {
    projectId: string;
  };
}

function parseError(error: unknown, fallback: string) {
  if (error instanceof ApiError && error.responseBody) return error.responseBody;
  return error instanceof Error ? error.message : fallback;
}

function formatTaskLabel(task: string, taskId: string | null | undefined): string {
  const normalized = (task || "classification").trim() || "classification";
  const shortId = typeof taskId === "string" && taskId.trim() ? taskId.slice(0, 8) : "";
  return shortId ? `${normalized} • ${shortId}` : normalized;
}

export default function ModelsPage({ params }: ModelsPageProps) {
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);
  const { selectedTaskId } = useProjectShell();
  const [rows, setRows] = useState<ModelTableRowView[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadModels() {
      setIsLoading(true);
      setErrorMessage(null);
      try {
        const [models, datasetVersions, experiments] = await Promise.all([
          listProjectModels(projectId),
          listDatasetVersions(projectId),
          listExperiments(projectId),
        ]);
        const datasetVersionNameById = Object.fromEntries(
          (datasetVersions.items ?? []).map((item) => {
            const version = item.version as Record<string, unknown>;
            const id = typeof version.dataset_version_id === "string" ? version.dataset_version_id : "";
            const name = typeof version.name === "string" ? version.name : id;
            return [id, name];
          }).filter(([id]) => Boolean(id)),
        ) as Record<string, string>;

        const modelDetails = await Promise.all(
          models.map(async (model) => ({
            summary: model,
            detail: await getProjectModel(projectId, model.id),
          })),
        );

        if (!active) return;

        const nextRows = modelDetails
          .filter(({ summary }) => !summary.task_id || !selectedTaskId || summary.task_id === selectedTaskId)
          .map(({ summary, detail }) => {
            const datasetVersion = deriveModelDatasetVersion(detail.config_json, datasetVersionNameById);
            return {
              id: summary.id,
              href: `/projects/${encodeURIComponent(projectId)}/models/${encodeURIComponent(summary.id)}`,
              name: summary.name,
              taskLabel: formatTaskLabel(summary.task, summary.task_id),
              datasetVersionName: datasetVersion.datasetVersionName,
              backboneName: summary.backbone_name,
              numClasses: summary.num_classes,
              status: deriveModelStatus(experiments.items ?? [], summary.id, datasetVersion.hasSourceDataset),
              createdAt: summary.created_at,
            } satisfies ModelTableRowView;
          });

        setRows(nextRows);
      } catch (error) {
        if (!active) return;
        setErrorMessage(parseError(error, "Failed to load models"));
        setRows([]);
      } finally {
        if (active) setIsLoading(false);
      }
    }

    void loadModels();
    return () => {
      active = false;
    };
  }, [projectId, selectedTaskId]);

  const createHref = buildModelCreateHref(projectId, { taskId: selectedTaskId });

  return (
    <ProjectSectionLayout
      title="Models"
      description="Create architecture drafts tied to dataset versions, review model readiness, and move into experiments with a clearer training pipeline."
      actions={
        <Link className="primary-button" href={createHref}>
          + New Model
        </Link>
      }
    >
      {errorMessage ? <p className="project-field-error">{errorMessage}</p> : null}

      {isLoading ? (
        <div className="placeholder-card">
          <p>Loading models...</p>
        </div>
      ) : null}

      {!isLoading && rows.length === 0 ? (
        <div className="placeholder-card">
          <h3>No models yet</h3>
          <p>Create a model from a dataset version to configure architecture, launch experiments, and prepare deployment artifacts.</p>
        </div>
      ) : null}

      {!isLoading && rows.length > 0 ? <ModelTable rows={rows} /> : null}
    </ProjectSectionLayout>
  );
}
