"use client";

import { useEffect, useMemo, useState } from "react";

import {
  ApiError,
  createDeployment,
  getExperimentOnnx,
  listDeployments,
  listExperiments,
  patchDeployment,
  warmupDeployment,
  type DeploymentItem,
  type DeploymentListResponse,
  type DeploymentDevicePreference,
  type ProjectExperimentSummary,
} from "../../../../lib/api";

interface DeployPageProps {
  params: { projectId: string };
}

function parseApiError(error: unknown, fallback: string): string {
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

export default function DeployPage({ params }: DeployPageProps) {
  const projectId = decodeURIComponent(params.projectId);
  const [deployments, setDeployments] = useState<DeploymentListResponse>({ active_deployment_id: null, items: [] });
  const [experiments, setExperiments] = useState<ProjectExperimentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [createNameByExperiment, setCreateNameByExperiment] = useState<Record<string, string>>({});
  const [creatingExperimentId, setCreatingExperimentId] = useState<string | null>(null);
  const [warmingDeploymentId, setWarmingDeploymentId] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoading(true);
      setErrorMessage(null);
      try {
        const [deploymentResp, experimentsResp] = await Promise.all([
          listDeployments(projectId),
          listExperiments(projectId),
        ]);
        if (!mounted) return;
        setDeployments(deploymentResp);
        setExperiments(experimentsResp.items ?? []);
      } catch (error) {
        if (!mounted) return;
        setErrorMessage(parseApiError(error, "Failed to load deploy page"));
      } finally {
        if (mounted) setLoading(false);
      }
    }
    void load();
    return () => {
      mounted = false;
    };
  }, [projectId]);

  const availableDeployments = useMemo(
    () => deployments.items.filter((item) => item.status === "available"),
    [deployments.items],
  );
  const activeDeployment = deployments.items.find((item) => item.deployment_id === deployments.active_deployment_id) ?? null;

  async function refreshDeployments() {
    const next = await listDeployments(projectId);
    setDeployments(next);
  }

  async function handleSetActive(nextDeploymentId: string | null) {
    if (!nextDeploymentId) {
      if (!deployments.active_deployment_id) return;
      await patchDeployment(projectId, deployments.active_deployment_id, { is_active: false });
      await refreshDeployments();
      return;
    }
    await patchDeployment(projectId, nextDeploymentId, { is_active: true });
    await refreshDeployments();
  }

  async function handlePatchDevice(deploymentId: string, devicePreference: DeploymentDevicePreference) {
    await patchDeployment(projectId, deploymentId, { device_preference: devicePreference });
    await refreshDeployments();
  }

  async function handleDeploy(experimentId: string) {
    setCreatingExperimentId(experimentId);
    setErrorMessage(null);
    try {
      const onnx = await getExperimentOnnx(projectId, experimentId);
      const name = createNameByExperiment[experimentId]?.trim() || `deploy_${experimentId.slice(0, 8)}`;
      await createDeployment(projectId, {
        name,
        task: "classification",
        device_preference: "auto",
        source: {
          experiment_id: experimentId,
          attempt: onnx.attempt,
          checkpoint_kind: "best_metric",
        },
        is_active: availableDeployments.length === 0,
      });
      await refreshDeployments();
    } catch (error) {
      setErrorMessage(parseApiError(error, "Failed to deploy model"));
    } finally {
      setCreatingExperimentId(null);
    }
  }

  async function handleWarmup(deploymentId: string) {
    setWarmingDeploymentId(deploymentId);
    setErrorMessage(null);
    try {
      await warmupDeployment(projectId, deploymentId);
    } catch (error) {
      setErrorMessage(parseApiError(error, "Warmup failed"));
    } finally {
      setWarmingDeploymentId(null);
    }
  }

  return (
    <main className="workspace-shell project-page-shell">
      <section className="workspace-frame project-content-frame placeholder-page">
        <header className="project-section-header">
          <h2>Deploy</h2>
        </header>

        {errorMessage ? <p className="project-field-error">{errorMessage}</p> : null}
        {loading ? <p>Loading deployments...</p> : null}

        {!loading ? (
          <>
            <section className="placeholder-card">
              <h3>Active Model</h3>
              <label className="project-field">
                <span>Deployment</span>
                <select
                  value={deployments.active_deployment_id ?? ""}
                  onChange={(event) => void handleSetActive(event.target.value || null)}
                >
                  <option value="">None</option>
                  {availableDeployments.map((item) => (
                    <option key={item.deployment_id} value={item.deployment_id}>
                      {item.name} ({new Date(item.created_at).toLocaleDateString()})
                    </option>
                  ))}
                </select>
              </label>
              {activeDeployment ? (
                <label className="project-field">
                  <span>Device preference</span>
                  <select
                    value={activeDeployment.device_preference}
                    onChange={(event) =>
                      void handlePatchDevice(activeDeployment.deployment_id, event.target.value as DeploymentDevicePreference)
                    }
                  >
                    <option value="auto">Auto</option>
                    <option value="cuda">CUDA</option>
                    <option value="cpu">CPU</option>
                  </select>
                </label>
              ) : null}
              {activeDeployment ? (
                <button
                  type="button"
                  className="ghost-button"
                  onClick={() => void handleWarmup(activeDeployment.deployment_id)}
                  disabled={warmingDeploymentId === activeDeployment.deployment_id}
                >
                  {warmingDeploymentId === activeDeployment.deployment_id ? "Warming..." : "Warm up model"}
                </button>
              ) : null}
            </section>

            <section className="placeholder-card">
              <h3>Deploy from Experiments</h3>
              <ul className="label-manage-list">
                {experiments.map((experiment) => (
                  <li key={experiment.id} className="label-manage-item">
                    <div>
                      <strong>{experiment.name}</strong>
                      <div>
                        {experiment.status}
                        {experiment.task_id ? ` | task=${experiment.task_id.slice(0, 8)}` : ""}
                      </div>
                    </div>
                    <input
                      className="label-manage-input"
                      placeholder="deployment name"
                      value={createNameByExperiment[experiment.id] ?? ""}
                      onChange={(event) =>
                        setCreateNameByExperiment((previous) => ({ ...previous, [experiment.id]: event.target.value }))
                      }
                    />
                    <button
                      type="button"
                      className="primary-button"
                      onClick={() => void handleDeploy(experiment.id)}
                      disabled={creatingExperimentId === experiment.id}
                    >
                      {creatingExperimentId === experiment.id ? "Deploying..." : "Deploy"}
                    </button>
                  </li>
                ))}
              </ul>
            </section>

            <section className="placeholder-card">
              <h3>All Deployments</h3>
              <ul className="label-manage-list">
                {deployments.items.map((item: DeploymentItem) => (
                  <li key={item.deployment_id} className="label-manage-item">
                    <div>
                      <strong>{item.name}</strong>
                      <div>
                        {item.status} | task={item.task} | device={item.device_preference} | model_key={item.model_key.slice(0, 12)}
                      </div>
                    </div>
                    <button
                      type="button"
                      className="ghost-button"
                      onClick={() => void patchDeployment(projectId, item.deployment_id, { status: item.status === "archived" ? "available" : "archived" }).then(refreshDeployments)}
                    >
                      {item.status === "archived" ? "Unarchive" : "Archive"}
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          </>
        ) : null}
      </section>
    </main>
  );
}
