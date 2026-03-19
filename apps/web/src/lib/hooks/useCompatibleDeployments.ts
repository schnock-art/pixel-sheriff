import { useEffect, useMemo, useState } from "react";

import { listDeployments, type DeploymentItem, type TaskKind } from "../api";

interface UseCompatibleDeploymentsParams {
  projectId: string | null;
  taskId: string | null;
  taskKind: TaskKind | null;
}

export function useCompatibleDeployments({ projectId, taskId, taskKind }: UseCompatibleDeploymentsParams) {
  const [items, setItems] = useState<DeploymentItem[]>([]);
  const [activeDeploymentId, setActiveDeploymentId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    let isMounted = true;

    async function loadDeployments() {
      if (!projectId) {
        if (!isMounted) return;
        setItems([]);
        setActiveDeploymentId(null);
        setIsLoading(false);
        return;
      }

      try {
        if (isMounted) setIsLoading(true);
        const response = await listDeployments(projectId);
        if (!isMounted) return;
        setItems(Array.isArray(response.items) ? response.items : []);
        setActiveDeploymentId(response.active_deployment_id ?? null);
      } catch {
        if (!isMounted) return;
        setItems([]);
        setActiveDeploymentId(null);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }

    void loadDeployments();
    return () => {
      isMounted = false;
    };
  }, [projectId]);

  const deployments = useMemo(
    () =>
      items.filter((item) => {
        if (item.status !== "available") return false;
        if (taskKind && item.task !== taskKind) return false;
        if (taskId && item.task_id && item.task_id !== taskId) return false;
        return true;
      }),
    [items, taskId, taskKind],
  );

  const activeCompatibleDeployment = useMemo(
    () => deployments.find((item) => item.deployment_id === activeDeploymentId) ?? null,
    [activeDeploymentId, deployments],
  );

  return {
    deployments,
    activeDeploymentId,
    activeCompatibleDeployment,
    isLoading,
  };
}
