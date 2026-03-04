import { useCallback, useEffect, useState } from "react";

import { createTask, listTasks, type Task, type TaskCreatePayload } from "../api";
import { toHookError, type HookError } from "./hookError";

export function useTasks(projectId: string | null) {
  const [data, setData] = useState<Task[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<HookError | null>(null);

  const refetch = useCallback(async () => {
    if (!projectId) {
      setData([]);
      setError(null);
      setIsLoading(false);
      return;
    }
    try {
      setIsLoading(true);
      setError(null);
      const rows = await listTasks(projectId);
      setData(rows);
    } catch (err) {
      setError(toHookError(err, "Failed to load tasks"));
      setData([]);
    } finally {
      setIsLoading(false);
    }
  }, [projectId]);

  const create = useCallback(
    async (payload: TaskCreatePayload): Promise<Task> => {
      if (!projectId) {
        throw new Error("Project is required");
      }
      const created = await createTask(projectId, payload);
      setData((previous) => [...previous.filter((row) => row.id !== created.id), created]);
      return created;
    },
    [projectId],
  );

  useEffect(() => {
    void refetch();
  }, [refetch]);

  return { data, isLoading, error, refetch, create };
}
