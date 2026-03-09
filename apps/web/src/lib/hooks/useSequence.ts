import { useCallback, useEffect, useState } from "react";

import { getSequence, type AssetSequence } from "../api";
import { toHookError, type HookError } from "./hookError";

export function useSequence(projectId: string | null, sequenceId: string | null, taskId: string | null) {
  const [data, setData] = useState<AssetSequence | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<HookError | null>(null);

  const load = useCallback(
    async (
      effectiveProjectId: string,
      effectiveSequenceId: string,
      effectiveTaskId: string | null,
      isActive: () => boolean = () => true,
    ) => {
      try {
        setIsLoading(true);
        setError(null);
        const sequence = await getSequence(effectiveProjectId, effectiveSequenceId, effectiveTaskId);
        if (!isActive()) return;
        setData(sequence);
      } catch (err) {
        if (!isActive()) return;
        setError(toHookError(err, "Failed to load sequence"));
        setData(null);
      } finally {
        if (isActive()) setIsLoading(false);
      }
    },
    [],
  );

  const refetch = useCallback(async () => {
    if (!projectId || !sequenceId) {
      setData(null);
      setError(null);
      setIsLoading(false);
      return;
    }
    await load(projectId, sequenceId, taskId);
  }, [load, projectId, sequenceId, taskId]);

  useEffect(() => {
    let isActive = true;

    if (!projectId || !sequenceId) {
      setData(null);
      setError(null);
      setIsLoading(false);
      return;
    }

    void load(projectId, sequenceId, taskId, () => isActive);
    return () => {
      isActive = false;
    };
  }, [load, projectId, sequenceId, taskId]);

  return { data, isLoading, error, refetch };
}
