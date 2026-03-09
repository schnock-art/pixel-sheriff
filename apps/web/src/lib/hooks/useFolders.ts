import { useCallback, useEffect, useMemo, useState } from "react";

import { listFolders, type Folder } from "../api";
import { toHookError, type HookError } from "./hookError";

export function useFolders(projectId: string | null) {
  const [data, setData] = useState<Folder[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<HookError | null>(null);

  const load = useCallback(async (effectiveProjectId: string, isActive: () => boolean = () => true) => {
    try {
      setIsLoading(true);
      setError(null);
      const folders = await listFolders(effectiveProjectId);
      if (!isActive()) return;
      setData(folders);
    } catch (err) {
      if (!isActive()) return;
      setError(toHookError(err, "Failed to load folders"));
      setData([]);
    } finally {
      if (isActive()) setIsLoading(false);
    }
  }, []);

  const refetch = useCallback(async (projectIdOverride?: string | null) => {
    const effectiveProjectId = projectIdOverride ?? projectId;
    if (!effectiveProjectId) {
      setData([]);
      setError(null);
      setIsLoading(false);
      return;
    }
    await load(effectiveProjectId);
  }, [load, projectId]);

  useEffect(() => {
    let isActive = true;

    if (!projectId) {
      setData([]);
      setError(null);
      setIsLoading(false);
      return;
    }

    void load(projectId, () => isActive);
    return () => {
      isActive = false;
    };
  }, [load, projectId]);

  const hasProcessingSequences = useMemo(
    () => data.some((folder) => folder.sequence_status === "processing"),
    [data],
  );

  return { data, isLoading, error, refetch, hasProcessingSequences };
}
