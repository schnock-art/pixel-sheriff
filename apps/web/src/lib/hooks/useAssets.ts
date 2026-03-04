import { useCallback, useEffect, useState } from "react";

import { listAnnotations, listAssets, type Annotation, type Asset } from "../api";
import { toHookError, type HookError } from "./hookError";

export function useAssets(projectId: string | null, taskId: string | null) {
  const [data, setData] = useState<Asset[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<HookError | null>(null);

  const load = useCallback(
    async (effectiveProjectId: string, effectiveTaskId: string | null, isActive: () => boolean = () => true) => {
      try {
        setIsLoading(true);
        setError(null);
        const assets = await listAssets(effectiveProjectId);
        let annotationRows: Annotation[] = [];
        if (effectiveTaskId) {
          annotationRows = await listAnnotations(effectiveProjectId, effectiveTaskId);
        }
        if (!isActive()) return;
        setData(assets);
        setAnnotations(annotationRows);
      } catch (err) {
        if (!isActive()) return;
        setError(toHookError(err, "Failed to load assets"));
        setData([]);
        setAnnotations([]);
      } finally {
        if (isActive()) setIsLoading(false);
      }
    },
    [],
  );

  const refetch = useCallback(async (projectIdOverride?: string | null) => {
    const effectiveProjectId = projectIdOverride ?? projectId;
    if (!effectiveProjectId) {
      setData([]);
      setAnnotations([]);
      setError(null);
      setIsLoading(false);
      return;
    }
    await load(effectiveProjectId, taskId);
  }, [load, projectId, taskId]);

  useEffect(() => {
    let isActive = true;

    if (!projectId) {
      setData([]);
      setAnnotations([]);
      setError(null);
      setIsLoading(false);
      return;
    }

    void load(projectId, taskId, () => isActive);

    return () => {
      isActive = false;
    };
  }, [load, projectId, taskId]);

  return { data, annotations, isLoading, error, setAnnotations, refetch };
}
