import { useCallback, useEffect, useState } from "react";

import { listAnnotations, listAssets, type Annotation, type Asset } from "../api";

export function useAssets(projectId: string | null) {
  const [data, setData] = useState<Asset[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    if (!projectId) {
      setData([]);
      setAnnotations([]);
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      const [assets, annotationRows] = await Promise.all([listAssets(projectId), listAnnotations(projectId)]);
      setData(assets);
      setAnnotations(annotationRows);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load assets");
      setData([]);
      setAnnotations([]);
    } finally {
      setIsLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    let isActive = true;

    async function load() {
      if (!projectId) {
        setData([]);
        setAnnotations([]);
        return;
      }

      try {
        setIsLoading(true);
        setError(null);
        const [assets, annotationRows] = await Promise.all([listAssets(projectId), listAnnotations(projectId)]);
        if (isActive) {
          setData(assets);
          setAnnotations(annotationRows);
        }
      } catch (err) {
        if (isActive) {
          setError(err instanceof Error ? err.message : "Failed to load assets");
          setData([]);
          setAnnotations([]);
        }
      } finally {
        if (isActive) setIsLoading(false);
      }
    }

    void load();

    return () => {
      isActive = false;
    };
  }, [projectId]);

  return { data, annotations, isLoading, error, setAnnotations, refetch };
}
