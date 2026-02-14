import { useCallback, useEffect, useState } from "react";

import { listCategories, type Category } from "../api";

export function useLabels(projectId: string | null) {
  const [data, setData] = useState<Category[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    if (!projectId) {
      setData([]);
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      const categories = await listCategories(projectId);
      setData(categories);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load labels");
      setData([]);
    } finally {
      setIsLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    let isActive = true;

    async function load() {
      if (!projectId) {
        setData([]);
        return;
      }

      try {
        setIsLoading(true);
        setError(null);
        const categories = await listCategories(projectId);
        if (isActive) setData(categories);
      } catch (err) {
        if (isActive) {
          setError(err instanceof Error ? err.message : "Failed to load labels");
          setData([]);
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

  return { data, isLoading, error, refetch };
}
