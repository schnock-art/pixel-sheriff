import { useCallback, useEffect, useState } from "react";

import { listProjects, type Project } from "../api";
import { toHookError, type HookError } from "./hookError";

export function useProject() {
  const [data, setData] = useState<Project[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<HookError | null>(null);

  const refetch = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const projects = await listProjects();
      setData(projects);
    } catch (err) {
      setError(toHookError(err, "Failed to load projects"));
      setData([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    let isActive = true;

    async function load() {
      try {
        setIsLoading(true);
        setError(null);
        const projects = await listProjects();
        if (isActive) setData(projects);
      } catch (err) {
        if (isActive) {
          setError(toHookError(err, "Failed to load projects"));
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
  }, []);

  return { data, isLoading, error, refetch };
}
