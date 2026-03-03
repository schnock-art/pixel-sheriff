import { useEffect, useState, type Dispatch, type SetStateAction } from "react";

export function useProjectMultiLabelSettings(storageKey: string): {
  projectMultiLabelSettings: Record<string, boolean>;
  setProjectMultiLabelSettings: Dispatch<SetStateAction<Record<string, boolean>>>;
} {
  const [projectMultiLabelSettings, setProjectMultiLabelSettings] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (!raw) return;
      const parsed: unknown = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object") return;

      const normalized: Record<string, boolean> = {};
      for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
        normalized[key] = Boolean(value);
      }
      setProjectMultiLabelSettings(normalized);
    } catch {
      setProjectMultiLabelSettings({});
    }
  }, [storageKey]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(storageKey, JSON.stringify(projectMultiLabelSettings));
  }, [projectMultiLabelSettings, storageKey]);

  return {
    projectMultiLabelSettings,
    setProjectMultiLabelSettings,
  };
}
