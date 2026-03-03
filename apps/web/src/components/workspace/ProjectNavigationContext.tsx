"use client";

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

import { shouldAllowNavigation } from "../../lib/workspace/navigationGuard";

export type ProjectShellSection = "datasets" | "dataset" | "models" | "experiments" | "deploy";

export interface ProjectNavigationGuardState {
  hasUnsavedDrafts: boolean;
  setHasUnsavedDrafts: (value: boolean) => void;
  guardedNavigate: (onNavigate: () => void) => boolean;
}

const ProjectNavigationContext = createContext<ProjectNavigationGuardState | null>(null);

export function ProjectNavigationGuardProvider({ children }: { children: ReactNode }) {
  const [hasUnsavedDrafts, setHasUnsavedDrafts] = useState(false);

  const guardedNavigate = useCallback(
    (onNavigate: () => void) => {
      const allow = shouldAllowNavigation({
        hasUnsavedDrafts,
        confirmDiscard: () => window.confirm("You have unsaved annotation drafts. Leave and discard?"),
      });
      if (!allow) return false;
      if (hasUnsavedDrafts) setHasUnsavedDrafts(false);
      onNavigate();
      return true;
    },
    [hasUnsavedDrafts],
  );

  const value = useMemo(
    () => ({ hasUnsavedDrafts, setHasUnsavedDrafts, guardedNavigate }),
    [guardedNavigate, hasUnsavedDrafts],
  );

  return <ProjectNavigationContext.Provider value={value}>{children}</ProjectNavigationContext.Provider>;
}

export function useProjectNavigationGuard(): ProjectNavigationGuardState {
  const context = useContext(ProjectNavigationContext);
  if (!context) {
    throw new Error("useProjectNavigationGuard must be used within ProjectNavigationGuardProvider");
  }
  return context;
}

