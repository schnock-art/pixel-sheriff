export type ProjectShellSection = "datasets" | "dataset" | "models" | "experiments" | "deploy";

export function normalizeSection(section: unknown): ProjectShellSection;
export function deriveProjectSectionFromPathname(pathname: unknown): ProjectShellSection;
export function buildProjectSectionHref(projectId: string, section: unknown): string;
export function buildModelBuilderHref(projectId: string, modelId: string | null | undefined): string;
export function buildModelCreateHref(
  projectId: string,
  options?: {
    taskId?: string | null | undefined;
    datasetVersionId?: string | null | undefined;
  },
): string;
