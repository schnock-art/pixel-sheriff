export type ProjectShellSection = "datasets" | "models" | "experiments" | "deploy";

export function normalizeSection(section: unknown): ProjectShellSection;
export function deriveProjectSectionFromPathname(pathname: unknown): ProjectShellSection;
export function buildProjectSectionHref(projectId: string, section: unknown): string;
export function buildModelBuilderHref(projectId: string, modelId?: string | null): string;

