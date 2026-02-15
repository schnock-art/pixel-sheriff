export type PageToken = { type: "page"; page: number } | { type: "ellipsis"; key: string };

export function estimateMaxVisiblePages(total: number, containerWidth: number): number;
export function buildPageTokens(total: number, current: number, maxVisiblePages: number): PageToken[];
