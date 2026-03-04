export interface DatasetVersionOption {
  id: string;
  name: string;
}

export function buildDatasetVersionOptions(items: unknown, configDatasetVersionId?: unknown): DatasetVersionOption[];
