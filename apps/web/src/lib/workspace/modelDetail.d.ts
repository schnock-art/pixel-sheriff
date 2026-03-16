export interface DatasetVersionRecord {
  id: string;
  name: string;
  task: string;
  label_mode?: "single_label" | "multi_label" | null;
  num_classes: number;
  class_order: string[];
  class_names: Record<string, string>;
}

export interface FamilyInputSizeRule {
  shape?: string;
  mode?: string;
  min_square_size?: number;
  step?: number;
  recommended_square_size?: number;
  required_square_size?: number;
}

export interface FamiliesMetadata {
  families: Array<{
    name: string;
    task: string;
    allowed_backbones?: string[];
    input_size?: FamilyInputSizeRule;
  }>;
}

export interface ValidationIssue {
  path?: string;
  keyword?: string;
  message?: string;
}

export const INPUT_SIZE_PRESETS: number[];
export const RESIZE_POLICY_OPTIONS: string[];
export const NORMALIZATION_OPTIONS: string[];
export const EMBEDDING_DIM_OPTIONS: number[];
export const EMBEDDING_NORMALIZE_OPTIONS: Array<"none" | "l2">;

export function asRecord(value: unknown): Record<string, unknown>;
export function parseApiErrorMessage(error: unknown, fallback: string): string;
export function normalizeModelTask(task: string | null | undefined): string | null;
export function tasksMatch(left: string | null | undefined, right: string | null | undefined): boolean;
export function getFamilyInputSizeRule(
  family: { input_size?: FamilyInputSizeRule } | null | undefined,
): FamilyInputSizeRule | null;
export function isAllowedSquareSize(rule: FamilyInputSizeRule | null, size: number): boolean;
export function formatFamilyInputSizeHint(rule: FamilyInputSizeRule | null): string | null;
export function mapDatasetVersionRecords(items: Array<{ version?: Record<string, unknown> }>): DatasetVersionRecord[];
export function deriveModelDetailState(options: {
  draftConfig?: Record<string, unknown> | null;
  allDatasetVersions?: DatasetVersionRecord[];
  familiesMetadata?: FamiliesMetadata;
  validationErrors?: ValidationIssue[];
}): {
  input: Record<string, unknown>;
  inputSizeWidth: number;
  inputSizeHeight: number;
  inputSizePresetValue: string;
  customInputSizeValue: string;
  normalizationType: string;
  resizePolicy: string;
  architecture: Record<string, unknown>;
  backboneName: string;
  pretrained: boolean;
  currentFamilyName: string | null;
  currentManifestId: string | null;
  currentVersionFromManifest: DatasetVersionRecord | null;
  currentFamilyFromMeta: { name: string; task: string; allowed_backbones?: string[]; input_size?: FamilyInputSizeRule } | null;
  currentTask: string | null;
  uniqueTasks: string[];
  familiesForTask: Array<{ name: string; task: string; allowed_backbones?: string[]; input_size?: FamilyInputSizeRule }>;
  versionsForTask: DatasetVersionRecord[];
  allowedBackbones: string[];
  currentFamilyInputSizeRule: FamilyInputSizeRule | null;
  allowedPresetSizes: number[];
  customInputAllowed: boolean;
  inputSizeHelpText: string | null;
  inputSizeIssue: ValidationIssue | null;
  embeddingEnabled: boolean;
  embeddingOutDim: number;
  embeddingNormalize: "none" | "l2";
  onnxEnabled: boolean;
  onnxOpset: number;
  dynamicBatch: boolean;
  dynamicHeightWidth: boolean;
};
