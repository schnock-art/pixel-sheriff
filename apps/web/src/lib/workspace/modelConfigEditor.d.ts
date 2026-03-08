export interface EmbeddingAuxOutput {
  name: string;
  type: "embedding";
  source: {
    block: "backbone";
    tap: "avgpool";
  };
  projection: {
    type: "linear";
    out_dim: number;
    normalize: "none" | "l2";
  };
}

export function cloneModelConfig(config: Record<string, unknown> | null | undefined): Record<string, unknown>;
export function isModelConfigDirty(
  savedConfig: Record<string, unknown> | null | undefined,
  draftConfig: Record<string, unknown> | null | undefined,
): boolean;
export function createEmbeddingAuxOutput(): EmbeddingAuxOutput;
export function setEmbeddingAuxEnabled(config: Record<string, unknown>, enabled: boolean): Record<string, unknown>;
export function setEmbeddingProjection(
  config: Record<string, unknown>,
  outDim: number,
  normalize: "none" | "l2",
): Record<string, unknown>;
export function setSquareInputSize(config: Record<string, unknown>, size: number): Record<string, unknown>;
export function setDynamicShapeFlags(config: Record<string, unknown>, batch: boolean, heightWidth: boolean): Record<string, unknown>;
export interface DatasetVersionSummary {
  id?: string;
  manifest_id: string;
  task?: string;
  label_mode?: "single_label" | "multi_label" | null;
  num_classes: number;
  class_order: string[];
  class_names: Record<string, string> | string[];
}
export interface FamiliesMetadata {
  schema_version: string;
  families: Array<{
    name: string;
    task: string;
    allowed_backbones: string[];
    input_size?: {
      shape?: string;
      mode?: string;
      min_square_size?: number;
      step?: number;
      recommended_square_size?: number;
      required_square_size?: number;
    };
  }>;
}
export function setSourceDataset(config: Record<string, unknown>, datasetVersionSummary: DatasetVersionSummary): Record<string, unknown>;
export function setArchitectureFamily(config: Record<string, unknown>, familyName: string, familiesMetadata: FamiliesMetadata): Record<string, unknown>;
export function setBackbone(config: Record<string, unknown>, backboneName: string): Record<string, unknown>;
