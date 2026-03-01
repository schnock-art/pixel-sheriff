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
