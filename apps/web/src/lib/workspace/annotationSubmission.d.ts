import type { AnnotationStatus } from "../api";

export interface ActiveLabelRow {
  id: number;
  name: string;
}

export interface AnnotationUpsertInput {
  status: AnnotationStatus;
  payload_json: Record<string, unknown>;
  isUnlabeledSelection: boolean;
}

export function resolveActiveSelection(labelIds: number[], activeLabelRows: ActiveLabelRow[]): number[];
export function buildClassificationPayload(
  assetId: string,
  selectedLabelIds: number[],
  activeLabelRows: ActiveLabelRow[],
):
  | {
      isUnlabeledSelection: boolean;
      selectedLabelIds: number[];
      payload_json: Record<string, unknown>;
    }
  | null;
export function buildAnnotationUpsertInput(params: {
  assetId: string;
  currentStatus: AnnotationStatus;
  selectedLabelIds: number[];
  activeLabelRows: ActiveLabelRow[];
}): AnnotationUpsertInput | null;
