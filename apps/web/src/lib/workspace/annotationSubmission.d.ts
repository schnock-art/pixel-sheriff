import type { AnnotationStatus } from "../api";
import type { GeometryObject, ImageBasis } from "./annotationState";

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
  objects?: GeometryObject[],
  imageBasis?: ImageBasis | null,
):
  | {
      isUnlabeledSelection: boolean;
      selectedLabelIds: number[];
      objects: GeometryObject[];
      payload_json: Record<string, unknown>;
    }
  | null;
export function buildAnnotationUpsertInput(params: {
  assetId: string;
  currentStatus: AnnotationStatus;
  selectedLabelIds: number[];
  activeLabelRows: ActiveLabelRow[];
  objects?: GeometryObject[];
  imageBasis?: ImageBasis | null;
}): AnnotationUpsertInput | null;
