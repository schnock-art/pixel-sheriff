import type { AnnotationStatus } from "../api";
import type { GeometryObject, ImageBasis } from "./annotationState";

export interface ActiveLabelRow {
  id: string;
  name: string;
}

export interface AnnotationUpsertInput {
  status: AnnotationStatus;
  payload_json: Record<string, unknown>;
  isUnlabeledSelection: boolean;
}

export function resolveActiveSelection(labelIds: string[], activeLabelRows: ActiveLabelRow[]): string[];
export function buildClassificationPayload(
  assetId: string,
  selectedLabelIds: string[],
  activeLabelRows: ActiveLabelRow[],
  objects?: GeometryObject[],
  imageBasis?: ImageBasis | null,
):
  | {
      isUnlabeledSelection: boolean;
      selectedLabelIds: string[];
      objects: GeometryObject[];
      payload_json: Record<string, unknown>;
    }
  | null;
export function buildAnnotationUpsertInput(params: {
  assetId: string;
  currentStatus: AnnotationStatus;
  selectedLabelIds: string[];
  activeLabelRows: ActiveLabelRow[];
  objects?: GeometryObject[];
  imageBasis?: ImageBasis | null;
}): AnnotationUpsertInput | null;
