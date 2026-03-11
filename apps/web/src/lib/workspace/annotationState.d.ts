import type { AnnotationStatus } from "../api";

export interface GeometryObjectProvenance {
  origin_kind: string;
  session_id?: string;
  proposal_id?: string;
  source_model?: string;
  prompt_text?: string;
  confidence?: number;
  review_decision?: string;
}

export interface GeometryBBoxObject {
  id: string;
  kind: "bbox";
  category_id: string;
  bbox: number[];
  provenance?: GeometryObjectProvenance;
}

export interface GeometryPolygonObject {
  id: string;
  kind: "polygon";
  category_id: string;
  segmentation: number[][];
  provenance?: GeometryObjectProvenance;
}

export type GeometryObject = GeometryBBoxObject | GeometryPolygonObject;

export interface ImageBasis {
  width: number;
  height: number;
}

export interface SelectionState {
  labelIds: string[];
  status: AnnotationStatus;
  objects?: GeometryObject[];
  imageBasis?: ImageBasis | null;
}

export interface AnnotationLike {
  status: AnnotationStatus;
  payload_json: Record<string, unknown>;
}

export function normalizeLabelIds(labelIds: Array<string | number>): string[];
export function normalizeAnnotationObjects(objects: unknown): GeometryObject[];
export function normalizeImageBasis(imageBasis: unknown): ImageBasis | null;
export function readAnnotationLabelIds(payload: Record<string, unknown>): string[];
export function readAnnotationObjects(payload: Record<string, unknown>): GeometryObject[];
export function readAnnotationImageBasis(payload: Record<string, unknown>): ImageBasis | null;
export function deriveNextAnnotationStatus(currentStatus: AnnotationStatus, labelIds: string[], objectCount?: number): AnnotationStatus;
export function areSelectionStatesEqual(left: SelectionState, right: SelectionState): boolean;
export function areGeometryStatesEqual(left: GeometryObject[], right: GeometryObject[]): boolean;
export function getCommittedSelectionState(annotation: AnnotationLike | null | undefined): SelectionState;
export function resolvePendingAnnotation(draftState: SelectionState, committedState: SelectionState): SelectionState | null;
export function canSubmitWithStates(params: {
  pendingCount: number;
  editMode: boolean;
  hasCurrentAsset: boolean;
  draftState: SelectionState;
  committedState: SelectionState;
}): boolean;
