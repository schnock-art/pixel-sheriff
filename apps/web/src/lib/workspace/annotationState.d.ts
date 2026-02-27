import type { AnnotationStatus } from "../api";

export interface GeometryBBoxObject {
  id: string;
  kind: "bbox";
  category_id: number;
  bbox: number[];
}

export interface GeometryPolygonObject {
  id: string;
  kind: "polygon";
  category_id: number;
  segmentation: number[][];
}

export type GeometryObject = GeometryBBoxObject | GeometryPolygonObject;

export interface ImageBasis {
  width: number;
  height: number;
}

export interface SelectionState {
  labelIds: number[];
  status: AnnotationStatus;
  objects?: GeometryObject[];
  imageBasis?: ImageBasis | null;
}

export interface AnnotationLike {
  status: AnnotationStatus;
  payload_json: Record<string, unknown>;
}

export function normalizeLabelIds(labelIds: number[]): number[];
export function normalizeAnnotationObjects(objects: unknown): GeometryObject[];
export function normalizeImageBasis(imageBasis: unknown): ImageBasis | null;
export function readAnnotationLabelIds(payload: Record<string, unknown>): number[];
export function readAnnotationObjects(payload: Record<string, unknown>): GeometryObject[];
export function readAnnotationImageBasis(payload: Record<string, unknown>): ImageBasis | null;
export function deriveNextAnnotationStatus(currentStatus: AnnotationStatus, labelIds: number[], objectCount?: number): AnnotationStatus;
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
