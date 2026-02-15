import type { AnnotationStatus } from "../api";

export interface SelectionState {
  labelIds: number[];
  status: AnnotationStatus;
}

export interface AnnotationLike {
  status: AnnotationStatus;
  payload_json: Record<string, unknown>;
}

export function normalizeLabelIds(labelIds: number[]): number[];
export function readAnnotationLabelIds(payload: Record<string, unknown>): number[];
export function deriveNextAnnotationStatus(currentStatus: AnnotationStatus, labelIds: number[]): AnnotationStatus;
export function areSelectionStatesEqual(left: SelectionState, right: SelectionState): boolean;
export function getCommittedSelectionState(annotation: AnnotationLike | null | undefined): SelectionState;
export function resolvePendingAnnotation(draftState: SelectionState, committedState: SelectionState): SelectionState | null;
export function canSubmitWithStates(params: {
  pendingCount: number;
  editMode: boolean;
  hasCurrentAsset: boolean;
  draftState: SelectionState;
  committedState: SelectionState;
}): boolean;
