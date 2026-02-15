function normalizeLabelIds(labelIds) {
  const normalized = [];
  const seen = new Set();
  for (const value of labelIds) {
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    if (seen.has(value)) continue;
    seen.add(value);
    normalized.push(value);
  }
  return normalized;
}

function readAnnotationLabelIds(payload) {
  if (!payload || typeof payload !== "object") return [];

  const categoryIds = payload.category_ids;
  if (Array.isArray(categoryIds)) {
    return normalizeLabelIds(categoryIds);
  }

  const categoryId = payload.category_id;
  if (typeof categoryId === "number" && Number.isFinite(categoryId)) {
    return [categoryId];
  }

  return [];
}

function deriveNextAnnotationStatus(currentStatus, labelIds) {
  if (labelIds.length === 0) return "unlabeled";
  return currentStatus === "unlabeled" ? "labeled" : currentStatus;
}

function comparableLabelIds(labelIds) {
  return normalizeLabelIds(labelIds)
    .slice()
    .sort((a, b) => a - b);
}

function areSelectionStatesEqual(left, right) {
  if (left.status !== right.status) return false;

  const leftIds = comparableLabelIds(left.labelIds);
  const rightIds = comparableLabelIds(right.labelIds);
  if (leftIds.length !== rightIds.length) return false;

  for (let index = 0; index < leftIds.length; index += 1) {
    if (leftIds[index] !== rightIds[index]) return false;
  }
  return true;
}

function getCommittedSelectionState(annotation) {
  if (!annotation) {
    return { labelIds: [], status: "unlabeled" };
  }

  return {
    labelIds: readAnnotationLabelIds(annotation.payload_json),
    status: annotation.status,
  };
}

function resolvePendingAnnotation(draftState, committedState) {
  if (areSelectionStatesEqual(draftState, committedState)) {
    return null;
  }

  return {
    labelIds: normalizeLabelIds(draftState.labelIds),
    status: draftState.status,
  };
}

function canSubmitWithStates(params) {
  const { pendingCount, editMode, hasCurrentAsset, draftState, committedState } = params;
  if (pendingCount > 0) return true;
  if (editMode) return false;
  if (!hasCurrentAsset) return false;
  return !areSelectionStatesEqual(draftState, committedState);
}

module.exports = {
  normalizeLabelIds,
  readAnnotationLabelIds,
  deriveNextAnnotationStatus,
  areSelectionStatesEqual,
  getCommittedSelectionState,
  resolvePendingAnnotation,
  canSubmitWithStates,
};
