const { readAnnotationLabelIds } = require("./annotationState.js");

function resolveSelectionForAsset({ currentAssetId, pendingAnnotations, annotationByAssetId }) {
  if (!currentAssetId) {
    return { labelIds: [], status: "unlabeled", source: "empty" };
  }

  const pending = pendingAnnotations[currentAssetId];
  if (pending) {
    return {
      labelIds: pending.labelIds,
      status: pending.status,
      source: "pending",
    };
  }

  const annotation = annotationByAssetId.get(currentAssetId);
  if (!annotation) {
    return { labelIds: [], status: "unlabeled", source: "empty" };
  }

  return {
    labelIds: readAnnotationLabelIds(annotation.payload_json),
    status: annotation.status,
    source: "committed",
  };
}

module.exports = {
  resolveSelectionForAsset,
};
