const { deriveNextAnnotationStatus, normalizeLabelIds } = require("./annotationState.js");

function resolveActiveSelection(labelIds, activeLabelRows) {
  const activeLabelIds = new Set(activeLabelRows.map((label) => label.id));
  return normalizeLabelIds(labelIds.filter((id) => activeLabelIds.has(id)));
}

function buildClassificationPayload(assetId, selectedLabelIds, activeLabelRows) {
  const resolvedLabelIds = resolveActiveSelection(selectedLabelIds, activeLabelRows);
  const isUnlabeledSelection = resolvedLabelIds.length === 0;
  if (isUnlabeledSelection) {
    return {
      isUnlabeledSelection: true,
      selectedLabelIds: resolvedLabelIds,
      payload_json: {
        type: "classification",
        category_ids: [],
        coco: { image_id: assetId, category_id: null },
        source: "web-ui",
      },
    };
  }

  const selectedLabel = activeLabelRows.find((label) => label.id === resolvedLabelIds[0]);
  if (!selectedLabel) return null;

  return {
    isUnlabeledSelection: false,
    selectedLabelIds: resolvedLabelIds,
    payload_json: {
      type: "classification",
      category_id: selectedLabel.id,
      category_ids: resolvedLabelIds,
      category_name: selectedLabel.name,
      coco: { image_id: assetId, category_id: selectedLabel.id },
      source: "web-ui",
    },
  };
}

function buildAnnotationUpsertInput({ assetId, currentStatus, selectedLabelIds, activeLabelRows }) {
  const payload = buildClassificationPayload(assetId, selectedLabelIds, activeLabelRows);
  if (!payload) return null;
  return {
    status: deriveNextAnnotationStatus(currentStatus, payload.selectedLabelIds),
    payload_json: payload.payload_json,
    isUnlabeledSelection: payload.isUnlabeledSelection,
  };
}

module.exports = {
  resolveActiveSelection,
  buildClassificationPayload,
  buildAnnotationUpsertInput,
};
