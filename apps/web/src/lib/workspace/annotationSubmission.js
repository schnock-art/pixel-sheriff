const { deriveNextAnnotationStatus, normalizeAnnotationObjects, normalizeImageBasis, normalizeLabelIds } = require("./annotationState.js");

function resolveActiveSelection(labelIds, activeLabelRows) {
  const activeLabelIds = new Set(activeLabelRows.map((label) => label.id));
  return normalizeLabelIds(labelIds.filter((id) => activeLabelIds.has(id)));
}

function resolveLabelIdsForPayload(selectedLabelIds, activeLabelRows, normalizedObjects) {
  const selectedActiveIds = resolveActiveSelection(selectedLabelIds, activeLabelRows);
  if (selectedActiveIds.length > 0) return selectedActiveIds;

  // In geometry modes we can still infer class IDs from object categories,
  // even when explicit label selection state is empty or stale.
  const objectCategoryIds = normalizeLabelIds(normalizedObjects.map((objectValue) => objectValue.category_id));
  if (objectCategoryIds.length > 0) return objectCategoryIds;

  return [];
}

function buildClassificationPayload(assetId, selectedLabelIds, activeLabelRows, objects = [], imageBasis = null) {
  const normalizedObjects = normalizeAnnotationObjects(objects);
  const resolvedLabelIds = resolveLabelIdsForPayload(selectedLabelIds, activeLabelRows, normalizedObjects);
  const normalizedImageBasis = normalizeImageBasis(imageBasis);
  const primaryCategoryId = resolvedLabelIds[0] ?? null;
  const isUnlabeledSelection = resolvedLabelIds.length === 0 && normalizedObjects.length === 0;
  if (isUnlabeledSelection) {
    return {
      isUnlabeledSelection: true,
      selectedLabelIds: resolvedLabelIds,
      objects: normalizedObjects,
      payload_json: {
        version: "2.0",
        type: "classification",
        category_id: primaryCategoryId,
        category_ids: [],
        classification: {
          category_ids: [],
          primary_category_id: null,
        },
        objects: normalizedObjects,
        image_basis: normalizedImageBasis,
        coco: { image_id: assetId, category_id: null },
        source: "web-ui",
      },
    };
  }

  const selectedLabel = activeLabelRows.find((label) => label.id === resolvedLabelIds[0]);

  return {
    isUnlabeledSelection: false,
    selectedLabelIds: resolvedLabelIds,
    objects: normalizedObjects,
    payload_json: {
      version: "2.0",
      type: "classification",
      category_id: primaryCategoryId,
      category_ids: resolvedLabelIds,
      classification: {
        category_ids: resolvedLabelIds,
        primary_category_id: primaryCategoryId,
      },
      objects: normalizedObjects,
      image_basis: normalizedImageBasis,
      ...(selectedLabel ? { category_name: selectedLabel.name } : {}),
      coco: { image_id: assetId, category_id: primaryCategoryId },
      source: "web-ui",
    },
  };
}

function buildAnnotationUpsertInput({ assetId, currentStatus, selectedLabelIds, activeLabelRows, objects = [], imageBasis = null }) {
  const payload = buildClassificationPayload(assetId, selectedLabelIds, activeLabelRows, objects, imageBasis);
  if (!payload) return null;
  return {
    status: deriveNextAnnotationStatus(currentStatus, payload.selectedLabelIds, payload.objects.length),
    payload_json: payload.payload_json,
    isUnlabeledSelection: payload.isUnlabeledSelection,
  };
}

module.exports = {
  resolveActiveSelection,
  buildClassificationPayload,
  buildAnnotationUpsertInput,
};
