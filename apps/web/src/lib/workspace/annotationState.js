function normalizeLabelIds(labelIds) {
  const normalized = [];
  const seen = new Set();
  for (const value of labelIds) {
    let next = value;
    if (typeof next === "number" && Number.isFinite(next)) next = String(next);
    if (typeof next !== "string") continue;
    next = next.trim();
    if (!next || seen.has(next)) continue;
    seen.add(next);
    normalized.push(next);
  }
  return normalized;
}

function readAnnotationLabelIds(payload) {
  if (!payload || typeof payload !== "object") return [];

  const classification = payload.classification;
  if (classification && typeof classification === "object") {
    const classificationIds = classification.category_ids;
    if (Array.isArray(classificationIds)) {
      return normalizeLabelIds(classificationIds);
    }

    const primaryCategoryId = classification.primary_category_id;
    if (typeof primaryCategoryId === "number" && Number.isFinite(primaryCategoryId)) {
      return [String(primaryCategoryId)];
    }
    if (typeof primaryCategoryId === "string" && primaryCategoryId.trim() !== "") {
      return [primaryCategoryId.trim()];
    }
  }

  const categoryIds = payload.category_ids;
  if (Array.isArray(categoryIds)) {
    return normalizeLabelIds(categoryIds);
  }

  const categoryId = payload.category_id;
  if (typeof categoryId === "number" && Number.isFinite(categoryId)) {
    return [String(categoryId)];
  }
  if (typeof categoryId === "string" && categoryId.trim() !== "") {
    return [categoryId.trim()];
  }

  return [];
}

function normalizeImageBasis(imageBasis) {
  if (!imageBasis || typeof imageBasis !== "object") return null;
  const width = imageBasis.width;
  const height = imageBasis.height;
  if (typeof width !== "number" || !Number.isFinite(width) || width <= 0) return null;
  if (typeof height !== "number" || !Number.isFinite(height) || height <= 0) return null;
  return { width: Math.round(width), height: Math.round(height) };
}

function normalizeSegmentation(rawSegmentation) {
  if (!Array.isArray(rawSegmentation)) return [];
  const segments = [];
  for (const segment of rawSegmentation) {
    if (!Array.isArray(segment) || segment.length < 6 || segment.length % 2 !== 0) continue;
    const normalized = [];
    let valid = true;
    for (const value of segment) {
      if (typeof value !== "number" || !Number.isFinite(value)) {
        valid = false;
        break;
      }
      normalized.push(value);
    }
    if (!valid) continue;
    segments.push(normalized);
  }
  return segments;
}

function normalizeProvenance(rawProvenance) {
  if (!rawProvenance || typeof rawProvenance !== "object") return null;
  const originKind = typeof rawProvenance.origin_kind === "string" ? rawProvenance.origin_kind.trim() : "";
  if (!originKind) return null;
  const normalized = { origin_kind: originKind };
  for (const fieldName of ["session_id", "proposal_id", "source_model", "prompt_text", "review_decision"]) {
    const value = rawProvenance[fieldName];
    if (typeof value === "string" && value.trim() !== "") normalized[fieldName] = value.trim();
  }
  if (typeof rawProvenance.confidence === "number" && Number.isFinite(rawProvenance.confidence)) {
    normalized.confidence = rawProvenance.confidence;
  }
  return normalized;
}

function normalizeAnnotationObjects(objects) {
  if (!Array.isArray(objects)) return [];
  const result = [];
  const seenIds = new Set();

  for (const item of objects) {
    if (!item || typeof item !== "object") continue;
    const id = typeof item.id === "string" && item.id.trim() !== "" ? item.id : null;
    if (!id || seenIds.has(id)) continue;
    const kind = item.kind;
    let categoryId = item.category_id;
    if (typeof categoryId === "number" && Number.isFinite(categoryId)) categoryId = String(categoryId);
    if (typeof categoryId !== "string" || categoryId.trim() === "") continue;
    categoryId = categoryId.trim();

    if (kind === "bbox") {
      const bbox = item.bbox;
      if (!Array.isArray(bbox) || bbox.length !== 4) continue;
      const normalized = bbox.map((value) => (typeof value === "number" && Number.isFinite(value) ? value : NaN));
      if (normalized.some((value) => Number.isNaN(value))) continue;
      if (normalized[2] <= 0 || normalized[3] <= 0) continue;
      seenIds.add(id);
      result.push({
        id,
        kind: "bbox",
        category_id: categoryId,
        bbox: normalized,
        ...(normalizeProvenance(item.provenance) ? { provenance: normalizeProvenance(item.provenance) } : {}),
      });
      continue;
    }

    if (kind === "polygon") {
      const segmentation = normalizeSegmentation(item.segmentation);
      if (segmentation.length === 0) continue;
      seenIds.add(id);
      result.push({
        id,
        kind: "polygon",
        category_id: categoryId,
        segmentation,
        ...(normalizeProvenance(item.provenance) ? { provenance: normalizeProvenance(item.provenance) } : {}),
      });
    }
  }

  return result;
}

function readAnnotationObjects(payload) {
  if (!payload || typeof payload !== "object") return [];
  return normalizeAnnotationObjects(payload.objects);
}

function readAnnotationImageBasis(payload) {
  if (!payload || typeof payload !== "object") return null;
  return normalizeImageBasis(payload.image_basis);
}

function deriveNextAnnotationStatus(currentStatus, labelIds, objectCount = 0) {
  if (labelIds.length === 0 && objectCount === 0) return "unlabeled";
  return currentStatus === "unlabeled" ? "labeled" : currentStatus;
}

function comparableLabelIds(labelIds) {
  return normalizeLabelIds(labelIds)
    .slice()
    .sort();
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

function comparableObjectValue(objectValue) {
  if (!objectValue || typeof objectValue !== "object") return "";
  if (objectValue.kind === "bbox" && Array.isArray(objectValue.bbox)) {
    return JSON.stringify({
      id: objectValue.id,
      kind: "bbox",
      category_id: objectValue.category_id,
      bbox: objectValue.bbox.map((value) => Number(value.toFixed(6))),
      provenance: normalizeProvenance(objectValue.provenance),
    });
  }
  if (objectValue.kind === "polygon" && Array.isArray(objectValue.segmentation)) {
    return JSON.stringify({
      id: objectValue.id,
      kind: "polygon",
      category_id: objectValue.category_id,
      segmentation: objectValue.segmentation.map((segment) => segment.map((value) => Number(value.toFixed(6)))),
      provenance: normalizeProvenance(objectValue.provenance),
    });
  }
  return "";
}

function areGeometryStatesEqual(leftObjects, rightObjects) {
  const leftComparable = normalizeAnnotationObjects(leftObjects)
    .map(comparableObjectValue)
    .sort();
  const rightComparable = normalizeAnnotationObjects(rightObjects)
    .map(comparableObjectValue)
    .sort();

  if (leftComparable.length !== rightComparable.length) return false;
  for (let index = 0; index < leftComparable.length; index += 1) {
    if (leftComparable[index] !== rightComparable[index]) return false;
  }
  return true;
}

function areImageBasisEqual(leftImageBasis, rightImageBasis) {
  const left = normalizeImageBasis(leftImageBasis);
  const right = normalizeImageBasis(rightImageBasis);
  if (!left && !right) return true;
  if (!left || !right) return false;
  return left.width === right.width && left.height === right.height;
}

function getCommittedSelectionState(annotation) {
  if (!annotation) {
    return { labelIds: [], status: "unlabeled", objects: [], imageBasis: null };
  }

  return {
    labelIds: readAnnotationLabelIds(annotation.payload_json),
    status: annotation.status,
    objects: readAnnotationObjects(annotation.payload_json),
    imageBasis: readAnnotationImageBasis(annotation.payload_json),
  };
}

function resolvePendingAnnotation(draftState, committedState) {
  const normalizedDraftState = {
    labelIds: normalizeLabelIds(draftState.labelIds),
    status: draftState.status,
    objects: normalizeAnnotationObjects(draftState.objects),
    imageBasis: normalizeImageBasis(draftState.imageBasis),
  };
  const normalizedCommittedState = {
    labelIds: normalizeLabelIds(committedState.labelIds),
    status: committedState.status,
    objects: normalizeAnnotationObjects(committedState.objects),
    imageBasis: normalizeImageBasis(committedState.imageBasis),
  };

  const selectionEqual = areSelectionStatesEqual(normalizedDraftState, normalizedCommittedState);
  const geometryEqual = areGeometryStatesEqual(normalizedDraftState.objects, normalizedCommittedState.objects);
  // imageBasis is auxiliary metadata that updates automatically when the image loads.
  // It must not drive the dirty/staged state on its own — only label/object changes matter.
  if (selectionEqual && geometryEqual) {
    return null;
  }

  return normalizedDraftState;
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
  normalizeAnnotationObjects,
  normalizeImageBasis,
  readAnnotationLabelIds,
  readAnnotationObjects,
  readAnnotationImageBasis,
  deriveNextAnnotationStatus,
  areSelectionStatesEqual,
  areGeometryStatesEqual,
  getCommittedSelectionState,
  resolvePendingAnnotation,
  canSubmitWithStates,
};
