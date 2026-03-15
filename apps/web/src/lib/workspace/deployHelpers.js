function deviceLabelToPreference(label) {
  const normalized = String(label || "").trim().toLowerCase();
  if (normalized === "cuda") return "cuda";
  if (normalized === "cpu") return "cpu";
  return "auto";
}

function deploymentTaskForExperiment(task) {
  const normalized = String(task || "").trim().toLowerCase();
  if (normalized === "detection" || normalized === "bbox") return "bbox";
  return "classification";
}

function buildPredictPayload({ assetId, deploymentId = null, task, scoreThreshold }) {
  const payload = { asset_id: assetId, deployment_id: deploymentId };
  if (task === "bbox") {
    payload.score_threshold = scoreThreshold;
    return payload;
  }
  payload.top_k = 5;
  return payload;
}

function buildPredictBatchPayload({ assetIds, deploymentId = null, task, scoreThreshold }) {
  const payload = { asset_ids: Array.isArray(assetIds) ? assetIds.filter((value) => typeof value === "string" && value.trim()) : [], deployment_id: deploymentId };
  if (task === "bbox") {
    payload.score_threshold = scoreThreshold;
    return payload;
  }
  payload.top_k = 5;
  return payload;
}

function detectionBoxesToPreviewObjects(boxes) {
  if (!Array.isArray(boxes)) return [];
  return boxes
    .filter((row) => row && typeof row === "object" && Array.isArray(row.bbox) && row.bbox.length === 4 && typeof row.class_id === "string")
    .map((row, index) => ({
      id:
        typeof row.review_item_id === "string" && row.review_item_id.trim()
          ? row.review_item_id.trim()
          : `deployment-prediction-${index + 1}`,
      category_id: row.class_id,
      bbox: row.bbox.map((value) => Number(value)),
      label_text: typeof row.class_name === "string" && row.class_name.trim() ? row.class_name.trim() : `#${row.class_id}`,
      confidence: typeof row.score === "number" && Number.isFinite(row.score) ? row.score : 0,
    }));
}

function detectionBoxesToGeometryObjects(boxes, options = {}) {
  if (!Array.isArray(boxes)) return [];
  const sourceModel =
    typeof options.sourceModel === "string" && options.sourceModel.trim() ? options.sourceModel.trim() : null;
  const reviewDecision =
    typeof options.reviewDecision === "string" && options.reviewDecision.trim() ? options.reviewDecision.trim() : null;
  const idPrefix = typeof options.idPrefix === "string" && options.idPrefix.trim() ? options.idPrefix.trim() : "deployment-prediction";

  return boxes
    .filter((row) => row && typeof row === "object" && Array.isArray(row.bbox) && row.bbox.length === 4 && typeof row.class_id === "string")
    .map((row, index) => {
      const provenance = {
        origin_kind: "deployment_prediction",
        ...(sourceModel ? { source_model: sourceModel } : {}),
        ...(typeof row.score === "number" && Number.isFinite(row.score) ? { confidence: row.score } : {}),
        ...(reviewDecision ? { review_decision: reviewDecision } : {}),
      };
      return {
        id:
          typeof row.review_item_id === "string" && row.review_item_id.trim()
            ? row.review_item_id.trim()
            : `${idPrefix}-${index + 1}`,
        kind: "bbox",
        category_id: row.class_id,
        bbox: row.bbox.map((value) => Number(value)),
        provenance,
      };
    });
}

function normalizePredictReview(response, options = {}) {
  if (!response || typeof response !== "object") return null;
  const assetId = typeof response.asset_id === "string" ? response.asset_id : "";
  const deploymentId = typeof response.deployment_id === "string" ? response.deployment_id : "";
  const deploymentName =
    typeof response.deployment_name === "string" && response.deployment_name.trim() ? response.deployment_name.trim() : null;
  const deviceSelected =
    typeof response.device_selected === "string" && response.device_selected.trim() ? response.device_selected.trim() : null;
  const devicePreference =
    typeof response.device_preference === "string" && response.device_preference.trim() ? response.device_preference.trim() : null;

  if (response.task === "bbox") {
    const items = Array.isArray(response.boxes)
      ? response.boxes
          .filter((row) => row && typeof row === "object" && typeof row.class_id === "string" && Array.isArray(row.bbox))
          .map((row, index) => ({
            review_item_id: `prediction-bbox-${index + 1}`,
            class_index: Number(row.class_index ?? index),
            class_id: row.class_id,
            class_name: typeof row.class_name === "string" ? row.class_name : `#${row.class_id}`,
            score: typeof row.score === "number" && Number.isFinite(row.score) ? row.score : 0,
            bbox: row.bbox.map((value) => Number(value)),
          }))
      : [];
    return {
      task: "bbox",
      asset_id: assetId,
      deployment_id: deploymentId,
      deployment_name: deploymentName,
      device_selected: deviceSelected,
      device_preference: devicePreference,
      score_threshold: typeof options.scoreThreshold === "number" && Number.isFinite(options.scoreThreshold) ? options.scoreThreshold : null,
      items,
      preview_objects: detectionBoxesToPreviewObjects(items),
    };
  }

  if (response.task === "classification") {
    const items = Array.isArray(response.predictions)
      ? response.predictions
          .filter((row) => row && typeof row === "object" && typeof row.class_id === "string")
          .map((row) => ({
            review_item_id: `prediction-class-${row.class_id}`,
            class_index: Number(row.class_index ?? 0),
            class_id: row.class_id,
            class_name: typeof row.class_name === "string" ? row.class_name : `#${row.class_id}`,
            score: typeof row.score === "number" && Number.isFinite(row.score) ? row.score : 0,
          }))
      : [];
    return {
      task: "classification",
      asset_id: assetId,
      deployment_id: deploymentId,
      deployment_name: deploymentName,
      device_selected: deviceSelected,
      device_preference: devicePreference,
      items,
    };
  }

  return null;
}

function resolveDefaultReviewItemId(review) {
  if (!review || typeof review !== "object" || !Array.isArray(review.items) || review.items.length === 0) return null;
  const first = review.items[0];
  return typeof first.review_item_id === "string" ? first.review_item_id : null;
}

function buildAcceptedPredictionReview(review, selectedReviewItemId = null) {
  if (!review || typeof review !== "object") return null;

  if (review.task === "classification") {
    const items = Array.isArray(review.items) ? review.items : [];
    const selected =
      items.find((item) => item && item.review_item_id === selectedReviewItemId) ??
      items[0] ??
      null;
    if (!selected) return null;
    return {
      task: "classification",
      categoryId: selected.class_id,
      predictionReview: {
        origin_kind: "deployment_prediction",
        task: "classification",
        deployment_id: review.deployment_id,
        ...(typeof review.deployment_name === "string" && review.deployment_name.trim()
          ? { deployment_name: review.deployment_name.trim() }
          : {}),
        ...(typeof review.device_selected === "string" && review.device_selected.trim()
          ? { device_selected: review.device_selected.trim() }
          : {}),
        ...(typeof review.device_preference === "string" && review.device_preference.trim()
          ? { device_preference: review.device_preference.trim() }
          : {}),
        selected_class_id: selected.class_id,
        ...(typeof selected.class_name === "string" && selected.class_name.trim()
          ? { selected_class_name: selected.class_name.trim() }
          : {}),
        ...(typeof selected.score === "number" && Number.isFinite(selected.score) ? { score: selected.score } : {}),
      },
    };
  }

  if (review.task === "bbox") {
    return {
      task: "bbox",
      objects: detectionBoxesToGeometryObjects(review.items, {
        sourceModel: review.deployment_name || review.deployment_id,
        reviewDecision: "accepted",
        idPrefix: "deployment-prediction",
      }),
    };
  }

  return null;
}

module.exports = {
  buildPredictBatchPayload,
  buildPredictPayload,
  buildAcceptedPredictionReview,
  detectionBoxesToPreviewObjects,
  deploymentTaskForExperiment,
  detectionBoxesToGeometryObjects,
  deviceLabelToPreference,
  normalizePredictReview,
  resolveDefaultReviewItemId,
};
