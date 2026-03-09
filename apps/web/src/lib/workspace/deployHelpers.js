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

function detectionBoxesToGeometryObjects(boxes) {
  if (!Array.isArray(boxes)) return [];
  return boxes
    .filter((row) => row && typeof row === "object" && Array.isArray(row.bbox) && row.bbox.length === 4 && typeof row.class_id === "string")
    .map((row, index) => ({
      id: `suggested-bbox-${index + 1}`,
      kind: "bbox",
      category_id: row.class_id,
      bbox: row.bbox.map((value) => Number(value)),
    }));
}

function suggestionsPanelState({ hasActiveDeployment, isSuggesting, predictions }) {
  if (!hasActiveDeployment) return "cta";
  if (isSuggesting) return "loading";
  if (Array.isArray(predictions) && predictions.length > 0) return "ready";
  return "empty";
}

module.exports = {
  buildPredictPayload,
  deploymentTaskForExperiment,
  detectionBoxesToGeometryObjects,
  deviceLabelToPreference,
  suggestionsPanelState,
};
