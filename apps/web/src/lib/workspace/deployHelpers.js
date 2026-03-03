function deviceLabelToPreference(label) {
  const normalized = String(label || "").trim().toLowerCase();
  if (normalized === "cuda") return "cuda";
  if (normalized === "cpu") return "cpu";
  return "auto";
}

function suggestionsPanelState({ hasActiveDeployment, isSuggesting, predictions }) {
  if (!hasActiveDeployment) return "cta";
  if (isSuggesting) return "loading";
  if (Array.isArray(predictions) && predictions.length > 0) return "ready";
  return "empty";
}

module.exports = {
  deviceLabelToPreference,
  suggestionsPanelState,
};
