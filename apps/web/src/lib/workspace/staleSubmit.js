function isAnnotationSubmitNotFoundError(error, projectId) {
  if (!error || typeof error !== "object") return false;
  const status = error.status;
  const url = typeof error.url === "string" ? error.url : "";
  if (status !== 404) return false;
  if (!projectId || typeof projectId !== "string") return false;
  return url.includes(`/api/v1/projects/${projectId}/annotations`);
}

function prunePendingAnnotationsForKnownAssets(pendingAnnotations, knownAssetIds) {
  const known = new Set(Array.isArray(knownAssetIds) ? knownAssetIds : []);
  const next = {};
  const removedAssetIds = [];
  for (const [assetId, pending] of Object.entries(pendingAnnotations || {})) {
    if (!known.has(assetId)) {
      removedAssetIds.push(assetId);
      continue;
    }
    next[assetId] = pending;
  }
  return { nextPendingAnnotations: next, removedAssetIds };
}

module.exports = {
  isAnnotationSubmitNotFoundError,
  prunePendingAnnotationsForKnownAssets,
};

