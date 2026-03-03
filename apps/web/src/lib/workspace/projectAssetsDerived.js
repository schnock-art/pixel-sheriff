const { folderChain } = require("./tree.js");

function buildVisibleTreeEntries(treeEntries, collapsedFolders) {
  function isHiddenByCollapsedAncestor(entry) {
    const parentPath = entry.kind === "folder" ? entry.path.split("/").slice(0, -1).join("/") : entry.folderPath ?? "";
    if (!parentPath) return false;
    for (const ancestor of folderChain(parentPath)) {
      if (collapsedFolders[ancestor]) return true;
    }
    return false;
  }

  return treeEntries.filter((entry) => !isHiddenByCollapsedAncestor(entry));
}

function buildAssetReviewStateById({ orderedAssetRows, pendingAnnotations, annotationByAssetId }) {
  const map = new Map();
  for (const asset of orderedAssetRows) {
    const pending = pendingAnnotations[asset.id];
    if (pending) {
      const isLabeled = pending.status !== "unlabeled" || pending.objects.length > 0;
      map.set(asset.id, { status: isLabeled ? "labeled" : "unlabeled", isDirty: true });
      continue;
    }

    const annotation = annotationByAssetId.get(asset.id);
    const isLabeled = Boolean(annotation && annotation.status !== "unlabeled");
    map.set(asset.id, { status: isLabeled ? "labeled" : "unlabeled", isDirty: false });
  }

  return map;
}

function buildFolderReviewStatusByPath({ folderAssetIds, assetReviewStateById }) {
  const status = {};
  for (const [folderPath, assetIds] of Object.entries(folderAssetIds)) {
    if (assetIds.length === 0) {
      status[folderPath] = "empty";
      continue;
    }
    const hasUnlabeled = assetIds.some((assetId) => (assetReviewStateById.get(assetId)?.status ?? "unlabeled") === "unlabeled");
    status[folderPath] = hasUnlabeled ? "has_unlabeled" : "all_labeled";
  }
  return status;
}

function buildFolderDirtyByPath({ folderAssetIds, assetReviewStateById }) {
  const flags = {};
  for (const [folderPath, assetIds] of Object.entries(folderAssetIds)) {
    flags[folderPath] = assetIds.some((assetId) => Boolean(assetReviewStateById.get(assetId)?.isDirty));
  }
  return flags;
}

function deriveMessageTone(message) {
  if (!message) return "info";
  const lower = message.toLowerCase();
  if (lower.includes("failed") || lower.includes("error")) return "error";
  return "success";
}

module.exports = {
  buildVisibleTreeEntries,
  buildAssetReviewStateById,
  buildFolderReviewStatusByPath,
  buildFolderDirtyByPath,
  deriveMessageTone,
};
