function pruneSelectedDeleteAssets(selectedDeleteAssets, hasAsset) {
  const next = {};
  for (const [assetId, isSelected] of Object.entries(selectedDeleteAssets)) {
    if (isSelected && hasAsset(assetId)) next[assetId] = true;
  }

  const previousKeys = Object.keys(selectedDeleteAssets);
  const nextKeys = Object.keys(next);
  if (previousKeys.length === nextKeys.length && previousKeys.every((key) => next[key] === selectedDeleteAssets[key])) {
    return selectedDeleteAssets;
  }
  return next;
}

function toggleSelectedDeleteAsset(selectedDeleteAssets, assetId) {
  const next = { ...selectedDeleteAssets };
  if (next[assetId]) delete next[assetId];
  else next[assetId] = true;
  return next;
}

function selectScopeDeleteAssets(assetRows) {
  return Object.fromEntries(assetRows.map((asset) => [asset.id, true]));
}

function clearSelectedDeleteAssets(selectedDeleteAssets, assetIds) {
  if (assetIds.length === 0) return selectedDeleteAssets;
  const next = { ...selectedDeleteAssets };
  for (const assetId of assetIds) delete next[assetId];
  return next;
}

function shouldResetSelectedFolderAfterDeletion(selectedTreeFolderPath, folderPath) {
  if (!selectedTreeFolderPath) return false;
  return selectedTreeFolderPath === folderPath || selectedTreeFolderPath.startsWith(`${folderPath}/`);
}

function pruneCollapsedFoldersForDeletedPath(collapsedFolders, folderPath) {
  const next = { ...collapsedFolders };
  for (const key of Object.keys(next)) {
    if (key === folderPath || key.startsWith(`${folderPath}/`)) {
      delete next[key];
    }
  }
  return next;
}

module.exports = {
  pruneSelectedDeleteAssets,
  toggleSelectedDeleteAsset,
  selectScopeDeleteAssets,
  clearSelectedDeleteAssets,
  shouldResetSelectedFolderAfterDeletion,
  pruneCollapsedFoldersForDeletedPath,
};
