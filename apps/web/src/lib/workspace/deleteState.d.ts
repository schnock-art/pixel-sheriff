export type DeleteSelectionState = Record<string, boolean>;

export function pruneSelectedDeleteAssets(
  selectedDeleteAssets: DeleteSelectionState,
  hasAsset: (assetId: string) => boolean,
): DeleteSelectionState;
export function toggleSelectedDeleteAsset(selectedDeleteAssets: DeleteSelectionState, assetId: string): DeleteSelectionState;
export function selectScopeDeleteAssets(assetRows: Array<{ id: string }>): DeleteSelectionState;
export function clearSelectedDeleteAssets(selectedDeleteAssets: DeleteSelectionState, assetIds: string[]): DeleteSelectionState;
export function shouldResetSelectedFolderAfterDeletion(selectedTreeFolderPath: string | null, folderPath: string): boolean;
export function pruneCollapsedFoldersForDeletedPath(collapsedFolders: Record<string, boolean>, folderPath: string): Record<string, boolean>;
