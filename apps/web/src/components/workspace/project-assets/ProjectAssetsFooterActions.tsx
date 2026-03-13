interface ProjectAssetsFooterActionsProps {
  isImporting: boolean;
  selectedProjectId: string | null;
  isDeletingAssets: boolean;
  hasCurrentAsset: boolean;
  bulkDeleteMode: boolean;
  selectedDeleteAssetIdsLength: number;
  selectedTreeFolderPath: string | null;
  selectedFolderAssetCount: number;
  isDeletingProject: boolean;
  isCreatingModel: boolean;
  onImport: () => void;
  onDeleteCurrentAsset: () => void;
  onToggleBulkDeleteMode: () => void;
  onDeleteSelectedAssets: () => void;
  onDeleteSelectedFolder: () => void;
  onDeleteCurrentProject: () => void;
  onBuildModel: () => void;
}

export function ProjectAssetsFooterActions({
  isImporting,
  selectedProjectId,
  isDeletingAssets,
  hasCurrentAsset,
  bulkDeleteMode,
  selectedDeleteAssetIdsLength,
  selectedTreeFolderPath,
  selectedFolderAssetCount,
  isDeletingProject,
  isCreatingModel,
  onImport,
  onDeleteCurrentAsset,
  onToggleBulkDeleteMode,
  onDeleteSelectedAssets,
  onDeleteSelectedFolder,
  onDeleteCurrentProject,
  onBuildModel,
}: ProjectAssetsFooterActionsProps) {
  return (
    <footer className="workspace-footer">
      <div className="footer-left">
        <button type="button" className="ghost-button" onClick={onImport} disabled={isImporting}>
          {isImporting ? "Importing..." : "Import"}
        </button>
        <button
          type="button"
          className="ghost-button danger-button"
          onClick={onDeleteCurrentAsset}
          disabled={isDeletingAssets || !selectedProjectId || !hasCurrentAsset}
        >
          {isDeletingAssets ? "Removing..." : "Remove Image"}
        </button>
        <button
          type="button"
          className={bulkDeleteMode ? "ghost-button active-toggle" : "ghost-button"}
          onClick={onToggleBulkDeleteMode}
          disabled={!selectedProjectId || isDeletingAssets}
        >
          {bulkDeleteMode ? "Exit Multi-delete" : "Multi-delete"}
        </button>
        <button
          type="button"
          className="ghost-button danger-button"
          onClick={onDeleteSelectedAssets}
          disabled={isDeletingAssets || selectedDeleteAssetIdsLength === 0}
        >
          {isDeletingAssets ? "Removing..." : `Delete Selected (${selectedDeleteAssetIdsLength})`}
        </button>
        <button
          type="button"
          className="ghost-button danger-button"
          onClick={onDeleteSelectedFolder}
          disabled={isDeletingAssets || !selectedTreeFolderPath}
        >
          Delete Folder
        </button>
        <button
          type="button"
          className="ghost-button danger-button"
          onClick={onDeleteCurrentProject}
          disabled={isDeletingProject || !selectedProjectId}
        >
          {isDeletingProject ? "Deleting..." : "Delete Project"}
        </button>
        <button type="button" className="primary-button" onClick={onBuildModel} disabled={!selectedProjectId || isCreatingModel}>
          {isCreatingModel ? "Building..." : "Build Model"}
        </button>
      </div>
    </footer>
  );
}
