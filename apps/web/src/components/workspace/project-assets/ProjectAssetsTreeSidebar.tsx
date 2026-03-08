import type { AnnotationStatus } from "../../../lib/api";
import { type TreeEntry } from "../../../lib/workspace/tree";

interface FilterLabelRow {
  id: string;
  name: string;
}

interface ProjectAssetsTreeSidebarProps {
  selectedTreeFolderPath: string | null;
  bulkDeleteMode: boolean;
  isDeletingAssets: boolean;
  selectedProjectId: string | null;
  selectedDeleteAssetIdsLength: number;
  selectedFolderAssetCount: number;
  visibleTreeEntries: TreeEntry[];
  collapsedFolders: Record<string, boolean>;
  folderReviewStatusByPath: Record<string, "all_labeled" | "has_unlabeled" | "empty">;
  folderDirtyByPath: Record<string, boolean>;
  selectedDeleteAssets: Record<string, boolean>;
  currentAssetId: string | null;
  assetReviewStateById: Map<string, { status: "labeled" | "unlabeled"; isDirty: boolean }>;
  filterStatus: "all" | AnnotationStatus;
  filterCategoryId: string;
  filterLabelRows: FilterLabelRow[];
  onCollapseAllFolders: () => void;
  onExpandAllFolders: () => void;
  onSelectFolderScope: (folderPath: string | null) => void;
  onToggleBulkDeleteMode: () => void;
  onSelectAllDeleteScope: () => void;
  onClearDeleteSelection: () => void;
  onDeleteSelectedAssets: () => void;
  onDeleteSelectedFolder: () => void;
  onToggleFolderCollapsed: (folderPath: string) => void;
  onDeleteFolderPath: (folderPath: string) => void;
  onToggleDeleteSelection: (assetId: string) => void;
  onSelectTreeAsset: (assetId: string, folderPath?: string) => void;
  onChangeFilterStatus: (status: "all" | AnnotationStatus) => void;
  onChangeFilterCategoryId: (categoryId: string) => void;
}

export function ProjectAssetsTreeSidebar({
  selectedTreeFolderPath,
  bulkDeleteMode,
  isDeletingAssets,
  selectedProjectId,
  selectedDeleteAssetIdsLength,
  selectedFolderAssetCount,
  visibleTreeEntries,
  collapsedFolders,
  folderReviewStatusByPath,
  folderDirtyByPath,
  selectedDeleteAssets,
  currentAssetId,
  assetReviewStateById,
  filterStatus,
  filterCategoryId,
  filterLabelRows,
  onCollapseAllFolders,
  onExpandAllFolders,
  onSelectFolderScope,
  onToggleBulkDeleteMode,
  onSelectAllDeleteScope,
  onClearDeleteSelection,
  onDeleteSelectedAssets,
  onDeleteSelectedFolder,
  onToggleFolderCollapsed,
  onDeleteFolderPath,
  onToggleDeleteSelection,
  onSelectTreeAsset,
  onChangeFilterStatus,
  onChangeFilterCategoryId,
}: ProjectAssetsTreeSidebarProps) {
  return (
    <aside className="workspace-sidebar">
      <section className="project-tree">
        <div className="project-tree-head">
          <h3>Files</h3>
          <div className="tree-head-actions">
            <button type="button" className="tree-scope-button" onClick={onCollapseAllFolders}>
              Collapse all
            </button>
            <button type="button" className="tree-scope-button" onClick={onExpandAllFolders}>
              Expand all
            </button>
            <button
              type="button"
              className={selectedTreeFolderPath === null ? "tree-scope-button active" : "tree-scope-button"}
              onClick={() => onSelectFolderScope(null)}
            >
              All files
            </button>
          </div>
        </div>
        <div className="tree-filter-toolbar">
          <select
            value={filterStatus}
            onChange={(event) => onChangeFilterStatus(event.target.value as "all" | AnnotationStatus)}
            className="tree-filter-select"
            aria-label="Filter by status"
          >
            <option value="all">All statuses</option>
            <option value="unlabeled">Unlabeled</option>
            <option value="labeled">Labeled</option>
            <option value="skipped">Skipped</option>
            <option value="needs_review">Needs review</option>
            <option value="approved">Approved</option>
          </select>
          <select
            value={filterCategoryId}
            onChange={(event) => onChangeFilterCategoryId(event.target.value)}
            className="tree-filter-select"
            aria-label="Filter by class"
            disabled={filterLabelRows.length === 0}
          >
            <option value="all">All classes</option>
            {filterLabelRows.map((label) => (
              <option key={label.id} value={label.id}>
                {label.name}
              </option>
            ))}
          </select>
        </div>
        <div className="tree-delete-toolbar">
          <button
            type="button"
            className={bulkDeleteMode ? "tree-scope-button danger active" : "tree-scope-button danger"}
            onClick={onToggleBulkDeleteMode}
            disabled={!selectedProjectId || isDeletingAssets}
          >
            {bulkDeleteMode ? "Exit multi-delete" : "Multi-delete"}
          </button>
          {bulkDeleteMode ? (
            <>
              <button type="button" className="tree-scope-button" onClick={onSelectAllDeleteScope} disabled={isDeletingAssets}>
                Select scope
              </button>
              <button type="button" className="tree-scope-button" onClick={onClearDeleteSelection} disabled={isDeletingAssets}>
                Clear
              </button>
              <button
                type="button"
                className="tree-scope-button danger"
                onClick={onDeleteSelectedAssets}
                disabled={isDeletingAssets || selectedDeleteAssetIdsLength === 0}
              >
                Delete selected ({selectedDeleteAssetIdsLength})
              </button>
            </>
          ) : null}
          {selectedTreeFolderPath ? (
            <button
              type="button"
              className="tree-scope-button danger"
              onClick={onDeleteSelectedFolder}
              disabled={isDeletingAssets || selectedFolderAssetCount === 0}
            >
              Delete folder ({selectedFolderAssetCount})
            </button>
          ) : null}
        </div>
        {selectedTreeFolderPath ? <p className="tree-scope-caption">Scope: {selectedTreeFolderPath}</p> : null}
        <ul>
          {visibleTreeEntries.map((entry) => (
            <li key={entry.key}>
              {entry.kind === "folder" ? (
                <div className="tree-folder-row" style={{ paddingLeft: `${entry.depth * 14 + 8}px` }}>
                  <button
                    type="button"
                    className="tree-folder-toggle"
                    aria-label={collapsedFolders[entry.path] ? "Expand folder" : "Collapse folder"}
                    onClick={() => onToggleFolderCollapsed(entry.path)}
                  >
                    {collapsedFolders[entry.path] ? ">" : "v"}
                  </button>
                  <button
                    type="button"
                    className={`tree-folder-button${selectedTreeFolderPath === entry.path ? " active" : ""} ${
                      folderReviewStatusByPath[entry.path] === "all_labeled"
                        ? "is-labeled"
                        : folderReviewStatusByPath[entry.path] === "has_unlabeled"
                          ? "has-unlabeled"
                          : "is-empty"
                    }${folderDirtyByPath[entry.path] ? " is-dirty" : ""}`}
                    onClick={() => onSelectFolderScope(entry.path)}
                    title={folderDirtyByPath[entry.path] ? `Folder "${entry.path}" has staged edits` : undefined}
                  >
                    {entry.name}
                  </button>
                  <button
                    type="button"
                    className="tree-row-delete"
                    onClick={() => void onDeleteFolderPath(entry.path)}
                    disabled={isDeletingAssets}
                    title={`Delete "${entry.path}"`}
                  >
                    x
                  </button>
                </div>
              ) : (
                <div className="tree-file-row" style={{ paddingLeft: `${entry.depth * 14 + 8}px` }}>
                  {bulkDeleteMode && entry.assetId ? (
                    <input
                      className="tree-file-checkbox"
                      type="checkbox"
                      checked={Boolean(selectedDeleteAssets[entry.assetId])}
                      onChange={() => {
                        if (entry.assetId) onToggleDeleteSelection(entry.assetId);
                      }}
                      disabled={isDeletingAssets}
                      aria-label={`Select ${entry.name} for delete`}
                    />
                  ) : null}
                  <button
                    type="button"
                    className={`tree-file${entry.assetId === currentAssetId ? " active" : ""} ${
                      entry.assetId && assetReviewStateById.get(entry.assetId)?.status === "labeled" ? "is-labeled" : "is-unlabeled"
                    }${entry.assetId && selectedDeleteAssets[entry.assetId] ? " delete-selected" : ""}${
                      entry.assetId && assetReviewStateById.get(entry.assetId)?.isDirty ? " is-dirty" : ""
                    }`}
                    onClick={() =>
                      entry.assetId &&
                      (bulkDeleteMode ? onToggleDeleteSelection(entry.assetId) : onSelectTreeAsset(entry.assetId, entry.folderPath))
                    }
                    title={entry.assetId && assetReviewStateById.get(entry.assetId)?.isDirty ? `${entry.name} has staged edits` : undefined}
                  >
                    {entry.name}
                  </button>
                </div>
              )}
            </li>
          ))}
        </ul>
      </section>
    </aside>
  );
}
