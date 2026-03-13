import { useMemo, useState } from "react";

import type { AnnotationStatus } from "../../../lib/api";
import { type TreeEntry } from "../../../lib/workspace/tree";
import { ImportMenu } from "./ImportMenu";
import { SequenceStatusBadge } from "./SequenceStatusBadge";

interface FilterLabelRow {
  id: string;
  name: string;
}

interface AssetBrowserProps {
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
  onImportImages: () => void;
  onImportVideo: () => void;
  onOpenWebcam: () => void;
  onCollapseAllFolders: () => void;
  onExpandAllFolders: () => void;
  onSelectFolderScope: (folderPath: string | null) => void;
  onToggleBulkDeleteMode: () => void;
  onSelectAllDeleteScope: () => void;
  onClearDeleteSelection: () => void;
  onDeleteCurrentAsset: () => void;
  onDeleteSelectedAssets: () => void;
  onDeleteSelectedFolder: () => void;
  onDeleteCurrentProject: () => void;
  onToggleFolderCollapsed: (folderPath: string) => void;
  onDeleteFolderPath: (folderPath: string) => void;
  onToggleDeleteSelection: (assetId: string) => void;
  onSelectTreeAsset: (assetId: string, folderPath?: string) => void;
  onChangeFilterStatus: (status: "all" | AnnotationStatus) => void;
  onChangeFilterCategoryId: (categoryId: string) => void;
}

export function AssetBrowser({
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
  onImportImages,
  onImportVideo,
  onOpenWebcam,
  onCollapseAllFolders,
  onExpandAllFolders,
  onSelectFolderScope,
  onToggleBulkDeleteMode,
  onSelectAllDeleteScope,
  onClearDeleteSelection,
  onDeleteCurrentAsset,
  onDeleteSelectedAssets,
  onDeleteSelectedFolder,
  onDeleteCurrentProject,
  onToggleFolderCollapsed,
  onDeleteFolderPath,
  onToggleDeleteSelection,
  onSelectTreeAsset,
  onChangeFilterStatus,
  onChangeFilterCategoryId,
}: AssetBrowserProps) {
  const [searchText, setSearchText] = useState("");
  const filteredEntries = useMemo(() => {
    const query = searchText.trim().toLowerCase();
    if (!query) return visibleTreeEntries;
    return visibleTreeEntries.filter((entry) => entry.name.toLowerCase().includes(query) || entry.path.toLowerCase().includes(query));
  }, [searchText, visibleTreeEntries]);

  return (
    <aside className="asset-browser" data-testid="asset-browser">
      <section className="placeholder-card asset-browser-panel" data-testid="asset-browser-panel">
        <div className="asset-browser-head">
          <div>
            <h3>Assets</h3>
            <p>Browse folders, import images, and manage the current labeling scope.</p>
          </div>
          <div className="asset-browser-head-actions">
            <ImportMenu onImportImages={onImportImages} onImportVideo={onImportVideo} onImportWebcam={onOpenWebcam} />
            <button type="button" className="ghost-button" disabled title="Folder creation is not wired yet">
              New Folder
            </button>
          </div>
        </div>

        <div className="asset-browser-toolbar">
          <input
            data-testid="asset-browser-search"
            value={searchText}
            onChange={(event) => setSearchText(event.target.value)}
            placeholder="Search folders or files"
            className="label-create-input"
          />
          <div className="asset-browser-scope-actions">
            <button type="button" className="ghost-button" onClick={onExpandAllFolders}>
              Expand All
            </button>
            <button type="button" className="ghost-button" onClick={onCollapseAllFolders}>
              Collapse All
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

        <button
          type="button"
          className={selectedTreeFolderPath === null ? "tree-scope-button active" : "tree-scope-button"}
          onClick={() => onSelectFolderScope(null)}
          data-testid="folder-tree-root"
        >
          All files
        </button>
        {selectedTreeFolderPath ? <p className="tree-scope-caption">Scope: {selectedTreeFolderPath}</p> : null}

        <ul className="asset-browser-tree" data-testid="folder-tree">
          {filteredEntries.map((entry) => (
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
                    data-testid="folder-tree-folder"
                    data-demo-path={entry.path}
                  >
                    <span>{entry.name}</span>
                    <SequenceStatusBadge
                      sourceType={entry.sequenceSourceType}
                      status={entry.sequenceStatus}
                      frameCount={entry.sequenceFrameCount}
                    />
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
                      onChange={() => entry.assetId && onToggleDeleteSelection(entry.assetId)}
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
                    data-testid="folder-tree-asset"
                    data-demo-path={entry.path}
                    data-demo-asset-id={entry.assetId ?? ""}
                  >
                    {entry.name}
                  </button>
                </div>
              )}
            </li>
          ))}
        </ul>

        <div className="asset-browser-dangerzone">
          <div className="asset-browser-dangerzone-head">
            <h4>Danger Zone</h4>
            <button
              type="button"
              className={bulkDeleteMode ? "ghost-button active-toggle" : "ghost-button"}
              onClick={onToggleBulkDeleteMode}
              disabled={!selectedProjectId || isDeletingAssets}
            >
              {bulkDeleteMode ? "Exit Multi-delete" : "Multi-delete"}
            </button>
          </div>
          {bulkDeleteMode ? (
            <div className="asset-browser-dangerzone-actions">
              <button type="button" className="ghost-button" onClick={onSelectAllDeleteScope} disabled={isDeletingAssets}>
                Select Scope
              </button>
              <button type="button" className="ghost-button" onClick={onClearDeleteSelection} disabled={isDeletingAssets}>
                Clear
              </button>
            </div>
          ) : null}
          <div className="asset-browser-dangerzone-actions">
            <button type="button" className="ghost-button danger-button" onClick={onDeleteCurrentAsset} disabled={isDeletingAssets}>
              Remove Image
            </button>
            <button
              type="button"
              className="ghost-button danger-button"
              onClick={onDeleteSelectedAssets}
              disabled={isDeletingAssets || selectedDeleteAssetIdsLength === 0}
            >
              Delete Selected ({selectedDeleteAssetIdsLength})
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
              disabled={isDeletingAssets || !selectedProjectId}
            >
              Delete Project
            </button>
          </div>
        </div>
      </section>
    </aside>
  );
}
