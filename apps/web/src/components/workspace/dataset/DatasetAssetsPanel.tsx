import type { AnnotationStatus, DatasetVersionAssetsPayload } from "../../../lib/api";
import { ALL_STATUSES } from "../../../lib/workspace/datasetPage";

type DatasetAssetRow = {
  asset_id: string;
  filename: string;
  relative_path: string;
  status: AnnotationStatus;
  split?: "train" | "val" | "test" | null;
  label_summary?: Record<string, unknown> | null;
};

function AssetList({
  items,
  resolveImageUrl,
}: {
  items: DatasetAssetRow[];
  resolveImageUrl: (assetId: string) => string;
}) {
  return (
    <div style={{ display: "grid", gap: 6 }}>
      {items.map((item) => (
        <div key={item.asset_id} style={{ display: "grid", gridTemplateColumns: "60px 1fr auto auto", gap: 8, alignItems: "center" }}>
          <img
            src={resolveImageUrl(item.asset_id)}
            alt={item.filename}
            style={{ width: 56, height: 42, objectFit: "cover", borderRadius: 6, border: "1px solid var(--line, #d8dce6)" }}
          />
          <span>{item.relative_path || item.filename}</span>
          <span>{item.status}</span>
          <span>{item.split ?? "-"}</span>
        </div>
      ))}
    </div>
  );
}

function AssetGrid({
  items,
  resolveImageUrl,
}: {
  items: DatasetAssetRow[];
  resolveImageUrl: (assetId: string) => string;
}) {
  return (
    <div style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))" }}>
      {items.map((item) => (
        <div key={item.asset_id} style={{ border: "1px solid var(--line, #d8dce6)", borderRadius: 8, padding: 8 }}>
          <img
            src={resolveImageUrl(item.asset_id)}
            alt={item.filename}
            style={{ width: "100%", height: 96, objectFit: "cover", borderRadius: 6, border: "1px solid var(--line, #d8dce6)" }}
          />
          <div style={{ marginTop: 6, display: "grid", gap: 4 }}>
            <span style={{ fontSize: 12, wordBreak: "break-word" }}>{item.relative_path || item.filename}</span>
            <span style={{ fontSize: 12, color: "var(--muted, #6f7b8a)" }}>
              {item.status} | {item.split ?? "-"}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

export function DatasetAssetsPanel({
  mode,
  splitFilter,
  statusFilter,
  classFilter,
  searchText,
  viewMode,
  classFilterOptions,
  isLoadingAssets,
  assetsPayload,
  previewSummary,
  filteredPreviewAssets,
  resolveImageUrl,
  onSplitFilterChange,
  onStatusFilterChange,
  onClassFilterChange,
  onSearchTextChange,
  onViewModeChange,
  onPreviousPage,
  onNextPage,
}: {
  mode: "browse" | "draft";
  splitFilter: "all" | "train" | "val" | "test";
  statusFilter: "all" | AnnotationStatus;
  classFilter: string;
  searchText: string;
  viewMode: "list" | "grid";
  classFilterOptions: Array<{ id: string; name: string }>;
  isLoadingAssets: boolean;
  assetsPayload: DatasetVersionAssetsPayload | null;
  previewSummary: { sample_assets: DatasetAssetRow[] } | null;
  filteredPreviewAssets: DatasetAssetRow[];
  resolveImageUrl: (assetId: string) => string;
  onSplitFilterChange: (value: "all" | "train" | "val" | "test") => void;
  onStatusFilterChange: (value: "all" | AnnotationStatus) => void;
  onClassFilterChange: (value: string) => void;
  onSearchTextChange: (value: string) => void;
  onViewModeChange: (value: "list" | "grid") => void;
  onPreviousPage: () => void;
  onNextPage: () => void;
}) {
  const savedItems = assetsPayload?.items ?? [];

  return (
    <>
      <h3 style={{ marginTop: 16 }}>{mode === "draft" ? "Sample Assets (Preview)" : "Assets"}</h3>
      <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
        <select value={splitFilter} onChange={(event) => onSplitFilterChange(event.target.value as "all" | "train" | "val" | "test")}>
          <option value="all">All splits</option>
          <option value="train">Train</option>
          <option value="val">Val</option>
          <option value="test">Test</option>
        </select>
        <select value={statusFilter} onChange={(event) => onStatusFilterChange(event.target.value as "all" | AnnotationStatus)}>
          <option value="all">All statuses</option>
          {ALL_STATUSES.map((status) => (
            <option key={status} value={status}>
              {status}
            </option>
          ))}
        </select>
        <select value={classFilter} onChange={(event) => onClassFilterChange(event.target.value)}>
          <option value="all">All classes</option>
          {classFilterOptions.map((row) => (
            <option key={row.id} value={row.id}>
              {row.name}
            </option>
          ))}
        </select>
        <input placeholder="Search assets..." value={searchText} onChange={(event) => onSearchTextChange(event.target.value)} />
        <div style={{ display: "flex", gap: 6 }}>
          <button type="button" className={viewMode === "list" ? "ghost-button active-toggle" : "ghost-button"} onClick={() => onViewModeChange("list")}>
            List
          </button>
          <button type="button" className={viewMode === "grid" ? "ghost-button active-toggle" : "ghost-button"} onClick={() => onViewModeChange("grid")}>
            Grid
          </button>
        </div>
      </div>

      {mode === "draft" ? (
        <>
          {!previewSummary ? (
            <p style={{ color: "var(--muted, #6f7b8a)", fontSize: 13 }}>Run preview to see sample assets.</p>
          ) : previewSummary.sample_assets.length === 0 ? (
            <p style={{ color: "var(--muted, #6f7b8a)", fontSize: 13 }}>No assets matched the current filters.</p>
          ) : filteredPreviewAssets.length === 0 ? (
            <p style={{ color: "var(--muted, #6f7b8a)", fontSize: 13 }}>No preview samples matched the current browser filters.</p>
          ) : viewMode === "list" ? (
            <AssetList items={filteredPreviewAssets} resolveImageUrl={resolveImageUrl} />
          ) : (
            <AssetGrid items={filteredPreviewAssets} resolveImageUrl={resolveImageUrl} />
          )}
        </>
      ) : (
        <>
          {isLoadingAssets ? <p>Loading assets...</p> : null}
          {!isLoadingAssets && viewMode === "list" ? <AssetList items={savedItems} resolveImageUrl={resolveImageUrl} /> : null}
          {!isLoadingAssets && viewMode === "grid" ? <AssetGrid items={savedItems} resolveImageUrl={resolveImageUrl} /> : null}
          {assetsPayload ? (
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 10 }}>
              <button type="button" className="ghost-button" disabled={assetsPayload.page <= 1} onClick={onPreviousPage}>
                Prev
              </button>
              <span>
                Page {assetsPayload.page} | {assetsPayload.total} assets
              </span>
              <button
                type="button"
                className="ghost-button"
                disabled={assetsPayload.page * assetsPayload.page_size >= assetsPayload.total}
                onClick={onNextPage}
              >
                Next
              </button>
            </div>
          ) : null}
        </>
      )}
    </>
  );
}
