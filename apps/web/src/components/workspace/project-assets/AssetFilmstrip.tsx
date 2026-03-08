import { Pagination } from "../../Pagination";
import { asRelativePath } from "../../../lib/workspace/tree";
import type { Asset } from "../../../lib/api";

interface AssetFilmstripProps {
  assetRows: Asset[];
  currentIndex: number;
  pageStatuses: Array<"labeled" | "unlabeled">;
  pageDirtyFlags: boolean[];
  onSelectIndex: (index: number) => void;
  onPrev: () => void;
  onNext: () => void;
}

export function AssetFilmstrip({
  assetRows,
  currentIndex,
  pageStatuses,
  pageDirtyFlags,
  onSelectIndex,
  onPrev,
  onNext,
}: AssetFilmstripProps) {
  return (
    <section className="asset-filmstrip" aria-label="Image navigator" data-testid="asset-filmstrip">
      <div className="asset-filmstrip-top">
        <Pagination
          total={Math.max(assetRows.length, 1)}
          current={Math.max(currentIndex, 0)}
          onSelect={onSelectIndex}
          statuses={pageStatuses}
          dirtyFlags={pageDirtyFlags}
        />
        <div className="asset-filmstrip-nav">
          <button type="button" className="ghost-button" onClick={onPrev} disabled={currentIndex <= 0}>
            Previous
          </button>
          <button type="button" className="ghost-button" onClick={onNext} disabled={currentIndex >= assetRows.length - 1}>
            Next
          </button>
        </div>
      </div>
      <div className="asset-filmstrip-list">
        {assetRows.map((asset, index) => (
          <button
            key={asset.id}
            type="button"
            className={`asset-filmstrip-item${index === currentIndex ? " active" : ""}${
              pageDirtyFlags[index] ? " is-dirty" : ""
            }`}
            onClick={() => onSelectIndex(index)}
            data-testid="filmstrip-item"
            data-demo-asset-id={asset.id}
          >
            <span className={`asset-filmstrip-status is-${pageStatuses[index] ?? "unlabeled"}`} />
            <span className="asset-filmstrip-index">{index + 1}</span>
            <span className="asset-filmstrip-name">{asRelativePath(asset)}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
