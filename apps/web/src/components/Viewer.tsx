import { Pagination } from "./Pagination";

interface ViewerAsset {
  id: string;
  uri: string;
}

interface ViewerProps {
  currentAsset: ViewerAsset | null;
  totalAssets: number;
  currentIndex: number;
  pageStatuses?: Array<"labeled" | "unlabeled">;
  pageDirtyFlags?: boolean[];
  onSelectIndex: (index: number) => void;
  onPrev: () => void;
  onNext: () => void;
}

export function Viewer({ currentAsset, totalAssets, currentIndex, pageStatuses, pageDirtyFlags, onSelectIndex, onPrev, onNext }: ViewerProps) {
  const hasImage = Boolean(currentAsset?.uri);
  const maxIndex = Math.max(totalAssets - 1, 0);

  function jump(delta: number) {
    const nextIndex = Math.min(maxIndex, Math.max(0, currentIndex + delta));
    onSelectIndex(nextIndex);
  }

  return (
    <section className="viewer-panel" aria-label="Image viewer">
      <div className={hasImage ? "viewer-canvas has-image" : "viewer-canvas"} role="img" aria-label="Traffic scene with annotations">
        {currentAsset?.uri ? <img src={currentAsset.uri} alt={`Asset ${currentIndex + 1}`} className="viewer-image" /> : null}
        <div className="skyline" />
        <div className="road" />
        <div className="car car-main" />
        <div className="car car-left" />
        <div className="car car-right" />
      </div>

      <div className="viewer-controls">
        <Pagination
          total={Math.max(totalAssets, 1)}
          current={Math.max(currentIndex, 0)}
          onSelect={onSelectIndex}
          statuses={pageStatuses}
          dirtyFlags={pageDirtyFlags}
        />
        <div className="viewer-nav">
          <button type="button" className="ghost-icon-button" aria-label="Back 10 frames" onClick={() => jump(-10)}>
            -10
          </button>
          <button type="button" className="ghost-icon-button" aria-label="Back 5 frames" onClick={() => jump(-5)}>
            -5
          </button>
          <button type="button" className="ghost-icon-button" aria-label="Previous frame" onClick={onPrev}>
            {"<"}
          </button>
          <button type="button" className="ghost-icon-button" aria-label="Next frame" onClick={onNext}>
            {">"}
          </button>
          <button type="button" className="ghost-icon-button" aria-label="Forward 5 frames" onClick={() => jump(5)}>
            +5
          </button>
          <button type="button" className="ghost-icon-button" aria-label="Forward 10 frames" onClick={() => jump(10)}>
            +10
          </button>
        </div>
      </div>
    </section>
  );
}
