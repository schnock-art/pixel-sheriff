import { Pagination } from "./Pagination";

interface ViewerAsset {
  id: string;
  uri: string;
}

interface ViewerProps {
  currentAsset: ViewerAsset | null;
  totalAssets: number;
  currentIndex: number;
  onSelectIndex: (index: number) => void;
  onPrev: () => void;
  onNext: () => void;
}

export function Viewer({ currentAsset, totalAssets, currentIndex, onSelectIndex, onPrev, onNext }: ViewerProps) {
  const hasImage = Boolean(currentAsset?.uri);

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
        <Pagination total={Math.max(totalAssets, 1)} current={Math.max(currentIndex, 0)} onSelect={onSelectIndex} />
        <div className="viewer-nav">
          <button type="button" className="ghost-button" onClick={onNext}>
            Next
          </button>
          <button type="button" className="ghost-icon-button" aria-label="Previous frame" onClick={onPrev}>
            {"<"}
          </button>
          <button type="button" className="ghost-icon-button" aria-label="Next frame" onClick={onNext}>
            {">"}
          </button>
        </div>
      </div>
    </section>
  );
}
