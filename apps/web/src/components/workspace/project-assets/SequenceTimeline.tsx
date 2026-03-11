import type { SequenceFrameAsset } from "../../../lib/api";

interface SequenceTimelineProps {
  assets: SequenceFrameAsset[];
  currentAssetId: string | null;
  onSelectAsset: (assetId: string) => void;
}

export function SequenceTimeline({ assets, currentAssetId, onSelectAsset }: SequenceTimelineProps) {
  return (
    <div className="sequence-timeline" aria-label="Sequence timeline">
      {assets.map((asset) => (
        <button
          key={asset.id}
          type="button"
          className={`sequence-timeline-dot${asset.id === currentAssetId ? " active" : ""}${
            asset.has_annotations ? " is-labeled" : ""
          }${asset.pending_prelabel_count > 0 ? " has-pending-prelabels" : ""}`}
          title={`Frame ${asset.frame_index ?? 0}${typeof asset.timestamp_seconds === "number" ? ` • ${asset.timestamp_seconds.toFixed(2)}s` : ""}${
            asset.pending_prelabel_count > 0 ? ` • ${asset.pending_prelabel_count} pending AI` : ""
          }`}
          onClick={() => onSelectAsset(asset.id)}
        >
          {asset.pending_prelabel_count > 0 ? <span className="sequence-timeline-count">{asset.pending_prelabel_count}</span> : null}
        </button>
      ))}
    </div>
  );
}
