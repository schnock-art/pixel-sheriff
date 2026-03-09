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
          }`}
          title={`Frame ${asset.frame_index ?? 0}${typeof asset.timestamp_seconds === "number" ? ` • ${asset.timestamp_seconds.toFixed(2)}s` : ""}`}
          onClick={() => onSelectAsset(asset.id)}
        />
      ))}
    </div>
  );
}
