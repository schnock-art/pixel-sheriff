import { resolveAssetUri, type SequenceFrameAsset } from "../../../lib/api";

interface SequenceThumbnailStripProps {
  assets: SequenceFrameAsset[];
  currentAssetId: string | null;
  onSelectAsset: (assetId: string) => void;
}

export function SequenceThumbnailStrip({ assets, currentAssetId, onSelectAsset }: SequenceThumbnailStripProps) {
  return (
    <div className="sequence-thumbnail-strip">
      {assets.map((asset) => (
        <button
          key={asset.id}
          type="button"
          className={`sequence-thumbnail-item${asset.id === currentAssetId ? " active" : ""}${
            asset.has_annotations ? " is-labeled" : ""
          }${asset.pending_prelabel_count > 0 ? " has-pending-prelabels" : ""}`}
          title={asset.pending_prelabel_count > 0 ? `${asset.pending_prelabel_count} pending AI prelabels` : undefined}
          onClick={() => onSelectAsset(asset.id)}
        >
          <img src={resolveAssetUri(asset.thumbnail_url)} alt={asset.file_name} />
          <span>{(asset.frame_index ?? 0) + 1}</span>
          {asset.pending_prelabel_count > 0 ? <strong>{asset.pending_prelabel_count}</strong> : null}
        </button>
      ))}
    </div>
  );
}
