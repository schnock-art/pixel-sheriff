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
          }`}
          onClick={() => onSelectAsset(asset.id)}
        >
          <img src={resolveAssetUri(asset.thumbnail_url)} alt={asset.file_name} />
          <span>{(asset.frame_index ?? 0) + 1}</span>
        </button>
      ))}
    </div>
  );
}
