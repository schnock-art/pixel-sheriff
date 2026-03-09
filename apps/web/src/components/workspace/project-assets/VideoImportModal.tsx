import { useEffect, useMemo, useState } from "react";

import type { VideoImportPayload } from "../../../lib/api";

interface VideoImportModalProps {
  open: boolean;
  defaultName: string;
  isImporting: boolean;
  errorMessage: string | null;
  onClose: () => void;
  onSubmit: (file: File, payload: VideoImportPayload) => void;
}

export function VideoImportModal({ open, defaultName, isImporting, errorMessage, onClose, onSubmit }: VideoImportModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [name, setName] = useState(defaultName);
  const [fps, setFps] = useState("2");
  const [maxFrames, setMaxFrames] = useState("500");
  const [resolution, setResolution] = useState<"original" | "1280" | "720">("original");

  useEffect(() => {
    if (!open) return;
    setSelectedFile(null);
    setName(defaultName);
    setFps("2");
    setMaxFrames("500");
    setResolution("original");
  }, [defaultName, open]);

  const canSubmit = useMemo(() => Boolean(selectedFile) && Number(fps) > 0 && Number(maxFrames) > 0, [fps, maxFrames, selectedFile]);

  if (!open) return null;

  function handleSubmit() {
    if (!selectedFile) return;
    const resizeMode = resolution === "original" ? "original" : "width";
    onSubmit(selectedFile, {
      name,
      fps: Number(fps),
      max_frames: Number(maxFrames),
      resize_mode: resizeMode,
      resize_width: resolution === "1280" ? 1280 : resolution === "720" ? 720 : null,
      resize_height: null,
    });
  }

  return (
    <div className="import-modal-backdrop">
      <div className="import-modal video-import-modal">
        <h3>Import Video</h3>
        <label className="import-field">
          <span>Video file</span>
          <input
            type="file"
            accept=".mp4,.mov,.avi,.mkv,video/*"
            onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
          />
        </label>
        <label className="import-field">
          <span>Session name</span>
          <input value={name} onChange={(event) => setName(event.target.value)} placeholder="my_video_session" />
        </label>
        <div className="import-inline-grid">
          <label className="import-field">
            <span>FPS</span>
            <input value={fps} onChange={(event) => setFps(event.target.value)} inputMode="decimal" />
          </label>
          <label className="import-field">
            <span>Max frames</span>
            <input value={maxFrames} onChange={(event) => setMaxFrames(event.target.value)} inputMode="numeric" />
          </label>
        </div>
        <label className="import-field">
          <span>Resolution</span>
          <select value={resolution} onChange={(event) => setResolution(event.target.value as "original" | "1280" | "720")}>
            <option value="original">Original</option>
            <option value="1280">1280px wide</option>
            <option value="720">720px wide</option>
          </select>
        </label>
        {errorMessage ? <p className="import-field-error">{errorMessage}</p> : null}
        <div className="import-modal-actions">
          <button type="button" className="ghost-button" onClick={onClose} disabled={isImporting}>
            Cancel
          </button>
          <button type="button" className="primary-button" onClick={handleSubmit} disabled={!canSubmit || isImporting}>
            {isImporting ? "Importing..." : "Import"}
          </button>
        </div>
      </div>
    </div>
  );
}
