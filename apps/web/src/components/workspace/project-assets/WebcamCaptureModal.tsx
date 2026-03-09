import { useEffect, useState } from "react";

import type { Asset, AssetSequence } from "../../../lib/api";
import { useWebcamCapture } from "../../../lib/hooks/useWebcamCapture";

interface WebcamCaptureModalProps {
  open: boolean;
  projectId: string | null;
  taskId: string | null;
  defaultName: string;
  onClose: () => void;
  onSequenceCreated?: (sequence: AssetSequence) => void;
  onFrameUploaded?: (asset: Asset, sequence: AssetSequence) => void;
  onFinished?: (sequence: AssetSequence | null) => void;
}

export function WebcamCaptureModal({
  open,
  projectId,
  taskId,
  defaultName,
  onClose,
  onSequenceCreated,
  onFrameUploaded,
  onFinished,
}: WebcamCaptureModalProps) {
  const [name, setName] = useState(defaultName);
  const [fps, setFps] = useState("2");
  const capture = useWebcamCapture({
    projectId,
    taskId,
    onSequenceCreated,
    onFrameUploaded,
  });

  useEffect(() => {
    if (!open) return;
    setName(defaultName);
    setFps("2");
  }, [defaultName, open]);

  const canStart = capture.isPreviewing && Number(fps) > 0;

  if (!open) return null;

  function handleFinish() {
    capture.stopCapture();
    capture.stopPreview();
    onFinished?.(capture.sequence);
    onClose();
  }

  return (
    <div className="import-modal-backdrop">
      <div className="import-modal webcam-capture-modal">
        <h3>Webcam Capture</h3>
        <div className="webcam-preview-shell">
          <video ref={capture.videoRef} className="webcam-preview" muted playsInline autoPlay />
        </div>
        <div className="import-inline-grid">
          <label className="import-field">
            <span>Capture FPS</span>
            <input value={fps} onChange={(event) => setFps(event.target.value)} inputMode="decimal" />
          </label>
          <label className="import-field">
            <span>Session name</span>
            <input value={name} onChange={(event) => setName(event.target.value)} />
          </label>
        </div>
        {capture.error ? <p className="import-field-error">{capture.error}</p> : null}
        <p className="webcam-capture-count">Captured frames: {capture.captureCount}</p>
        <div className="import-modal-actions">
          <button type="button" className="ghost-button" onClick={() => void capture.requestPreview()} disabled={capture.isPreviewing}>
            {capture.isPreviewing ? "Preview Ready" : "Start Preview"}
          </button>
          <button type="button" className="primary-button" onClick={() => void capture.startCapture({ name, fps: Number(fps) })} disabled={!canStart || capture.isCapturing}>
            Start Capture
          </button>
          <button type="button" className="ghost-button" onClick={capture.stopCapture} disabled={!capture.isCapturing}>
            Stop
          </button>
          <button type="button" className="ghost-button" onClick={handleFinish}>
            Finish
          </button>
        </div>
      </div>
    </div>
  );
}
