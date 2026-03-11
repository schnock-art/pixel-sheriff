import { useEffect, useMemo, useState } from "react";

import type { Asset, AssetSequence } from "../../../lib/api";
import { useWebcamCapture } from "../../../lib/hooks/useWebcamCapture";
import { buildCameraDestinations } from "../../../lib/workspace/webcamCapture";

interface CameraDestinationView {
  deviceId: string;
  cameraLabel: string;
  folderPath: string;
  sequenceName: string;
}

interface WebcamCaptureModalProps {
  open: boolean;
  projectId: string | null;
  taskId: string | null;
  defaultName: string;
  folderOptions: string[];
  defaultRootFolderPath?: string | null;
  onClose: () => void;
  onSequenceCreated?: (sequence: AssetSequence) => void;
  onFrameUploaded?: (asset: Asset, sequence: AssetSequence) => void;
  onFinished?: (sequences: AssetSequence[]) => void;
}

export function WebcamCaptureModal({
  open,
  projectId,
  taskId,
  defaultName,
  folderOptions,
  defaultRootFolderPath,
  onClose,
  onSequenceCreated,
  onFrameUploaded,
  onFinished,
}: WebcamCaptureModalProps) {
  const [name, setName] = useState(defaultName);
  const [fps, setFps] = useState("2");
  const [rootFolderPath, setRootFolderPath] = useState("");
  const capture = useWebcamCapture({
    projectId,
    taskId,
    onSequenceCreated,
    onFrameUploaded,
  });
  const { refreshDevices, reset } = capture;

  useEffect(() => {
    if (!open) return;
    setName(defaultName);
    setFps("2");
    setRootFolderPath(defaultRootFolderPath ?? "");
    reset();
    void refreshDevices();
  }, [defaultName, defaultRootFolderPath, open, refreshDevices, reset]);

  const selectedDevices = useMemo(
    () => capture.devices.filter((device) => capture.selectedDeviceIds.includes(device.deviceId)),
    [capture.devices, capture.selectedDeviceIds],
  );
  const destinations = useMemo(
    () =>
      buildCameraDestinations({
        devices: capture.devices.map((device) => ({ deviceId: device.deviceId, label: device.label })),
        selectedDeviceIds: capture.selectedDeviceIds,
        sessionName: name,
        rootFolderPath,
        existingPaths: folderOptions,
      }) as CameraDestinationView[],
    [capture.devices, capture.selectedDeviceIds, folderOptions, name, rootFolderPath],
  );
  const destinationByDeviceId = useMemo(
    () => new Map(destinations.map((destination) => [destination.deviceId, destination])),
    [destinations],
  );

  const canStart = destinations.some((destination) => {
    const device = capture.devices.find((item) => item.deviceId === destination.deviceId);
    return Boolean(device?.isPreviewing);
  }) && Number(fps) > 0;

  if (!open) return null;

  function handleSelectionChange(values: string[]) {
    capture.setSelectedDeviceIds(values);
  }

  function handleFinish() {
    capture.stopCapture();
    capture.stopPreview();
    onFinished?.(capture.sequences);
    onClose();
  }

  return (
    <div className="import-modal-backdrop">
      <div className="import-modal webcam-capture-modal">
        <div className="webcam-modal-head">
          <div>
            <h3>Webcam Capture</h3>
            <p>Select one or more cameras, preview them, then start a synchronized capture.</p>
          </div>
          <button type="button" className="ghost-button" onClick={() => void capture.refreshDevices({ requestAccess: true })} disabled={capture.isLoadingDevices}>
            {capture.isLoadingDevices ? "Refreshing..." : "Refresh Cameras"}
          </button>
        </div>
        <div className="import-inline-grid">
          <label className="import-field">
            <span>Session name</span>
            <input value={name} onChange={(event) => setName(event.target.value)} />
          </label>
          <label className="import-field">
            <span>Capture FPS</span>
            <input value={fps} onChange={(event) => setFps(event.target.value)} inputMode="decimal" />
          </label>
        </div>
        <div className="import-inline-grid">
          <label className="import-field">
            <span>Cameras</span>
            <select
              multiple
              size={Math.max(3, Math.min(6, capture.devices.length || 3))}
              value={capture.selectedDeviceIds}
              onChange={(event) => {
                const values = Array.from(event.currentTarget.selectedOptions, (option) => option.value);
                handleSelectionChange(values);
              }}
            >
              {capture.devices.map((device) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label}
                </option>
              ))}
            </select>
            <span className="import-field-hint">Hold Ctrl/Cmd to select more than one camera. If only one camera appears, click Refresh Cameras and allow access.</span>
          </label>
          <label className="import-field">
            <span>Destination root (optional)</span>
            <select value={rootFolderPath} onChange={(event) => setRootFolderPath(event.target.value)}>
              <option value="">Project root</option>
              {folderOptions.map((folderPath) => (
                <option key={folderPath} value={folderPath}>
                  {folderPath}
                </option>
              ))}
            </select>
            <span className="import-field-hint">Each camera gets its own subfolder under this root.</span>
          </label>
        </div>
        {capture.error ? <p className="import-field-error">{capture.error}</p> : null}
        <p className="webcam-capture-count">Captured frames: {capture.captureCount}</p>
        <div className="webcam-preview-grid">
          {selectedDevices.length === 0 ? (
            <div className="webcam-preview-empty">Select at least one camera to preview it here.</div>
          ) : (
            selectedDevices.map((device) => {
              const destination = destinationByDeviceId.get(device.deviceId);
              return (
                <article key={device.deviceId} className="webcam-preview-card">
                  <header>
                    <strong>{device.label}</strong>
                    <span className={device.isCapturing ? "webcam-device-status is-live" : device.isPreviewing ? "webcam-device-status" : "webcam-device-status is-idle"}>
                      {device.isCapturing ? "Capturing" : device.isPreviewing ? "Preview ready" : "Idle"}
                    </span>
                  </header>
                  <div className="webcam-preview-shell">
                    <video
                      ref={(node) => capture.attachVideoRef(device.deviceId, node)}
                      className="webcam-preview"
                      muted
                      playsInline
                      autoPlay
                    />
                  </div>
                  <div className="webcam-preview-meta">
                    <span>Sequence: {destination?.sequenceName ?? "Pending"}</span>
                    <span>Folder: {destination?.folderPath ?? "Pending"}</span>
                    <span>Frames: {device.captureCount}</span>
                  </div>
                  {device.error ? <p className="import-field-error">{device.error}</p> : null}
                </article>
              );
            })
          )}
        </div>
        <div className="import-modal-actions">
          <button
            type="button"
            className="ghost-button"
            onClick={() => void capture.requestPreview()}
            disabled={capture.selectedDeviceIds.length === 0}
          >
            Start Preview
          </button>
          <button
            type="button"
            className="primary-button"
            onClick={() => void capture.startCapture({ fps: Number(fps), destinations })}
            disabled={!canStart || capture.isCapturing}
          >
            Start Capture
          </button>
          <button type="button" className="ghost-button" onClick={() => capture.stopCapture()} disabled={!capture.isCapturing}>
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
