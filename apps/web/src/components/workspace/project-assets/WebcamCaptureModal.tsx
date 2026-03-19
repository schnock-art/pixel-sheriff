import { useEffect, useMemo, useRef, useState } from "react";

import { closePrelabelInput, type Asset, type AssetSequence, type PrelabelConfig, type TaskKind } from "../../../lib/api";
import { useCompatibleDeployments } from "../../../lib/hooks/useCompatibleDeployments";
import { usePreviewInference } from "../../../lib/hooks/usePreviewInference";
import { useWebcamCapture } from "../../../lib/hooks/useWebcamCapture";
import { captureVideoFrame, subscribeToMediaFrames } from "../../../lib/workspace/mediaPreview";
import { finishWebcamCapture } from "../../../lib/workspace/webcamCaptureFinish";
import { buildCameraDestinations } from "../../../lib/workspace/webcamCapture";
import { DeploymentSelectField } from "./DeploymentSelectField";
import { MediaImportWorkspaceModal } from "./MediaImportWorkspaceModal";
import { ModalMediaPreview } from "./ModalMediaPreview";
import { PrelabelSettingsSection } from "./PrelabelSettingsSection";

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
  taskKind: TaskKind | null;
  defaultName: string;
  folderOptions: string[];
  defaultRootFolderPath?: string | null;
  enablePrelabels?: boolean;
  defaultPrompts?: string[];
  onClose: () => void;
  onSequenceCreated?: (sequence: AssetSequence) => void;
  onFrameUploaded?: (asset: Asset, sequence: AssetSequence) => void;
  onFinished?: (sequences: AssetSequence[]) => void;
}

export function WebcamCaptureModal({
  open,
  projectId,
  taskId,
  taskKind,
  defaultName,
  folderOptions,
  defaultRootFolderPath,
  enablePrelabels = false,
  defaultPrompts = [],
  onClose,
  onSequenceCreated,
  onFrameUploaded,
  onFinished,
}: WebcamCaptureModalProps) {
  const wasOpenRef = useRef(false);
  const activeVideoRef = useRef<HTMLVideoElement | null>(null);

  const [name, setName] = useState(defaultName);
  const [fps, setFps] = useState("2");
  const [rootFolderPath, setRootFolderPath] = useState("");
  const [prelabelConfig, setPrelabelConfig] = useState<PrelabelConfig | null>(null);
  const [selectedClassificationDeploymentId, setSelectedClassificationDeploymentId] = useState<string | null>(null);
  const [activeDeviceId, setActiveDeviceId] = useState<string | null>(null);
  const [activeVideoMeta, setActiveVideoMeta] = useState<{ width: number; height: number } | null>(null);
  const [previewTick, setPreviewTick] = useState(0);
  const [previewEventTick, setPreviewEventTick] = useState(0);
  const previewTaskKind = taskKind === "classification" || taskKind === "bbox" ? taskKind : null;
  const deployments = useCompatibleDeployments({ projectId, taskId, taskKind: previewTaskKind });

  const capture = useWebcamCapture({
    projectId,
    taskId,
    onSequenceCreated,
    onFrameUploaded,
  });
  const { refreshDevices, reset } = capture;

  useEffect(() => {
    if (!open) {
      wasOpenRef.current = false;
      return;
    }
    if (wasOpenRef.current) return;

    wasOpenRef.current = true;
    setName(defaultName);
    setFps("2");
    setRootFolderPath(defaultRootFolderPath ?? "");
    setActiveDeviceId(null);
    setActiveVideoMeta(null);
    setPreviewTick(0);
    setPreviewEventTick(0);
    setPrelabelConfig(
      enablePrelabels
        ? {
            source_type: "florence2",
            deployment_id: null,
            prompts: defaultPrompts,
            frame_sampling: { mode: "every_n_frames", value: 2 },
            confidence_threshold: 0.25,
            max_detections_per_frame: 20,
          }
        : null,
    );
    setSelectedClassificationDeploymentId(null);
    reset();
    void refreshDevices();
  }, [defaultName, defaultPrompts, defaultRootFolderPath, enablePrelabels, open, refreshDevices, reset]);

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
  const activeDevice =
    selectedDevices.find((device) => device.deviceId === activeDeviceId) ??
    selectedDevices[0] ??
    null;
  const canStart = destinations.some((destination) => {
    const device = capture.devices.find((item) => item.deviceId === destination.deviceId);
    return Boolean(device?.isPreviewing);
  }) && Number(fps) > 0;

  useEffect(() => {
    if (previewTaskKind !== "classification") {
      setSelectedClassificationDeploymentId(null);
      return;
    }
    const nextDeploymentId =
      deployments.activeCompatibleDeployment?.deployment_id ??
      deployments.deployments[0]?.deployment_id ??
      null;
    setSelectedClassificationDeploymentId((current) => {
      if (current && deployments.deployments.some((item) => item.deployment_id === current)) return current;
      return nextDeploymentId;
    });
  }, [deployments.activeCompatibleDeployment, deployments.deployments, previewTaskKind]);

  useEffect(() => {
    if (selectedDevices.length === 0) {
      setActiveDeviceId(null);
      setActiveVideoMeta(null);
      return;
    }
    setActiveDeviceId((current) => (current && selectedDevices.some((device) => device.deviceId === current) ? current : selectedDevices[0].deviceId));
  }, [selectedDevices]);

  useEffect(() => {
    setActiveVideoMeta(null);
  }, [activeDeviceId]);

  useEffect(() => {
    if (!open || !activeDevice?.isPreviewing) return;
    setPreviewEventTick((value) => value + 1);
  }, [activeDevice?.deviceId, activeDevice?.isPreviewing, open]);

  useEffect(() => {
    if (!open || !activeDevice?.isPreviewing) return;
    if (!activeVideoRef.current) return;
    return subscribeToMediaFrames(activeVideoRef.current, {
      minIntervalMs: activeDevice.isCapturing ? 300 : 500,
      onTick: () => setPreviewTick((value) => value + 1),
    });
  }, [activeDevice?.deviceId, activeDevice?.isCapturing, activeDevice?.isPreviewing, open]);

  const preview = usePreviewInference({
    enabled: open && Boolean(activeDevice?.isPreviewing) && (previewTaskKind !== "classification" || Boolean(selectedClassificationDeploymentId)),
    projectId,
    taskId,
    taskKind: previewTaskKind,
    captureFrame: () =>
      captureVideoFrame(activeVideoRef.current, {
        filename: `${activeDevice?.deviceId ?? "webcam"}-preview.jpg`,
        quality: 0.75,
        maxDimension: 960,
      }),
    prelabelConfig: previewTaskKind === "bbox" ? prelabelConfig : null,
    deploymentId: previewTaskKind === "classification" ? selectedClassificationDeploymentId : null,
    refreshToken: previewTick,
    immediateRefreshToken: previewEventTick,
    minIntervalMs: activeDevice?.isCapturing ? 300 : 500,
  });

  if (!open) return null;

  function handleSelectionChange(values: string[]) {
    capture.setSelectedDeviceIds(values);
  }

  async function handleFinish() {
    const result = await finishWebcamCapture({
      projectId,
      taskId,
      devices: capture.devices,
      sequences: capture.sequences,
      stopCapture: capture.stopCapture,
      waitForPendingUploads: capture.waitForPendingUploads,
      stopPreview: capture.stopPreview,
      closePrelabelInput,
    });
    onFinished?.(result.sequences);
    onClose();
  }

  const activeDestination = activeDevice ? destinationByDeviceId.get(activeDevice.deviceId) : null;
  const metaItems = [
    activeDevice ? `Camera: ${activeDevice.label}` : null,
    activeVideoMeta ? `Preview: ${activeVideoMeta.width}x${activeVideoMeta.height}` : null,
    activeDestination ? `Sequence: ${activeDestination.sequenceName}` : null,
    activeDestination ? `Folder: ${activeDestination.folderPath}` : null,
    activeDevice ? `Frames written: ${activeDevice.captureCount}` : null,
  ].filter((value): value is string => Boolean(value));

  return (
    <MediaImportWorkspaceModal
      title="Webcam Capture"
      subtitle="Select one or more cameras, preview them, then start a synchronized capture."
      headerAction={
        <button
          type="button"
          className="ghost-button"
          onClick={() => void capture.refreshDevices({ requestAccess: true })}
          disabled={capture.isLoadingDevices}
        >
          {capture.isLoadingDevices ? "Refreshing..." : "Refresh Cameras"}
        </button>
      }
      controls={
        <>
          <section className="placeholder-card import-config-card">
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
          </section>

          {selectedDevices.length > 0 ? (
            <section className="placeholder-card webcam-device-focus-panel">
              <h4>Focused Preview</h4>
              <div className="webcam-device-focus-list">
                {selectedDevices.map((device) => (
                  <button
                    key={device.deviceId}
                    type="button"
                    className={device.deviceId === activeDevice?.deviceId ? "webcam-device-focus-button active" : "webcam-device-focus-button"}
                    onClick={() => setActiveDeviceId(device.deviceId)}
                    data-testid="webcam-device-focus-button"
                  >
                    <strong>{device.label}</strong>
                    <span>{device.isCapturing ? "Capturing" : device.isPreviewing ? "Preview ready" : "Idle"}</span>
                  </button>
                ))}
              </div>
              <ul className="webcam-device-summary-list">
                {selectedDevices.map((device) => {
                  const destination = destinationByDeviceId.get(device.deviceId);
                  return (
                    <li key={device.deviceId}>
                      <strong>{device.label}</strong>
                      <span>{destination?.folderPath ?? "Pending folder"}</span>
                      <span>{device.captureCount} frame{device.captureCount === 1 ? "" : "s"} written</span>
                    </li>
                  );
                })}
              </ul>
            </section>
          ) : null}

          <PrelabelSettingsSection
            enabled={enablePrelabels}
            projectId={projectId}
            taskId={taskId}
            value={prelabelConfig}
            defaultPrompts={defaultPrompts}
            deploymentOptions={deployments.deployments}
            activeDeploymentId={deployments.activeDeploymentId}
            deploymentsLoading={deployments.isLoading}
            onChange={setPrelabelConfig}
            samplingLabel="Sample every N frames"
            samplingHint="Use 2 for roughly one box pass every second at 2 FPS."
          />

          {!enablePrelabels && previewTaskKind === "classification" ? (
            <section className="placeholder-card import-preview-info-card">
              <h4>Model Preview</h4>
              <DeploymentSelectField
                label="Project deployment"
                deployments={deployments.deployments}
                activeDeploymentId={deployments.activeDeploymentId}
                value={selectedClassificationDeploymentId}
                loading={deployments.isLoading}
                emptyMessage="No compatible classification deployments are available for this task yet."
                helpText="The selected deployment is used only for preview overlays."
                onChange={setSelectedClassificationDeploymentId}
              />
              <p>Classification preview uses the selected compatible deployment for this task and refreshes while the focused camera is live.</p>
            </section>
          ) : null}

          {capture.error ? <p className="import-field-error">{capture.error}</p> : null}
          <p className="webcam-capture-count">
            Captured frames: {capture.captureCount}
            {capture.isCapturing ? " | Recording live while uploads continue in the background." : ""}
          </p>
        </>
      }
      preview={
        <ModalMediaPreview
          title={activeDevice ? activeDevice.label : "Preview"}
          subtitle="The overlay is advisory only and refreshes on the focused live camera."
          emptyMessage="Select at least one camera, then start preview to inspect the live frame here."
          mediaBasis={activeVideoMeta}
          mediaReady={Boolean(activeDevice?.isPreviewing && activeVideoMeta)}
          liveBadgeText={activeDevice?.isCapturing ? "REC" : null}
          preview={preview}
          metaItems={metaItems}
        >
          <>
            {activeDevice ? (
              <video
                ref={(node) => {
                  activeVideoRef.current = node;
                  capture.attachVideoRef(activeDevice.deviceId, node);
                }}
                className="modal-media-preview-element"
                muted
                playsInline
                autoPlay
                data-testid="webcam-active-preview"
                onLoadedMetadata={(event) => {
                  setActiveVideoMeta({
                    width: event.currentTarget.videoWidth,
                    height: event.currentTarget.videoHeight,
                  });
                  setPreviewEventTick((value) => value + 1);
                }}
              />
            ) : null}
            <div className="webcam-hidden-streams" aria-hidden="true">
              {selectedDevices
                .filter((device) => device.deviceId !== activeDevice?.deviceId)
                .map((device) => (
                  <video
                    key={device.deviceId}
                    ref={(node) => capture.attachVideoRef(device.deviceId, node)}
                    className="webcam-hidden-stream"
                    muted
                    playsInline
                    autoPlay
                  />
                ))}
            </div>
          </>
        </ModalMediaPreview>
      }
      footer={
        <>
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
            onClick={() => void capture.startCapture({ fps: Number(fps), destinations, prelabelConfig })}
            disabled={!canStart || capture.isCapturing}
          >
            Start Capture
          </button>
          <button type="button" className="ghost-button" onClick={() => capture.stopCapture()} disabled={!capture.isCapturing}>
            Stop
          </button>
          <button type="button" className="ghost-button" onClick={() => void handleFinish()}>
            Finish
          </button>
        </>
      }
    />
  );
}
