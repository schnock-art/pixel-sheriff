import { useEffect, useMemo, useRef, useState } from "react";

import type { PrelabelConfig, TaskKind, VideoImportPayload } from "../../../lib/api";
import { useCompatibleDeployments } from "../../../lib/hooks/useCompatibleDeployments";
import { usePreviewInference } from "../../../lib/hooks/usePreviewInference";
import { captureVideoFrame, formatMediaDuration, subscribeToMediaFrames } from "../../../lib/workspace/mediaPreview";
import { DeploymentSelectField } from "./DeploymentSelectField";
import { MediaImportWorkspaceModal } from "./MediaImportWorkspaceModal";
import { ModalMediaPreview } from "./ModalMediaPreview";
import { PrelabelSettingsSection } from "./PrelabelSettingsSection";

interface VideoImportModalProps {
  open: boolean;
  projectId: string | null;
  taskId: string | null;
  taskKind: TaskKind | null;
  defaultName: string;
  isImporting: boolean;
  errorMessage: string | null;
  enablePrelabels: boolean;
  defaultPrompts: string[];
  onClose: () => void;
  onSubmit: (file: File, payload: VideoImportPayload) => void;
}

export function VideoImportModal({
  open,
  projectId,
  taskId,
  taskKind,
  defaultName,
  isImporting,
  errorMessage,
  enablePrelabels,
  defaultPrompts,
  onClose,
  onSubmit,
}: VideoImportModalProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [name, setName] = useState(defaultName);
  const [fps, setFps] = useState("2");
  const [maxFrames, setMaxFrames] = useState("500");
  const [resolution, setResolution] = useState<"original" | "1280" | "720">("original");
  const [prelabelConfig, setPrelabelConfig] = useState<PrelabelConfig | null>(null);
  const [selectedClassificationDeploymentId, setSelectedClassificationDeploymentId] = useState<string | null>(null);
  const [videoMeta, setVideoMeta] = useState<{ width: number; height: number; duration: number } | null>(null);
  const [previewTick, setPreviewTick] = useState(0);
  const [previewEventTick, setPreviewEventTick] = useState(0);
  const [isPreviewPlaying, setIsPreviewPlaying] = useState(false);
  const previewTaskKind = taskKind === "classification" || taskKind === "bbox" ? taskKind : null;
  const deployments = useCompatibleDeployments({ projectId, taskId, taskKind: previewTaskKind });

  useEffect(() => {
    if (!open) return;
    setSelectedFile(null);
    setPreviewUrl(null);
    setVideoMeta(null);
    setPreviewTick(0);
    setPreviewEventTick(0);
    setIsPreviewPlaying(false);
    setName(defaultName);
    setFps("2");
    setMaxFrames("500");
    setResolution("original");
    setPrelabelConfig(
      enablePrelabels
        ? {
            source_type: "florence2",
            deployment_id: null,
            prompts: defaultPrompts,
            frame_sampling: { mode: "every_n_frames", value: 15 },
            confidence_threshold: 0.25,
            max_detections_per_frame: 20,
          }
        : null,
    );
    setSelectedClassificationDeploymentId(null);
  }, [defaultName, defaultPrompts, enablePrelabels, open]);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      setVideoMeta(null);
      return;
    }
    const nextUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(nextUrl);
    setVideoMeta(null);
    return () => URL.revokeObjectURL(nextUrl);
  }, [selectedFile]);

  useEffect(() => {
    if (!open || !previewUrl || !isPreviewPlaying) return;
    if (!videoRef.current) return;
    return subscribeToMediaFrames(videoRef.current, {
      minIntervalMs: 500,
      onTick: () => setPreviewTick((value) => value + 1),
    });
  }, [isPreviewPlaying, open, previewUrl]);

  const canSubmit = useMemo(() => Boolean(selectedFile) && Number(fps) > 0 && Number(maxFrames) > 0, [fps, maxFrames, selectedFile]);

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

  const preview = usePreviewInference({
    enabled: open && Boolean(previewUrl) && Boolean(selectedFile) && (previewTaskKind !== "classification" || Boolean(selectedClassificationDeploymentId)),
    projectId,
    taskId,
    taskKind: previewTaskKind,
    captureFrame: () =>
      captureVideoFrame(videoRef.current, {
        filename: "video-preview.jpg",
        quality: 0.75,
        maxDimension: 960,
      }),
    prelabelConfig: previewTaskKind === "bbox" ? prelabelConfig : null,
    deploymentId: previewTaskKind === "classification" ? selectedClassificationDeploymentId : null,
    refreshToken: previewTick,
    immediateRefreshToken: previewEventTick,
    minIntervalMs: 500,
  });

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
      prelabel_config: prelabelConfig,
    });
  }

  const metaItems = [
    selectedFile ? `File: ${selectedFile.name}` : null,
    videoMeta ? `Source: ${videoMeta.width}x${videoMeta.height}` : null,
    videoMeta ? `Duration: ${formatMediaDuration(videoMeta.duration)}` : null,
    `Import FPS: ${fps}`,
    `Resolution: ${resolution === "original" ? "Original" : `${resolution}px wide`}`,
  ].filter((value): value is string => Boolean(value));

  return (
    <MediaImportWorkspaceModal
      title="Import Video"
      subtitle="Preview a frame, tune your AI settings, then import the sequence."
      controls={
        <>
          <section className="placeholder-card import-config-card">
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
          </section>

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
            samplingHint="Default is every 15 frames."
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
              <p>Classification preview uses the selected compatible deployment for this task. Seek or pause the video to refresh the ranking overlay.</p>
            </section>
          ) : null}

          {errorMessage ? <p className="import-field-error">{errorMessage}</p> : null}
        </>
      }
      preview={
        <ModalMediaPreview
          title="Preview"
          subtitle="The overlay is advisory only and does not save annotations yet."
          emptyMessage="Choose a video file to preview frames and AI overlays."
          mediaBasis={videoMeta}
          mediaReady={Boolean(previewUrl && videoMeta)}
          preview={preview}
          metaItems={metaItems}
        >
          {previewUrl ? (
            <video
              ref={videoRef}
              src={previewUrl}
              className="modal-media-preview-element"
              controls
              muted
              playsInline
              preload="metadata"
              data-testid="video-import-preview"
              onLoadedMetadata={(event) => {
                const node = event.currentTarget;
                setVideoMeta({
                  width: node.videoWidth,
                  height: node.videoHeight,
                  duration: Number.isFinite(node.duration) ? node.duration : 0,
                });
                setPreviewEventTick((value) => value + 1);
              }}
              onLoadedData={() => setPreviewEventTick((value) => value + 1)}
              onPlay={() => {
                setIsPreviewPlaying(true);
                setPreviewEventTick((value) => value + 1);
              }}
              onPause={() => {
                setIsPreviewPlaying(false);
                setPreviewEventTick((value) => value + 1);
              }}
              onSeeked={() => setPreviewEventTick((value) => value + 1)}
              onEnded={() => setIsPreviewPlaying(false)}
            />
          ) : null}
        </ModalMediaPreview>
      }
      footer={
        <>
          <button type="button" className="ghost-button" onClick={onClose} disabled={isImporting}>
            Cancel
          </button>
          <button type="button" className="primary-button" onClick={handleSubmit} disabled={!canSubmit || isImporting}>
            {isImporting ? "Importing..." : "Import"}
          </button>
        </>
      }
    />
  );
}
