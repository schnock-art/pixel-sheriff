import { useCallback, useEffect, useRef, useState } from "react";

import { createWebcamSession, uploadSequenceFrame, type Asset, type AssetSequence } from "../api";

interface StartCaptureParams {
  name: string;
  fps: number;
  folderId?: string | null;
}

interface UseWebcamCaptureParams {
  projectId: string | null;
  taskId: string | null;
  onSequenceCreated?: (sequence: AssetSequence) => void;
  onFrameUploaded?: (asset: Asset, sequence: AssetSequence) => void;
}

function streamTracks(stream: MediaStream | null): MediaStreamTrack[] {
  return stream ? stream.getTracks() : [];
}

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Failed to encode webcam frame"));
    }, "image/jpeg", 0.9);
  });
}

export function useWebcamCapture({ projectId, taskId, onSequenceCreated, onFrameUploaded }: UseWebcamCaptureParams) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureTimerRef = useRef<number | null>(null);
  const uploadInFlightRef = useRef(false);
  const frameIndexRef = useRef(0);
  const startedAtRef = useRef(0);
  const sequenceRef = useRef<AssetSequence | null>(null);

  const [error, setError] = useState<string | null>(null);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [captureCount, setCaptureCount] = useState(0);
  const [sequence, setSequence] = useState<AssetSequence | null>(null);

  const stopPreview = useCallback(() => {
    if (captureTimerRef.current) {
      window.clearInterval(captureTimerRef.current);
      captureTimerRef.current = null;
    }
    setIsCapturing(false);
    setIsPreviewing(false);
    uploadInFlightRef.current = false;
    sequenceRef.current = null;
    for (const track of streamTracks(streamRef.current)) track.stop();
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }
  }, []);

  useEffect(() => () => stopPreview(), [stopPreview]);

  const requestPreview = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser does not support webcam capture.");
      return;
    }

    try {
      setError(null);
      setSequence(null);
      setCaptureCount(0);
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsPreviewing(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to access the webcam.");
      setIsPreviewing(false);
    }
  }, []);

  const stopCapture = useCallback(() => {
    if (captureTimerRef.current) {
      window.clearInterval(captureTimerRef.current);
      captureTimerRef.current = null;
    }
    setIsCapturing(false);
  }, []);

  const captureOnce = useCallback(async () => {
    const activeSequence = sequenceRef.current;
    if (!projectId || !activeSequence || !videoRef.current || uploadInFlightRef.current) return;
    const video = videoRef.current;
    if (video.videoWidth <= 0 || video.videoHeight <= 0) return;

    uploadInFlightRef.current = true;
    try {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Failed to initialize webcam canvas.");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await canvasToBlob(canvas);
      const frameIndex = frameIndexRef.current;
      const elapsedMs = Date.now() - startedAtRef.current;
      const uploaded = await uploadSequenceFrame(
        projectId,
        activeSequence.id,
        blob,
        `frame_${String(frameIndex + 1).padStart(6, "0")}.jpg`,
        frameIndex,
        elapsedMs / 1000,
      );
      frameIndexRef.current += 1;
      setCaptureCount(frameIndexRef.current);
      onFrameUploaded?.(uploaded, activeSequence);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload webcam frame.");
      stopCapture();
    } finally {
      uploadInFlightRef.current = false;
    }
  }, [onFrameUploaded, projectId, stopCapture]);

  const startCapture = useCallback(async ({ name, fps, folderId }: StartCaptureParams) => {
    if (!projectId) {
      setError("Select a project before starting webcam capture.");
      return;
    }
    if (!isPreviewing) {
      setError("Start the webcam preview before capturing.");
      return;
    }

    try {
      setError(null);
      frameIndexRef.current = 0;
      setCaptureCount(0);
      startedAtRef.current = Date.now();
      const created = await createWebcamSession(projectId, {
        task_id: taskId,
        folder_id: folderId ?? null,
        name,
        fps,
      });
      sequenceRef.current = created.sequence;
      setSequence(created.sequence);
      onSequenceCreated?.(created.sequence);
      setIsCapturing(true);
      captureTimerRef.current = window.setInterval(() => {
        void captureOnce();
      }, Math.max(1000 / Math.max(fps, 0.1), 150));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start webcam capture.");
      setIsCapturing(false);
    }
  }, [captureOnce, isPreviewing, onSequenceCreated, projectId, taskId]);

  return {
    videoRef,
    error,
    sequence,
    captureCount,
    isPreviewing,
    isCapturing,
    requestPreview,
    stopPreview,
    startCapture,
    stopCapture,
  };
}
