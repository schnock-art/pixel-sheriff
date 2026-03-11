import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { createWebcamSession, uploadSequenceFrame, type Asset, type AssetSequence } from "../api";

export interface WebcamCaptureDestination {
  deviceId: string;
  sequenceName: string;
  folderPath: string;
}

export interface WebcamDeviceState {
  deviceId: string;
  label: string;
  error: string | null;
  isPreviewing: boolean;
  isCapturing: boolean;
  captureCount: number;
  sequence: AssetSequence | null;
}

interface StartCaptureParams {
  fps: number;
  destinations: WebcamCaptureDestination[];
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

function resolveCameraLabel(device: MediaDeviceInfo, index: number): string {
  return device.label.trim() || `Camera ${index + 1}`;
}

export function useWebcamCapture({ projectId, taskId, onSequenceCreated, onFrameUploaded }: UseWebcamCaptureParams) {
  const videoNodeRefs = useRef(new Map<string, HTMLVideoElement | null>());
  const streamRefs = useRef(new Map<string, MediaStream>());
  const captureTimerRefs = useRef(new Map<string, number>());
  const uploadInFlightRefs = useRef(new Set<string>());
  const frameIndexRefs = useRef(new Map<string, number>());
  const startedAtRefs = useRef(new Map<string, number>());
  const sequenceRefs = useRef(new Map<string, AssetSequence>());
  const selectedDeviceIdsRef = useRef<string[]>([]);

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceIds, setSelectedDeviceIdsState] = useState<string[]>([]);
  const [deviceStates, setDeviceStates] = useState<Record<string, WebcamDeviceState>>({});
  const [isLoadingDevices, setIsLoadingDevices] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const patchDeviceState = useCallback((deviceId: string, partial: Partial<WebcamDeviceState>) => {
    setDeviceStates((previous) => {
      const existing = previous[deviceId] ?? {
        deviceId,
        label: deviceId,
        error: null,
        isPreviewing: false,
        isCapturing: false,
        captureCount: 0,
        sequence: null,
      };
      return {
        ...previous,
        [deviceId]: {
          ...existing,
          ...partial,
        },
      };
    });
  }, []);

  const attachStreamToVideo = useCallback(async (deviceId: string) => {
    const node = videoNodeRefs.current.get(deviceId);
    const stream = streamRefs.current.get(deviceId);
    if (!node || !stream) return;
    if (node.srcObject !== stream) node.srcObject = stream;
    try {
      await node.play();
    } catch {
      // Browsers can reject play() transiently while the video element mounts.
    }
  }, []);

  const stopCapture = useCallback((deviceIds?: string[]) => {
    const ids = deviceIds ?? Array.from(captureTimerRefs.current.keys());
    for (const deviceId of ids) {
      const timerId = captureTimerRefs.current.get(deviceId);
      if (typeof timerId === "number") {
        window.clearInterval(timerId);
        captureTimerRefs.current.delete(deviceId);
      }
      uploadInFlightRefs.current.delete(deviceId);
      patchDeviceState(deviceId, { isCapturing: false });
    }
  }, [patchDeviceState]);

  const stopPreview = useCallback((deviceIds?: string[]) => {
    const ids = deviceIds ?? Array.from(new Set([...Array.from(streamRefs.current.keys()), ...selectedDeviceIdsRef.current]));
    stopCapture(ids);
    for (const deviceId of ids) {
      const stream = streamRefs.current.get(deviceId) ?? null;
      for (const track of streamTracks(stream)) track.stop();
      streamRefs.current.delete(deviceId);
      const node = videoNodeRefs.current.get(deviceId);
      if (node) {
        node.pause();
        node.srcObject = null;
      }
      patchDeviceState(deviceId, { isPreviewing: false });
    }
  }, [patchDeviceState, stopCapture]);

  const reset = useCallback(() => {
    stopPreview();
    frameIndexRefs.current.clear();
    startedAtRefs.current.clear();
    sequenceRefs.current.clear();
    setError(null);
    setDeviceStates((previous) => {
      const next: Record<string, WebcamDeviceState> = {};
      for (const [deviceId, state] of Object.entries(previous)) {
        next[deviceId] = {
          ...state,
          error: null,
          isPreviewing: false,
          isCapturing: false,
          captureCount: 0,
          sequence: null,
        };
      }
      return next;
    });
  }, [stopPreview]);

  useEffect(() => () => stopPreview(), [stopPreview]);

  const refreshDevices = useCallback(async (options?: { requestAccess?: boolean }) => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setError("This browser does not support camera selection.");
      setDevices([]);
      setSelectedDeviceIdsState([]);
      return [];
    }

    let permissionStream: MediaStream | null = null;
    try {
      setIsLoadingDevices(true);
      setError(null);
      if (options?.requestAccess && navigator.mediaDevices?.getUserMedia) {
        permissionStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }
      const availableDevices = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === "videoinput");
      setDevices(availableDevices);
      setDeviceStates((previous) => {
        const next: Record<string, WebcamDeviceState> = {};
        availableDevices.forEach((device, index) => {
          const existing = previous[device.deviceId];
          next[device.deviceId] = {
            deviceId: device.deviceId,
            label: resolveCameraLabel(device, index),
            error: existing?.error ?? null,
            isPreviewing: existing?.isPreviewing ?? false,
            isCapturing: existing?.isCapturing ?? false,
            captureCount: existing?.captureCount ?? 0,
            sequence: existing?.sequence ?? null,
          };
        });
        return next;
      });
      setSelectedDeviceIdsState((previous) => {
        const filtered = previous.filter((deviceId) => availableDevices.some((device) => device.deviceId === deviceId));
        if (filtered.length > 0) return filtered;
        return availableDevices[0] ? [availableDevices[0].deviceId] : [];
      });
      return availableDevices;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to enumerate cameras.");
      setDevices([]);
      setSelectedDeviceIdsState([]);
      return [];
    } finally {
      for (const track of streamTracks(permissionStream)) track.stop();
      setIsLoadingDevices(false);
    }
  }, []);

  const setSelectedDeviceIds = useCallback((nextDeviceIds: string[]) => {
    setSelectedDeviceIdsState((previous) => {
      const next = Array.from(new Set(nextDeviceIds));
      selectedDeviceIdsRef.current = next;
      const removed = previous.filter((deviceId) => !next.includes(deviceId));
      if (removed.length > 0) stopPreview(removed);
      return next;
    });
  }, [stopPreview]);

  const requestPreview = useCallback(async (deviceIds?: string[]) => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser does not support webcam capture.");
      return;
    }

    const refreshedDevices = await refreshDevices({ requestAccess: true });
    const currentSelection = deviceIds ?? selectedDeviceIdsRef.current;
    const targetIds = currentSelection.filter((deviceId) => refreshedDevices.some((device) => device.deviceId === deviceId));
    if (targetIds.length === 0) {
      setError("Select at least one camera.");
      return;
    }

    setError(null);
    const results = await Promise.allSettled(
      targetIds.map(async (deviceId) => {
        try {
          const existingStream = streamRefs.current.get(deviceId) ?? null;
          for (const track of streamTracks(existingStream)) track.stop();
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: { exact: deviceId } },
            audio: false,
          });
          streamRefs.current.set(deviceId, stream);
          patchDeviceState(deviceId, { error: null, isPreviewing: true });
          await attachStreamToVideo(deviceId);
        } catch (err) {
          patchDeviceState(deviceId, {
            error: err instanceof Error ? err.message : "Failed to access the camera.",
            isPreviewing: false,
          });
        }
      }),
    );

    if (results.some((result) => result.status === "fulfilled")) {
      const refreshed = await refreshDevices();
      if (refreshed.length > 0) void Promise.all(targetIds.map((deviceId) => attachStreamToVideo(deviceId)));
    }
  }, [attachStreamToVideo, patchDeviceState, refreshDevices]);

  const captureOnce = useCallback(async (deviceId: string) => {
    const sequence = sequenceRefs.current.get(deviceId);
    const node = videoNodeRefs.current.get(deviceId);
    if (!projectId || !sequence || !node || uploadInFlightRefs.current.has(deviceId)) return;
    if (node.videoWidth <= 0 || node.videoHeight <= 0) return;

    uploadInFlightRefs.current.add(deviceId);
    try {
      const canvas = document.createElement("canvas");
      canvas.width = node.videoWidth;
      canvas.height = node.videoHeight;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Failed to initialize webcam canvas.");
      context.drawImage(node, 0, 0, canvas.width, canvas.height);
      const blob = await canvasToBlob(canvas);
      const frameIndex = frameIndexRefs.current.get(deviceId) ?? 0;
      const startedAt = startedAtRefs.current.get(deviceId) ?? Date.now();
      const elapsedMs = Date.now() - startedAt;
      const uploaded = await uploadSequenceFrame(
        projectId,
        sequence.id,
        blob,
        `frame_${String(frameIndex + 1).padStart(6, "0")}.jpg`,
        frameIndex,
        elapsedMs / 1000,
      );
      frameIndexRefs.current.set(deviceId, frameIndex + 1);
      patchDeviceState(deviceId, { captureCount: frameIndex + 1 });
      onFrameUploaded?.(uploaded, sequence);
    } catch (err) {
      patchDeviceState(deviceId, { error: err instanceof Error ? err.message : "Failed to upload webcam frame." });
      stopCapture([deviceId]);
    } finally {
      uploadInFlightRefs.current.delete(deviceId);
    }
  }, [onFrameUploaded, patchDeviceState, projectId, stopCapture]);

  const startCapture = useCallback(async ({ fps, destinations }: StartCaptureParams) => {
    if (!projectId) {
      setError("Select a project before starting webcam capture.");
      return;
    }
    if (destinations.length === 0) {
      setError("Select at least one previewed camera.");
      return;
    }

    setError(null);
    const previewableDestinations = destinations.filter((destination) => deviceStates[destination.deviceId]?.isPreviewing);
    if (previewableDestinations.length === 0) {
      setError("Start preview for at least one selected camera before capturing.");
      return;
    }

    await Promise.allSettled(
      previewableDestinations.map(async (destination) => {
        const { deviceId, sequenceName, folderPath } = destination;
        try {
          frameIndexRefs.current.set(deviceId, 0);
          startedAtRefs.current.set(deviceId, Date.now());
          patchDeviceState(deviceId, { captureCount: 0, error: null });
          const created = await createWebcamSession(projectId, {
            task_id: taskId,
            folder_path: folderPath,
            name: sequenceName,
            fps,
          });
          sequenceRefs.current.set(deviceId, created.sequence);
          patchDeviceState(deviceId, { sequence: created.sequence, isCapturing: true });
          onSequenceCreated?.(created.sequence);
          void captureOnce(deviceId);
          const timerId = window.setInterval(() => {
            void captureOnce(deviceId);
          }, Math.max(1000 / Math.max(fps, 0.1), 150));
          captureTimerRefs.current.set(deviceId, timerId);
        } catch (err) {
          patchDeviceState(deviceId, {
            error: err instanceof Error ? err.message : "Failed to start webcam capture.",
            isCapturing: false,
          });
        }
      }),
    );
  }, [captureOnce, deviceStates, onSequenceCreated, patchDeviceState, projectId, taskId]);

  const attachVideoRef = useCallback((deviceId: string, node: HTMLVideoElement | null) => {
    videoNodeRefs.current.set(deviceId, node);
    if (node) void attachStreamToVideo(deviceId);
  }, [attachStreamToVideo]);

  const activeDeviceStates = useMemo(
    () =>
      devices.map((device, index) => {
        const fallback = {
          deviceId: device.deviceId,
          label: resolveCameraLabel(device, index),
          error: null,
          isPreviewing: false,
          isCapturing: false,
          captureCount: 0,
          sequence: null,
        };
        return deviceStates[device.deviceId] ?? fallback;
      }),
    [deviceStates, devices],
  );

  const captureCount = useMemo(
    () => activeDeviceStates.reduce((sum, device) => sum + device.captureCount, 0),
    [activeDeviceStates],
  );
  const sequences = useMemo(
    () => activeDeviceStates.map((device) => device.sequence).filter((sequence): sequence is AssetSequence => Boolean(sequence)),
    [activeDeviceStates],
  );
  const isPreviewing = activeDeviceStates.some((device) => device.isPreviewing);
  const isCapturing = activeDeviceStates.some((device) => device.isCapturing);

  useEffect(() => {
    selectedDeviceIdsRef.current = selectedDeviceIds;
  }, [selectedDeviceIds]);

  return {
    devices: activeDeviceStates,
    selectedDeviceIds,
    setSelectedDeviceIds,
    isLoadingDevices,
    error,
    captureCount,
    isPreviewing,
    isCapturing,
    sequences,
    refreshDevices,
    requestPreview,
    stopPreview,
    startCapture,
    stopCapture,
    attachVideoRef,
    reset,
  };
}
