import { useEffect, useRef, useState } from "react";

import { previewInference, type PrelabelConfig, type PreviewInferenceResponse, type PreviewInferenceTaskKind } from "../api";
import { createPreviewScheduler } from "../workspace/previewScheduler";
import { toHookError, type HookError } from "./hookError";

export interface CapturedPreviewFrame {
  blob: Blob;
  filename?: string;
  width?: number | null;
  height?: number | null;
}

type PreviewInferenceStatus = "idle" | "loading" | "ready" | "error";

interface UsePreviewInferenceParams {
  enabled: boolean;
  projectId: string | null;
  taskId: string | null;
  taskKind: PreviewInferenceTaskKind | null;
  captureFrame: () => Promise<CapturedPreviewFrame | null>;
  prelabelConfig?: PrelabelConfig | null;
  deploymentId?: string | null;
  topK?: number;
  refreshToken?: string | number | null;
  immediateRefreshToken?: string | number | null;
  minIntervalMs?: number;
}

export function usePreviewInference({
  enabled,
  projectId,
  taskId,
  taskKind,
  captureFrame,
  prelabelConfig = null,
  deploymentId = null,
  topK = 5,
  refreshToken = null,
  immediateRefreshToken = null,
  minIntervalMs = 0,
}: UsePreviewInferenceParams) {
  const captureFrameRef = useRef(captureFrame);
  const projectIdRef = useRef(projectId);
  const taskIdRef = useRef(taskId);
  const taskKindRef = useRef(taskKind);
  const prelabelConfigRef = useRef(prelabelConfig);
  const deploymentIdRef = useRef(deploymentId);
  const topKRef = useRef(topK);
  const minIntervalMsRef = useRef(minIntervalMs);
  const requestIdRef = useRef(0);
  const activeControllerRef = useRef<AbortController | null>(null);
  const schedulerRef = useRef<ReturnType<typeof createPreviewScheduler> | null>(null);

  const [status, setStatus] = useState<PreviewInferenceStatus>("idle");
  const [response, setResponse] = useState<PreviewInferenceResponse | null>(null);
  const [error, setError] = useState<HookError | null>(null);

  captureFrameRef.current = captureFrame;
  projectIdRef.current = projectId;
  taskIdRef.current = taskId;
  taskKindRef.current = taskKind;
  prelabelConfigRef.current = prelabelConfig;
  deploymentIdRef.current = deploymentId;
  topKRef.current = topK;
  minIntervalMsRef.current = minIntervalMs;

  useEffect(() => {
    schedulerRef.current = createPreviewScheduler({
      getMinIntervalMs: () => minIntervalMsRef.current,
      onRun: async () => {
        const nextProjectId = projectIdRef.current;
        const nextTaskId = taskIdRef.current;
        const nextTaskKind = taskKindRef.current;
        const nextPrelabelConfig = prelabelConfigRef.current;
        const nextDeploymentId = deploymentIdRef.current;
        const nextTopK = topKRef.current;
        const canRun =
          enabled &&
          Boolean(nextProjectId) &&
          Boolean(nextTaskId) &&
          Boolean(nextTaskKind) &&
          (nextTaskKind === "classification" || Boolean(nextPrelabelConfig));

        if (!canRun || !nextProjectId || !nextTaskId || !nextTaskKind) {
          setStatus("idle");
          setResponse(null);
          setError(null);
          return;
        }

        const requestId = requestIdRef.current + 1;
        requestIdRef.current = requestId;
        const controller = new AbortController();
        activeControllerRef.current = controller;

        try {
          setStatus("loading");
          setError(null);

          const captured = await captureFrameRef.current();
          if (controller.signal.aborted || requestIdRef.current !== requestId) return;
          if (!captured?.blob) {
            setStatus("idle");
            setResponse(null);
            setError(null);
            return;
          }

          const nextResponse = await previewInference(nextProjectId, nextTaskId, {
            file: captured.blob,
            filename: captured.filename,
            taskKind: nextTaskKind,
            prelabelConfig: nextTaskKind === "classification" ? null : nextPrelabelConfig,
            deploymentId: nextTaskKind === "classification" ? nextDeploymentId : null,
            topK: nextTopK,
            signal: controller.signal,
          });
          if (controller.signal.aborted || requestIdRef.current !== requestId) return;

          setResponse(nextResponse);
          setStatus("ready");
        } catch (err) {
          if (controller.signal.aborted || requestIdRef.current !== requestId) return;
          setResponse(null);
          setError(toHookError(err, "Failed to analyze the preview"));
          setStatus("error");
        } finally {
          if (activeControllerRef.current === controller) {
            activeControllerRef.current = null;
          }
        }
      },
    });

    return () => {
      activeControllerRef.current?.abort();
      schedulerRef.current?.dispose();
      schedulerRef.current = null;
    };
  }, [enabled]);

  useEffect(() => {
    const canRun =
      enabled &&
      Boolean(projectId) &&
      Boolean(taskId) &&
      Boolean(taskKind) &&
      (taskKind === "classification" || Boolean(prelabelConfig));

    if (!canRun || !projectId || !taskId || !taskKind) {
      requestIdRef.current += 1;
      activeControllerRef.current?.abort();
      schedulerRef.current?.cancel();
      setStatus("idle");
      setResponse(null);
      setError(null);
      return;
    }

    schedulerRef.current?.requestRun({ immediate: true });
  }, [deploymentId, enabled, prelabelConfig, projectId, taskId, taskKind, topK]);

  useEffect(() => {
    if (refreshToken === null || refreshToken === undefined) return;

    const canRun =
      enabled &&
      Boolean(projectId) &&
      Boolean(taskId) &&
      Boolean(taskKind) &&
      (taskKind === "classification" || Boolean(prelabelConfig));
    if (!canRun) return;
    schedulerRef.current?.requestRun({ immediate: false });
  }, [enabled, prelabelConfig, projectId, refreshToken, taskId, taskKind]);

  useEffect(() => {
    if (immediateRefreshToken === null || immediateRefreshToken === undefined) return;

    const canRun =
      enabled &&
      Boolean(projectId) &&
      Boolean(taskId) &&
      Boolean(taskKind) &&
      (taskKind === "classification" || Boolean(prelabelConfig));
    if (!canRun) return;
    schedulerRef.current?.requestRun({ immediate: true });
  }, [enabled, immediateRefreshToken, prelabelConfig, projectId, taskId, taskKind]);

  return {
    status,
    response,
    error,
    isLoading: status === "loading",
    isReady: status === "ready",
  };
}
