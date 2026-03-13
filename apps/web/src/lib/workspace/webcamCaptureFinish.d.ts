import type { AssetSequence } from "../api";
import type { WebcamDeviceState } from "../hooks/useWebcamCapture";

export function collectPrelabelSessionIds(devices: WebcamDeviceState[]): string[];

export function finishWebcamCapture(params: {
  projectId: string | null;
  taskId: string | null;
  devices: WebcamDeviceState[];
  sequences: AssetSequence[];
  stopCapture: () => void;
  waitForPendingUploads: () => Promise<void>;
  stopPreview: () => void;
  closePrelabelInput: (projectId: string, taskId: string, sessionId: string) => Promise<unknown>;
}): Promise<{ closedSessionIds: string[]; sequences: AssetSequence[] }>;
