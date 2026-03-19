export interface CapturedVideoFrame {
  blob: Blob;
  width: number;
  height: number;
  filename: string;
}

export function captureVideoFrame(
  videoNode: HTMLVideoElement | null,
  options?: { mimeType?: string; quality?: number; filename?: string; maxDimension?: number },
): Promise<CapturedVideoFrame | null>;

export function previewDimensions(
  width: number,
  height: number,
  maxDimension?: number,
): { width: number; height: number } | null;

export function subscribeToMediaFrames(
  videoNode: HTMLVideoElement | null,
  options: { minIntervalMs?: number; onTick: () => void },
): () => void;

export function formatMediaDuration(seconds: number | null | undefined): string;
