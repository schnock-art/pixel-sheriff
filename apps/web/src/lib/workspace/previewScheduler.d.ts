export function createPreviewScheduler(options: {
  getMinIntervalMs?: () => number;
  minIntervalMs?: number;
  onRun: () => Promise<void> | void;
  now?: () => number;
  scheduleTimeout?: (callback: () => void, delay: number) => unknown;
  clearScheduledTimeout?: (handle: unknown) => void;
}): {
  requestRun: (options?: { immediate?: boolean }) => void;
  cancel: () => void;
  dispose: () => void;
};
