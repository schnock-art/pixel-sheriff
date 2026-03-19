function defaultNow() {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return performance.now();
  }
  return Date.now();
}

function createPreviewScheduler(options) {
  const getMinIntervalMs =
    typeof options?.getMinIntervalMs === "function"
      ? options.getMinIntervalMs
      : () => Number(options?.minIntervalMs ?? 0);
  const onRun = typeof options?.onRun === "function" ? options.onRun : async () => {};
  const now = typeof options?.now === "function" ? options.now : defaultNow;
  const scheduleTimeout = typeof options?.scheduleTimeout === "function" ? options.scheduleTimeout : setTimeout;
  const clearScheduledTimeout = typeof options?.clearScheduledTimeout === "function" ? options.clearScheduledTimeout : clearTimeout;

  let disposed = false;
  let inFlight = false;
  let pending = false;
  let pendingImmediate = false;
  let lastStartedAt = -Infinity;
  let timerId = null;
  let timerDueAt = Infinity;

  const clearTimer = () => {
    if (timerId !== null) {
      clearScheduledTimeout(timerId);
      timerId = null;
      timerDueAt = Infinity;
    }
  };

  const run = async () => {
    if (disposed || inFlight) return;
    clearTimer();
    inFlight = true;
    lastStartedAt = now();
    try {
      await onRun();
    } finally {
      inFlight = false;
      if (disposed || !pending) return;
      const shouldRunImmediately = pendingImmediate;
      pending = false;
      pendingImmediate = false;
      requestRun({ immediate: shouldRunImmediately });
    }
  };

  const requestRun = ({ immediate = false } = {}) => {
    if (disposed) return;
    if (inFlight) {
      pending = true;
      pendingImmediate = pendingImmediate || immediate;
      return;
    }

    const safeMinIntervalMs = Math.max(0, Number(getMinIntervalMs() ?? 0));
    const delay = immediate ? 0 : Math.max(0, safeMinIntervalMs - (now() - lastStartedAt));
    const dueAt = now() + delay;
    if (timerId !== null && timerDueAt <= dueAt) return;

    clearTimer();
    timerDueAt = dueAt;
    timerId = scheduleTimeout(() => {
      timerId = null;
      timerDueAt = Infinity;
      void run();
    }, delay);
  };

  return {
    requestRun,
    cancel() {
      pending = false;
      pendingImmediate = false;
      clearTimer();
    },
    dispose() {
      disposed = true;
      pending = false;
      pendingImmediate = false;
      clearTimer();
    },
  };
}

module.exports = {
  createPreviewScheduler,
};
