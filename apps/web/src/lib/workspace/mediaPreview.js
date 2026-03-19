function canvasToBlob(canvas, mimeType = "image/jpeg", quality = 0.92) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Failed to encode preview frame"));
    }, mimeType, quality);
  });
}

const captureCanvasBySource = new WeakMap();

function previewDimensions(width, height, maxDimension = 960) {
  const safeWidth = Number(width);
  const safeHeight = Number(height);
  const safeMaxDimension = Number(maxDimension);
  if (!Number.isFinite(safeWidth) || safeWidth <= 0 || !Number.isFinite(safeHeight) || safeHeight <= 0) {
    return null;
  }
  if (!Number.isFinite(safeMaxDimension) || safeMaxDimension <= 0) {
    return {
      width: Math.round(safeWidth),
      height: Math.round(safeHeight),
    };
  }

  const longestSide = Math.max(safeWidth, safeHeight);
  if (longestSide <= safeMaxDimension) {
    return {
      width: Math.round(safeWidth),
      height: Math.round(safeHeight),
    };
  }

  const scale = safeMaxDimension / longestSide;
  return {
    width: Math.max(1, Math.round(safeWidth * scale)),
    height: Math.max(1, Math.round(safeHeight * scale)),
  };
}

function resolveCaptureCanvas(videoNode) {
  if (!videoNode || (typeof videoNode !== "object" && typeof videoNode !== "function")) {
    return document.createElement("canvas");
  }
  const existingCanvas = captureCanvasBySource.get(videoNode);
  if (existingCanvas) return existingCanvas;
  const nextCanvas = document.createElement("canvas");
  captureCanvasBySource.set(videoNode, nextCanvas);
  return nextCanvas;
}

async function captureVideoFrame(videoNode, options = {}) {
  if (!videoNode || typeof videoNode.videoWidth !== "number" || typeof videoNode.videoHeight !== "number") {
    return null;
  }
  if (videoNode.videoWidth <= 0 || videoNode.videoHeight <= 0) return null;

  const targetSize = previewDimensions(videoNode.videoWidth, videoNode.videoHeight, options.maxDimension ?? null);
  if (!targetSize) return null;

  const canvas = resolveCaptureCanvas(videoNode);
  canvas.width = targetSize.width;
  canvas.height = targetSize.height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Failed to initialize preview canvas");
  context.drawImage(videoNode, 0, 0, canvas.width, canvas.height);
  const blob = await canvasToBlob(canvas, options.mimeType || "image/jpeg", options.quality ?? 0.92);
  return {
    blob,
    width: canvas.width,
    height: canvas.height,
    filename: options.filename || "preview.jpg",
  };
}

function formatMediaDuration(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "Unknown duration";
  const wholeSeconds = Math.round(seconds);
  const minutes = Math.floor(wholeSeconds / 60);
  const remainder = wholeSeconds % 60;
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function mediaClockNow() {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return performance.now();
  }
  return Date.now();
}

function subscribeToMediaFrames(mediaNode, options = {}) {
  if (!mediaNode || typeof options.onTick !== "function") return () => {};

  const minIntervalMs = Math.max(0, Number(options.minIntervalMs ?? 500));
  let lastTickAt = -Infinity;
  let disposed = false;
  let timerId = null;
  let frameCallbackId = null;

  const maybeTick = (timestamp) => {
    const now = Number.isFinite(timestamp) ? timestamp : mediaClockNow();
    if (now - lastTickAt < minIntervalMs) return;
    lastTickAt = now;
    options.onTick();
  };

  if (typeof mediaNode.requestVideoFrameCallback === "function") {
    const scheduleFrame = () => {
      frameCallbackId = mediaNode.requestVideoFrameCallback((timestamp) => {
        if (disposed) return;
        maybeTick(timestamp);
        if (!disposed) scheduleFrame();
      });
    };
    scheduleFrame();
    return () => {
      disposed = true;
      if (frameCallbackId !== null && typeof mediaNode.cancelVideoFrameCallback === "function") {
        mediaNode.cancelVideoFrameCallback(frameCallbackId);
      }
    };
  }

  timerId = setInterval(() => {
    if (mediaNode.paused === true || mediaNode.ended === true) return;
    maybeTick(mediaClockNow());
  }, Math.max(50, minIntervalMs || 50));

  return () => {
    disposed = true;
    if (timerId !== null) clearInterval(timerId);
  };
}

module.exports = {
  captureVideoFrame,
  formatMediaDuration,
  previewDimensions,
  subscribeToMediaFrames,
};
