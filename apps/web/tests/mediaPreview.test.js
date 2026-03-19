const test = require("node:test");
const assert = require("node:assert/strict");

const { captureVideoFrame, formatMediaDuration, previewDimensions, subscribeToMediaFrames } = require("../src/lib/workspace/mediaPreview.js");

test("formatMediaDuration renders mm:ss and handles invalid input safely", () => {
  assert.equal(formatMediaDuration(0), "0:00");
  assert.equal(formatMediaDuration(125.2), "2:05");
  assert.equal(formatMediaDuration(null), "Unknown duration");
  assert.equal(formatMediaDuration(-3), "Unknown duration");
});

test("captureVideoFrame encodes the current video frame into a preview blob", async () => {
  const originalDocument = global.document;
  let drawnImage = null;
  let encodedQuality = null;
  const mockCanvas = {
    width: 0,
    height: 0,
    getContext(kind) {
      assert.equal(kind, "2d");
      return {
        drawImage(image, x, y, width, height) {
          drawnImage = { image, x, y, width, height };
        },
      };
    },
    toBlob(callback, mimeType, quality) {
      encodedQuality = quality;
      callback(new Blob(["preview"], { type: mimeType }));
    },
  };

  global.document = {
    createElement(tagName) {
      assert.equal(tagName, "canvas");
      return mockCanvas;
    },
  };

  try {
    const videoNode = { videoWidth: 1280, videoHeight: 720 };
    const frame = await captureVideoFrame(videoNode, { filename: "frame.jpg", mimeType: "image/jpeg", quality: 0.8 });
    assert.ok(frame);
    assert.equal(frame.width, 1280);
    assert.equal(frame.height, 720);
    assert.equal(frame.filename, "frame.jpg");
    assert.equal(frame.blob.type, "image/jpeg");
    assert.equal(encodedQuality, 0.8);
    assert.deepEqual(drawnImage, { image: videoNode, x: 0, y: 0, width: 1280, height: 720 });
  } finally {
    global.document = originalDocument;
  }
});

test("captureVideoFrame reuses the same canvas for repeated captures from one media source", async () => {
  const originalDocument = global.document;
  let createdCanvasCount = 0;
  let drawCount = 0;
  const mockCanvas = {
    width: 0,
    height: 0,
    getContext() {
      return {
        drawImage() {
          drawCount += 1;
        },
      };
    },
    toBlob(callback, mimeType) {
      callback(new Blob(["preview"], { type: mimeType }));
    },
  };

  global.document = {
    createElement(tagName) {
      assert.equal(tagName, "canvas");
      createdCanvasCount += 1;
      return mockCanvas;
    },
  };

  try {
    const videoNode = { videoWidth: 640, videoHeight: 480 };
    await captureVideoFrame(videoNode, { filename: "one.jpg" });
    await captureVideoFrame(videoNode, { filename: "two.jpg" });
    assert.equal(createdCanvasCount, 1);
    assert.equal(drawCount, 2);
  } finally {
    global.document = originalDocument;
  }
});

test("captureVideoFrame downscales preview frames before encoding when requested", async () => {
  const originalDocument = global.document;
  let drawnImage = null;
  let encodedQuality = null;
  const mockCanvas = {
    width: 0,
    height: 0,
    getContext() {
      return {
        drawImage(image, x, y, width, height) {
          drawnImage = { image, x, y, width, height };
        },
      };
    },
    toBlob(callback, mimeType, quality) {
      encodedQuality = quality;
      callback(new Blob(["preview"], { type: mimeType }));
    },
  };

  global.document = {
    createElement() {
      return mockCanvas;
    },
  };

  try {
    const videoNode = { videoWidth: 1920, videoHeight: 1080 };
    const frame = await captureVideoFrame(videoNode, { maxDimension: 960, quality: 0.75, filename: "scaled.jpg" });
    assert.ok(frame);
    assert.equal(frame.width, 960);
    assert.equal(frame.height, 540);
    assert.equal(encodedQuality, 0.75);
    assert.deepEqual(drawnImage, { image: videoNode, x: 0, y: 0, width: 960, height: 540 });
  } finally {
    global.document = originalDocument;
  }
});

test("captureVideoFrame returns null when the media element has no loaded dimensions", async () => {
  const frame = await captureVideoFrame({ videoWidth: 0, videoHeight: 720 });
  assert.equal(frame, null);
});

test("previewDimensions preserves aspect ratio while constraining the longest side", () => {
  assert.deepEqual(previewDimensions(1920, 1080, 960), { width: 960, height: 540 });
  assert.deepEqual(previewDimensions(800, 1200, 960), { width: 640, height: 960 });
  assert.deepEqual(previewDimensions(640, 480, null), { width: 640, height: 480 });
});

test("subscribeToMediaFrames uses requestVideoFrameCallback when available and throttles ticks", () => {
  const tickTimes = [];
  let callback = null;
  let nextId = 1;
  let cancelledId = null;
  const videoNode = {
    requestVideoFrameCallback(nextCallback) {
      callback = nextCallback;
      return nextId++;
    },
    cancelVideoFrameCallback(id) {
      cancelledId = id;
    },
  };

  const stop = subscribeToMediaFrames(videoNode, {
    minIntervalMs: 300,
    onTick: () => {
      tickTimes.push(tickTimes.length);
    },
  });

  assert.equal(typeof callback, "function");
  callback(0);
  callback(100);
  callback(350);
  assert.equal(tickTimes.length, 2);

  stop();
  assert.equal(cancelledId, 4);
});

test("subscribeToMediaFrames falls back to a timer and skips paused media", () => {
  const originalSetInterval = global.setInterval;
  const originalClearInterval = global.clearInterval;
  let timerCallback = null;
  let clearedTimerId = null;

  global.setInterval = (callback) => {
    timerCallback = callback;
    return 42;
  };
  global.clearInterval = (timerId) => {
    clearedTimerId = timerId;
  };

  try {
    const ticks = [];
    const videoNode = { paused: true, ended: false };
    const stop = subscribeToMediaFrames(videoNode, {
      minIntervalMs: 500,
      onTick: () => ticks.push("tick"),
    });

    assert.equal(typeof timerCallback, "function");
    timerCallback();
    assert.deepEqual(ticks, []);

    videoNode.paused = false;
    timerCallback();
    assert.deepEqual(ticks, ["tick"]);

    stop();
    assert.equal(clearedTimerId, 42);
  } finally {
    global.setInterval = originalSetInterval;
    global.clearInterval = originalClearInterval;
  }
});
