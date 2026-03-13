const test = require("node:test");
const assert = require("node:assert/strict");

const { collectPrelabelSessionIds, finishWebcamCapture } = require("../src/lib/workspace/webcamCaptureFinish.js");

test("collectPrelabelSessionIds trims blanks and dedupes session ids", () => {
  assert.deepEqual(
    collectPrelabelSessionIds([
      { prelabelSessionId: " session-a " },
      { prelabelSessionId: "session-b" },
      { prelabelSessionId: "session-a" },
      { prelabelSessionId: null },
    ]),
    ["session-a", "session-b"],
  );
});

test("finishWebcamCapture drains uploads before closing prelabel sessions and preview", async () => {
  const calls = [];
  const result = await finishWebcamCapture({
    projectId: "project-1",
    taskId: "task-1",
    devices: [
      { prelabelSessionId: "session-a" },
      { prelabelSessionId: "session-a" },
      { prelabelSessionId: "session-b" },
      { prelabelSessionId: null },
    ],
    sequences: [{ id: "seq-1" }, { id: "seq-2" }],
    stopCapture: () => {
      calls.push("stopCapture");
    },
    waitForPendingUploads: async () => {
      calls.push("waitForPendingUploads");
    },
    closePrelabelInput: async (_projectId, _taskId, sessionId) => {
      calls.push(`close:${sessionId}`);
    },
    stopPreview: () => {
      calls.push("stopPreview");
    },
  });

  assert.deepEqual(calls, [
    "stopCapture",
    "waitForPendingUploads",
    "close:session-a",
    "close:session-b",
    "stopPreview",
  ]);
  assert.deepEqual(result.closedSessionIds, ["session-a", "session-b"]);
  assert.deepEqual(result.sequences, [{ id: "seq-1" }, { id: "seq-2" }]);
});
