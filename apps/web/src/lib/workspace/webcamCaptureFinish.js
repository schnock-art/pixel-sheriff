function collectPrelabelSessionIds(devices) {
  return Array.from(
    new Set(
      (Array.isArray(devices) ? devices : [])
        .map((device) => (device && typeof device.prelabelSessionId === "string" ? device.prelabelSessionId.trim() : ""))
        .filter(Boolean),
    ),
  );
}

async function finishWebcamCapture({
  projectId,
  taskId,
  devices,
  sequences,
  stopCapture,
  waitForPendingUploads,
  stopPreview,
  closePrelabelInput,
}) {
  const finishedSequences = Array.isArray(sequences) ? sequences.slice() : [];
  const prelabelSessionIds =
    projectId && taskId ? collectPrelabelSessionIds(devices) : [];

  if (typeof stopCapture === "function") stopCapture();
  if (typeof waitForPendingUploads === "function") await waitForPendingUploads();
  if (projectId && taskId && typeof closePrelabelInput === "function" && prelabelSessionIds.length > 0) {
    await Promise.allSettled(prelabelSessionIds.map((sessionId) => closePrelabelInput(projectId, taskId, sessionId)));
  }
  if (typeof stopPreview === "function") stopPreview();

  return {
    closedSessionIds: prelabelSessionIds,
    sequences: finishedSequences,
  };
}

module.exports = {
  collectPrelabelSessionIds,
  finishWebcamCapture,
};
