const test = require("node:test");
const assert = require("node:assert/strict");

const {
  buildCameraDestinations,
  buildCameraFolderPath,
  buildCameraOptionLabel,
  buildCameraSessionName,
  normalizeFolderRootPath,
  sanitizeCameraFolderSegment,
} = require("../src/lib/workspace/webcamCapture.js");

test("webcam helpers sanitize folder roots and camera segments", () => {
  assert.equal(normalizeFolderRootPath("/captures/loading-bay/"), "captures/loading-bay");
  assert.equal(sanitizeCameraFolderSegment("Front Door / Cam #1"), "front-door-cam-1");
});

test("webcam helpers build readable option labels and session names", () => {
  assert.equal(buildCameraOptionLabel({ deviceId: "cam-1", label: "" }, 0), "Camera 1");
  assert.equal(buildCameraSessionName("shift-a", "Dock Camera"), "shift-a (Dock Camera)");
});

test("webcam helpers make unique folder paths against existing and reserved paths", () => {
  const first = buildCameraFolderPath({
    baseName: "shift a",
    cameraLabel: "Cam 1",
    rootFolderPath: "captures",
    existingPaths: ["captures/shift-a-cam-1"],
  });
  assert.equal(first, "captures/shift-a-cam-1-2");

  const destinations = buildCameraDestinations({
    devices: [
      { deviceId: "cam-1", label: "Dock A" },
      { deviceId: "cam-2", label: "Dock A" },
    ],
    selectedDeviceIds: ["cam-1", "cam-2"],
    sessionName: "shift a",
    rootFolderPath: "captures",
    existingPaths: [],
  });

  assert.deepEqual(
    destinations.map((item) => item.folderPath),
    ["captures/shift-a-dock-a", "captures/shift-a-dock-a-2"],
  );
  assert.deepEqual(
    destinations.map((item) => item.sequenceName),
    ["shift a (Dock A)", "shift a (Dock A) 2"],
  );
});
