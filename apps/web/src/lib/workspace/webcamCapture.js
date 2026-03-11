function normalizeFolderRootPath(folderPath) {
  return String(folderPath || "")
    .replaceAll("\\", "/")
    .trim()
    .replace(/^\/+|\/+$/g, "");
}

function sanitizeCameraFolderSegment(value, fallback = "camera") {
  const normalized = String(value || "")
    .toLowerCase()
    .replaceAll("\\", " ")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return normalized || fallback;
}

function buildCameraOptionLabel(device, index) {
  const rawLabel = typeof device?.label === "string" ? device.label.trim() : "";
  return rawLabel || `Camera ${index + 1}`;
}

function buildCameraSessionName(baseName, cameraLabel) {
  const sessionName = String(baseName || "").trim() || "webcam";
  const label = String(cameraLabel || "").trim() || "Camera";
  return `${sessionName} (${label})`;
}

function buildCameraFolderPath({ baseName, cameraLabel, rootFolderPath = "", existingPaths = [], reservedPaths = [] }) {
  const normalizedRoot = normalizeFolderRootPath(rootFolderPath);
  const sessionSlug = sanitizeCameraFolderSegment(baseName, "webcam");
  const cameraSlug = sanitizeCameraFolderSegment(cameraLabel, "camera");
  const existing = new Set(
    [...existingPaths, ...reservedPaths]
      .map((value) => normalizeFolderRootPath(value))
      .filter(Boolean),
  );

  const baseLeaf = `${sessionSlug}-${cameraSlug}`;
  let candidate = normalizedRoot ? `${normalizedRoot}/${baseLeaf}` : baseLeaf;
  let suffix = 2;
  while (existing.has(candidate)) {
    candidate = normalizedRoot ? `${normalizedRoot}/${baseLeaf}-${suffix}` : `${baseLeaf}-${suffix}`;
    suffix += 1;
  }
  return candidate;
}

function buildCameraDestinations({ devices, selectedDeviceIds, sessionName, rootFolderPath = "", existingPaths = [] }) {
  const reservedPaths = [];
  const reservedNames = new Set();
  return selectedDeviceIds.map((deviceId, index) => {
    const device = devices.find((item) => item.deviceId === deviceId);
    const cameraLabel = buildCameraOptionLabel(device, index);
    const folderPath = buildCameraFolderPath({
      baseName: sessionName,
      cameraLabel,
      rootFolderPath,
      existingPaths,
      reservedPaths,
    });
    reservedPaths.push(folderPath);
    const baseSequenceName = buildCameraSessionName(sessionName, cameraLabel);
    let sequenceName = baseSequenceName;
    let suffix = 2;
    while (reservedNames.has(sequenceName)) {
      sequenceName = `${baseSequenceName} ${suffix}`;
      suffix += 1;
    }
    reservedNames.add(sequenceName);
    return {
      deviceId,
      cameraLabel,
      folderPath,
      sequenceName,
    };
  });
}

module.exports = {
  normalizeFolderRootPath,
  sanitizeCameraFolderSegment,
  buildCameraOptionLabel,
  buildCameraSessionName,
  buildCameraFolderPath,
  buildCameraDestinations,
};
