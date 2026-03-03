function normalizeDeviceLabel(rawDevice) {
  const normalized = String(rawDevice ?? "").trim().toLowerCase();
  if (normalized === "cuda") return "CUDA";
  if (normalized === "mps") return "MPS";
  if (normalized === "cpu") return "CPU";
  return null;
}

function runtimeBadgeLabel(runtime) {
  if (!runtime || typeof runtime !== "object") return null;
  return normalizeDeviceLabel(runtime.device_selected);
}

function mergeLogChunk(currentContent, chunk, options = {}) {
  const maxBytes = Number.isFinite(options.maxBytes) ? Math.max(1024, Math.floor(options.maxBytes)) : 200 * 1024;
  const maxLines = Number.isFinite(options.maxLines) ? Math.max(100, Math.floor(options.maxLines)) : 5000;
  const safeCurrent = typeof currentContent === "string" ? currentContent : "";
  const safeChunk = chunk && typeof chunk === "object" ? chunk : {};
  const fromByte = Number.isFinite(safeChunk.from_byte) ? Math.max(0, Math.floor(safeChunk.from_byte)) : 0;
  const toByte = Number.isFinite(safeChunk.to_byte) ? Math.max(0, Math.floor(safeChunk.to_byte)) : 0;
  const incoming = typeof safeChunk.content === "string" ? safeChunk.content : "";

  let combined = fromByte === 0 ? incoming : `${safeCurrent}${incoming}`;
  if (combined.length > maxBytes) {
    combined = combined.slice(combined.length - maxBytes);
  }

  const lines = combined.split("\n");
  if (lines.length > maxLines) {
    combined = lines.slice(lines.length - maxLines).join("\n");
  }

  return {
    content: combined,
    cursor: toByte,
  };
}

module.exports = {
  normalizeDeviceLabel,
  runtimeBadgeLabel,
  mergeLogChunk,
};
