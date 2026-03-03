function onnxStatusLabel(onnxInfo, experimentStatus) {
  const status = onnxInfo && typeof onnxInfo.status === "string" ? onnxInfo.status : "";
  if (status === "exported") return "Exported";
  if (status === "failed") return "Failed";
  if (experimentStatus === "completed") return "Pending";
  return "Pending";
}

function onnxInputShapeText(onnxInfo) {
  const shape = onnxInfo && Array.isArray(onnxInfo.input_shape) ? onnxInfo.input_shape.filter((value) => Number.isFinite(value)) : [];
  if (!shape.length) return "-";
  return shape.join(" x ");
}

function onnxClassNamesText(onnxInfo, options = {}) {
  const maxNames = Number.isFinite(options.maxNames) ? Math.max(1, Math.floor(options.maxNames)) : 6;
  const names = onnxInfo && Array.isArray(onnxInfo.class_names) ? onnxInfo.class_names.filter((row) => typeof row === "string") : [];
  if (!names.length) return "-";
  if (names.length <= maxNames) return names.join(", ");
  return `${names.slice(0, maxNames).join(", ")} (+${names.length - maxNames})`;
}

function onnxValidationText(onnxInfo) {
  const validation = onnxInfo && typeof onnxInfo.validation === "object" ? onnxInfo.validation : null;
  if (!validation || typeof validation !== "object") return "-";
  const status = typeof validation.status === "string" ? validation.status.toLowerCase() : "";
  if (status === "passed") return "Validated with ONNX Runtime";
  if (status === "failed") return "ONNX Runtime validation failed";
  return "-";
}

module.exports = {
  onnxStatusLabel,
  onnxInputShapeText,
  onnxClassNamesText,
  onnxValidationText,
};
