function asString(value, fallback = "-") {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function asNumber(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function formatInputSize(inputSize) {
  const values = asArray(inputSize).filter((value) => typeof value === "number" && Number.isFinite(value));
  if (values.length !== 2) return "-";
  return `${values[0]} x ${values[1]}`;
}

function readModelSummary(config) {
  const cfg = config && typeof config === "object" ? config : {};
  const source = cfg.source_dataset && typeof cfg.source_dataset === "object" ? cfg.source_dataset : {};
  const input = cfg.input && typeof cfg.input === "object" ? cfg.input : {};
  const normalization = input.normalization && typeof input.normalization === "object" ? input.normalization : {};
  const architecture = cfg.architecture && typeof cfg.architecture === "object" ? cfg.architecture : {};
  const backbone = architecture.backbone && typeof architecture.backbone === "object" ? architecture.backbone : {};
  const neck = architecture.neck && typeof architecture.neck === "object" ? architecture.neck : {};
  const head = architecture.head && typeof architecture.head === "object" ? architecture.head : {};
  const outputs = cfg.outputs && typeof cfg.outputs === "object" ? cfg.outputs : {};
  const primaryOutput = outputs.primary && typeof outputs.primary === "object" ? outputs.primary : {};
  const exportSpec = cfg.export && typeof cfg.export === "object" ? cfg.export : {};
  const onnx = exportSpec.onnx && typeof exportSpec.onnx === "object" ? exportSpec.onnx : {};
  const dynamicShapes = onnx.dynamic_shapes && typeof onnx.dynamic_shapes === "object" ? onnx.dynamic_shapes : {};
  const classNames = asArray(source.class_names).filter((item) => typeof item === "string");

  return {
    task: asString(source.task),
    numClasses: asNumber(source.num_classes, 0),
    classNames,
    classNamesText: classNames.length > 0 ? classNames.join(", ") : "-",
    inputSizeText: formatInputSize(input.input_size),
    resizePolicy: asString(input.resize_policy),
    normalizationType: asString(normalization.type),
    architectureFamily: asString(architecture.family),
    backboneName: asString(backbone.name),
    neckType: asString(neck.type),
    headType: asString(head.type),
    primaryOutputFormat: asString(primaryOutput.format),
    onnxEnabled: Boolean(onnx.enabled),
    onnxOpset: asNumber(onnx.opset, 0),
    dynamicBatch: Boolean(dynamicShapes.batch),
    dynamicHeightWidth: Boolean(dynamicShapes.height_width),
  };
}

module.exports = {
  readModelSummary,
};

