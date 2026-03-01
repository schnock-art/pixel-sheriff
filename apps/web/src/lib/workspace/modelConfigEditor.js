function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function stableClone(value) {
  if (Array.isArray(value)) return value.map(stableClone);
  if (!isPlainObject(value)) return value;

  const next = {};
  for (const key of Object.keys(value).sort()) {
    next[key] = stableClone(value[key]);
  }
  return next;
}

function cloneModelConfig(config) {
  if (!isPlainObject(config)) return {};
  return stableClone(config);
}

function stableStringify(value) {
  return JSON.stringify(stableClone(value));
}

function isModelConfigDirty(savedConfig, draftConfig) {
  return stableStringify(savedConfig || {}) !== stableStringify(draftConfig || {});
}

function createEmbeddingAuxOutput() {
  return {
    name: "embedding",
    type: "embedding",
    source: {
      block: "backbone",
      tap: "avgpool",
    },
    projection: {
      type: "linear",
      out_dim: 256,
      normalize: "l2",
    },
  };
}

function readAuxOutputs(config) {
  const outputs = isPlainObject(config?.outputs) ? config.outputs : {};
  const aux = Array.isArray(outputs.aux) ? outputs.aux : [];
  return aux.filter((item) => isPlainObject(item));
}

function setEmbeddingAuxEnabled(config, enabled) {
  const nextConfig = cloneModelConfig(config);
  const outputs = isPlainObject(nextConfig.outputs) ? nextConfig.outputs : {};
  const aux = readAuxOutputs(nextConfig);
  const withoutEmbedding = aux.filter((item) => !(item.type === "embedding" && item.name === "embedding"));

  outputs.aux = enabled ? [...withoutEmbedding, createEmbeddingAuxOutput()] : withoutEmbedding;
  nextConfig.outputs = outputs;
  return nextConfig;
}

function setEmbeddingProjection(config, outDim, normalize) {
  const nextConfig = cloneModelConfig(config);
  const outputs = isPlainObject(nextConfig.outputs) ? nextConfig.outputs : {};
  const aux = readAuxOutputs(nextConfig);
  const nextAux = [];
  let patched = false;
  for (const item of aux) {
    if (item.type !== "embedding" || item.name !== "embedding") {
      nextAux.push(item);
      continue;
    }
    patched = true;
    const projection = isPlainObject(item.projection) ? item.projection : {};
    nextAux.push({
      ...item,
      projection: {
        ...projection,
        type: "linear",
        out_dim: outDim,
        normalize,
      },
    });
  }

  if (!patched) {
    const nextEmbedding = createEmbeddingAuxOutput();
    nextEmbedding.projection.out_dim = outDim;
    nextEmbedding.projection.normalize = normalize;
    nextAux.push(nextEmbedding);
  }

  outputs.aux = nextAux;
  nextConfig.outputs = outputs;
  return nextConfig;
}

function setSquareInputSize(config, size) {
  const normalized = Number.isFinite(size) && size > 0 ? Math.floor(size) : 1;
  const nextConfig = cloneModelConfig(config);
  const input = isPlainObject(nextConfig.input) ? nextConfig.input : {};
  input.input_size = [normalized, normalized];
  nextConfig.input = input;
  return nextConfig;
}

function setDynamicShapeFlags(config, batch, heightWidth) {
  const nextConfig = cloneModelConfig(config);
  const exportSpec = isPlainObject(nextConfig.export) ? nextConfig.export : {};
  const onnx = isPlainObject(exportSpec.onnx) ? exportSpec.onnx : {};
  const dynamicShapes = isPlainObject(onnx.dynamic_shapes) ? onnx.dynamic_shapes : {};

  dynamicShapes.batch = Boolean(batch);
  dynamicShapes.height_width = Boolean(heightWidth);
  dynamicShapes.enabled = Boolean(batch || heightWidth);
  onnx.dynamic_shapes = dynamicShapes;
  exportSpec.onnx = onnx;
  nextConfig.export = exportSpec;
  return nextConfig;
}

module.exports = {
  cloneModelConfig,
  isModelConfigDirty,
  createEmbeddingAuxOutput,
  setEmbeddingAuxEnabled,
  setEmbeddingProjection,
  setSquareInputSize,
  setDynamicShapeFlags,
};
