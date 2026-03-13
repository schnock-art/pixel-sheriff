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

function normalizeModelTask(task) {
  const normalized = typeof task === "string" ? task.trim().toLowerCase() : "";
  if (!normalized) return "classification";
  if (normalized === "bbox" || normalized === "detection") return "detection";
  if (normalized === "segmentation") return "segmentation";
  return "classification";
}

function readSourceDataset(config) {
  return isPlainObject(config?.source_dataset) ? config.source_dataset : {};
}

function readSourceDatasetNumClasses(config) {
  const sourceDataset = readSourceDataset(config);
  const value = Number(sourceDataset.num_classes);
  return Number.isInteger(value) && value > 0 ? value : 2;
}

function readSourceDatasetLabelMode(config) {
  const sourceDataset = readSourceDataset(config);
  return sourceDataset.label_mode === "multi_label" ? "multi_label" : "single_label";
}

function buildFamilyDefaults(familyName, config) {
  const numClasses = readSourceDatasetNumClasses(config);
  if (familyName === "ssdlite320_mobilenet_v3_large") {
    return {
      input: {
        input_size: [320, 320],
      },
      architecture: {
        family: "ssdlite320_mobilenet_v3_large",
        framework: "torchvision",
        precision: "fp32",
        backbone: { name: "mobilenet_v3_large", pretrained: true },
        neck: { type: "none" },
        head: { type: "ssdlite", num_classes: numClasses },
      },
      loss: { type: "ssdlite_default" },
      outputs: {
        primary: {
          name: "coco_detections",
          type: "task_output",
          task: "detection",
          format: "coco_detections",
        },
        aux: [],
      },
    };
  }

  if (familyName === "retinanet") {
    return {
      architecture: {
        family: "retinanet",
        framework: "torchvision",
        precision: "fp32",
        backbone: { name: "resnet50", pretrained: true },
        neck: { type: "fpn", fpn_channels: 256 },
        head: { type: "retinanet", num_classes: numClasses },
      },
      loss: { type: "retinanet_default" },
      outputs: {
        primary: {
          name: "coco_detections",
          type: "task_output",
          task: "detection",
          format: "coco_detections",
        },
        aux: [],
      },
    };
  }

  if (familyName === "deeplabv3") {
    return {
      architecture: {
        family: "deeplabv3",
        framework: "torchvision",
        precision: "fp32",
        backbone: { name: "resnet50", pretrained: true },
        neck: { type: "none" },
        head: { type: "deeplabv3_head", num_classes: numClasses },
      },
      loss: { type: "deeplabv3_default" },
      outputs: {
        primary: {
          name: "coco_segmentation",
          type: "task_output",
          task: "segmentation",
          format: "coco_segmentation",
        },
        aux: [],
      },
    };
  }

  if (familyName === "efficientnet_v2_classifier") {
    return {
      input: {
        input_size: [384, 384],
      },
      architecture: {
        family: "efficientnet_v2_classifier",
        framework: "torchvision",
        precision: "fp32",
        backbone: { name: "efficientnet_v2_s", pretrained: true },
        neck: { type: "none" },
        head: { type: "linear", num_classes: numClasses },
      },
      loss: {
        type: readSourceDatasetLabelMode(config) === "multi_label"
          ? "classification_bce_with_logits"
          : "classification_cross_entropy",
      },
      outputs: {
        primary: {
          name: "classification_logits",
          type: "task_output",
          task: "classification",
          format: "classification_logits",
        },
        aux: [],
      },
    };
  }

  return {
    architecture: {
      family: "resnet_classifier",
      framework: "torchvision",
      precision: "fp32",
      backbone: { name: "resnet18", pretrained: true },
      neck: { type: "none" },
      head: { type: "linear", num_classes: numClasses },
    },
    loss: {
      type: readSourceDatasetLabelMode(config) === "multi_label"
        ? "classification_bce_with_logits"
        : "classification_cross_entropy",
    },
    outputs: {
      primary: {
        name: "classification_logits",
        type: "task_output",
        task: "classification",
        format: "classification_logits",
      },
      aux: [],
    },
  };
}

function setSourceDataset(config, datasetVersionSummary) {
  const nextConfig = cloneModelConfig(config);
  const sourceDataset = isPlainObject(nextConfig.source_dataset) ? nextConfig.source_dataset : {};
  const manifestId = datasetVersionSummary.manifest_id || datasetVersionSummary.id;
  const classOrder = Array.isArray(datasetVersionSummary.class_order) ? datasetVersionSummary.class_order.slice() : [];
  const normalizedTask = normalizeModelTask(datasetVersionSummary.task);
  let classNames = [];
  if (Array.isArray(datasetVersionSummary.class_names)) {
    classNames = datasetVersionSummary.class_names.slice();
  } else if (isPlainObject(datasetVersionSummary.class_names)) {
    classNames = classOrder.map((classId) => {
      const value = datasetVersionSummary.class_names[classId];
      return typeof value === "string" && value ? value : String(classId);
    });
  }
  sourceDataset.manifest_id = manifestId;
  sourceDataset.task = normalizedTask;
  if (datasetVersionSummary.label_mode === "single_label" || datasetVersionSummary.label_mode === "multi_label") {
    sourceDataset.label_mode = datasetVersionSummary.label_mode;
  }
  sourceDataset.num_classes = datasetVersionSummary.num_classes;
  sourceDataset.class_order = classOrder;
  sourceDataset.class_names = classNames;
  nextConfig.source_dataset = sourceDataset;
  const architecture = isPlainObject(nextConfig.architecture) ? nextConfig.architecture : {};
  const head = isPlainObject(architecture.head) ? architecture.head : {};
  head.num_classes = datasetVersionSummary.num_classes;
  architecture.head = head;
  nextConfig.architecture = architecture;
  return nextConfig;
}

function setArchitectureFamily(config, familyName, familiesMetadata) {
  const families = Array.isArray(familiesMetadata?.families) ? familiesMetadata.families : [];
  const family = families.find((f) => f.name === familyName);
  const allowedBackbones = Array.isArray(family?.allowed_backbones) ? family.allowed_backbones : [];

  const currentBackboneName = config?.architecture?.backbone?.name;
  const newBackboneName = allowedBackbones.includes(currentBackboneName)
    ? currentBackboneName
    : allowedBackbones[0] || "resnet50";

  const defaults = buildFamilyDefaults(familyName, config);
  const nextConfig = cloneModelConfig(config);

  if (defaults) {
    const defaultArch = stableClone(defaults.architecture);
    const defaultBackbone = isPlainObject(defaultArch.backbone) ? defaultArch.backbone : {};
    defaultArch.backbone = {
      ...defaultBackbone,
      name: newBackboneName,
    };
    defaultArch.family = familyName;
    nextConfig.architecture = defaultArch;
    if (isPlainObject(defaults.input)) {
      const currentInput = isPlainObject(nextConfig.input) ? nextConfig.input : {};
      nextConfig.input = {
        ...currentInput,
        ...stableClone(defaults.input),
      };
    }
    nextConfig.loss = stableClone(defaults.loss);
    nextConfig.outputs = stableClone(defaults.outputs);
  } else {
    const architecture = isPlainObject(nextConfig.architecture) ? nextConfig.architecture : {};
    architecture.backbone = { name: newBackboneName };
    architecture.family = familyName;
    nextConfig.architecture = architecture;
  }

  return nextConfig;
}

function setBackbone(config, backboneName) {
  const nextConfig = cloneModelConfig(config);
  const architecture = isPlainObject(nextConfig.architecture) ? nextConfig.architecture : {};
  architecture.backbone = isPlainObject(architecture.backbone) ? { ...architecture.backbone, name: backboneName } : { name: backboneName };
  nextConfig.architecture = architecture;
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
  setSourceDataset,
  setArchitectureFamily,
  setBackbone,
};
