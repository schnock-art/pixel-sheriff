const INPUT_SIZE_PRESETS = [224, 320, 384, 512, 640];
const RESIZE_POLICY_OPTIONS = ["letterbox", "stretch", "longest_side_pad"];
const NORMALIZATION_OPTIONS = ["imagenet", "none", "custom"];
const EMBEDDING_DIM_OPTIONS = [128, 256, 512];
const EMBEDDING_NORMALIZE_OPTIONS = ["none", "l2"];

function asRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function parseApiErrorMessage(error, fallback) {
  if (error && typeof error === "object") {
    const responseBody = typeof error.responseBody === "string" ? error.responseBody : "";
    if (responseBody) {
      try {
        const parsed = JSON.parse(responseBody);
        const message = parsed?.error?.message;
        if (typeof message === "string" && message.trim()) return message;
      } catch {
        return responseBody;
      }
    }
    if (typeof error.message === "string" && error.message.trim()) return error.message;
  }
  if (error instanceof Error) return error.message;
  return fallback;
}

function normalizeModelTask(task) {
  if (typeof task !== "string") return null;
  const normalized = task.trim().toLowerCase();
  if (!normalized) return null;
  if (normalized === "bbox" || normalized === "detection") return "detection";
  if (normalized === "classification_single" || normalized === "classification") return "classification";
  if (normalized === "segmentation") return "segmentation";
  return normalized;
}

function tasksMatch(left, right) {
  const normalizedLeft = normalizeModelTask(left);
  const normalizedRight = normalizeModelTask(right);
  return normalizedLeft !== null && normalizedLeft === normalizedRight;
}

function getFamilyInputSizeRule(family) {
  if (!family?.input_size || typeof family.input_size !== "object") return null;
  return family.input_size;
}

function isAllowedSquareSize(rule, size) {
  if (!Number.isFinite(size) || size < 1) return false;
  if (!rule || rule.shape !== "square") return true;
  if (rule.mode === "fixed") {
    return size === rule.required_square_size;
  }
  if (rule.mode === "range") {
    const minimum = typeof rule.min_square_size === "number" ? rule.min_square_size : 1;
    const step = typeof rule.step === "number" && rule.step > 0 ? rule.step : 1;
    return size >= minimum && (size - minimum) % step === 0;
  }
  return true;
}

function formatFamilyInputSizeHint(rule) {
  if (!rule || rule.shape !== "square") return null;
  if (rule.mode === "fixed" && typeof rule.required_square_size === "number") {
    return `Required for this family: ${rule.required_square_size} x ${rule.required_square_size}.`;
  }
  if (rule.mode === "range") {
    const minimum = typeof rule.min_square_size === "number" ? rule.min_square_size : 1;
    const step = typeof rule.step === "number" && rule.step > 0 ? rule.step : 1;
    const recommended = typeof rule.recommended_square_size === "number" ? rule.recommended_square_size : null;
    const recommendedText = recommended ? ` Recommended: ${recommended} x ${recommended}.` : "";
    if (step === 1) {
      return `Allowed for this family: any square >= ${minimum}.${recommendedText}`;
    }
    return `Allowed for this family: square sizes >= ${minimum} in steps of ${step}.${recommendedText}`;
  }
  return null;
}

function mapDatasetVersionRecords(items) {
  return (Array.isArray(items) ? items : [])
    .map((envelope) => {
      const version = asRecord(envelope?.version);
      return {
        id: typeof version.dataset_version_id === "string" ? version.dataset_version_id : "",
        name: typeof version.name === "string" ? version.name : "",
        task: typeof version.task === "string" ? version.task : "",
        label_mode:
          typeof version.label_mode === "string" && (version.label_mode === "single_label" || version.label_mode === "multi_label")
            ? version.label_mode
            : null,
        num_classes: typeof version.num_classes === "number" ? version.num_classes : 0,
        class_order: Array.isArray(version.class_order) ? version.class_order : [],
        class_names: version.class_names && typeof version.class_names === "object" ? version.class_names : {},
      };
    })
    .filter((version) => version.id !== "");
}

function deriveModelDetailState(options = {}) {
  const draftConfig = asRecord(options.draftConfig);
  const allDatasetVersions = Array.isArray(options.allDatasetVersions) ? options.allDatasetVersions : [];
  const families = Array.isArray(options.familiesMetadata?.families) ? options.familiesMetadata.families : [];
  const validationErrors = Array.isArray(options.validationErrors) ? options.validationErrors : [];

  const input = asRecord(draftConfig.input);
  const inputSize = Array.isArray(input.input_size) ? input.input_size : [];
  const inputSizeWidth = typeof inputSize[0] === "number" ? Math.floor(inputSize[0]) : 0;
  const inputSizeHeight = typeof inputSize[1] === "number" ? Math.floor(inputSize[1]) : 0;
  const isSquareInputSize = inputSizeWidth > 0 && inputSizeWidth === inputSizeHeight;
  const inputSizePresetValue =
    isSquareInputSize && INPUT_SIZE_PRESETS.includes(inputSizeWidth)
      ? String(inputSizeWidth)
      : "custom";
  const customInputSizeValue = inputSizeWidth > 0 ? String(inputSizeWidth) : "";

  const normalization = asRecord(input.normalization);
  const normalizationType = typeof normalization.type === "string" ? normalization.type : "imagenet";
  const resizePolicy = typeof input.resize_policy === "string" ? input.resize_policy : "letterbox";

  const architecture = asRecord(draftConfig.architecture);
  const backbone = asRecord(architecture.backbone);
  const backboneName = typeof backbone.name === "string" ? backbone.name : "resnet18";
  const pretrained = Boolean(backbone.pretrained);

  const sourceDataset = asRecord(draftConfig.source_dataset);
  const currentFamilyName = typeof architecture.family === "string" ? architecture.family : null;
  const currentManifestId = typeof sourceDataset.manifest_id === "string" ? sourceDataset.manifest_id : null;
  const currentVersionFromManifest = allDatasetVersions.find((version) => version.id === currentManifestId) ?? null;
  const currentFamilyFromMeta = families.find((family) => family.name === currentFamilyName) ?? null;
  const currentTask =
    currentVersionFromManifest?.task
    ?? allDatasetVersions.find((version) => tasksMatch(version.task, currentFamilyFromMeta?.task))?.task
    ?? currentFamilyFromMeta?.task
    ?? null;

  const uniqueTasks = Array.from(new Set(allDatasetVersions.map((version) => version.task))).filter(Boolean);
  const familiesForTask = currentTask
    ? families.filter((family) => tasksMatch(family.task, currentTask))
    : families;
  const versionsForTask = currentTask
    ? allDatasetVersions.filter((version) => tasksMatch(version.task, currentTask))
    : allDatasetVersions;
  const allowedBackbones = Array.isArray(currentFamilyFromMeta?.allowed_backbones) ? currentFamilyFromMeta.allowed_backbones : [];
  const currentFamilyInputSizeRule = getFamilyInputSizeRule(currentFamilyFromMeta);
  const allowedPresetSizes = INPUT_SIZE_PRESETS.filter((value) => isAllowedSquareSize(currentFamilyInputSizeRule, value));
  const customInputAllowed = currentFamilyInputSizeRule?.mode !== "fixed";
  const inputSizeHelpText = formatFamilyInputSizeHint(currentFamilyInputSizeRule);
  const inputSizeIssue = validationErrors.find((issue) => issue?.path === "$.input.input_size" && issue?.keyword === "familyInputSize") ?? null;

  const outputs = asRecord(draftConfig.outputs);
  const auxOutputs = Array.isArray(outputs.aux) ? outputs.aux : [];
  const embeddingAux = auxOutputs.find((item) => {
    const row = asRecord(item);
    return row.type === "embedding" && row.name === "embedding";
  });
  const embeddingAuxRecord = asRecord(embeddingAux);
  const embeddingProjection = asRecord(embeddingAuxRecord.projection);
  const embeddingEnabled = Boolean(embeddingAux);
  const embeddingOutDim = typeof embeddingProjection.out_dim === "number" ? Math.floor(embeddingProjection.out_dim) : 256;
  const embeddingNormalize = embeddingProjection.normalize === "none" ? "none" : "l2";

  const exportSpec = asRecord(draftConfig.export);
  const onnx = asRecord(exportSpec.onnx);
  const dynamicShapes = asRecord(onnx.dynamic_shapes);
  const onnxEnabled = Boolean(onnx.enabled);
  const onnxOpset = typeof onnx.opset === "number" ? Math.floor(onnx.opset) : 17;
  const dynamicBatch = Boolean(dynamicShapes.batch);
  const dynamicHeightWidth = Boolean(dynamicShapes.height_width);

  return {
    input,
    inputSizeWidth,
    inputSizeHeight,
    inputSizePresetValue,
    customInputSizeValue,
    normalizationType,
    resizePolicy,
    architecture,
    backboneName,
    pretrained,
    currentFamilyName,
    currentManifestId,
    currentVersionFromManifest,
    currentFamilyFromMeta,
    currentTask,
    uniqueTasks,
    familiesForTask,
    versionsForTask,
    allowedBackbones,
    currentFamilyInputSizeRule,
    allowedPresetSizes,
    customInputAllowed,
    inputSizeHelpText,
    inputSizeIssue,
    embeddingEnabled,
    embeddingOutDim,
    embeddingNormalize,
    onnxEnabled,
    onnxOpset,
    dynamicBatch,
    dynamicHeightWidth,
  };
}

module.exports = {
  INPUT_SIZE_PRESETS,
  RESIZE_POLICY_OPTIONS,
  NORMALIZATION_OPTIONS,
  EMBEDDING_DIM_OPTIONS,
  EMBEDDING_NORMALIZE_OPTIONS,
  asRecord,
  parseApiErrorMessage,
  normalizeModelTask,
  tasksMatch,
  getFamilyInputSizeRule,
  isAllowedSquareSize,
  formatFamilyInputSizeHint,
  mapDatasetVersionRecords,
  deriveModelDetailState,
};
