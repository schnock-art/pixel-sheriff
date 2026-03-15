const AUGMENTATION_PROFILE_LABELS = ["none", "light", "medium", "heavy", "custom"];

function defaultAugmentationProfileForTask(task) {
  const normalized = String(task ?? "classification").trim().toLowerCase();
  if (normalized === "detection" || normalized === "bbox" || normalized === "segmentation") return "none";
  return "light";
}

function createAugmentationStep(type = "horizontal_flip") {
  const normalizedType = String(type ?? "horizontal_flip").trim().toLowerCase();
  if (normalizedType === "color_jitter") {
    return {
      type: "color_jitter",
      p: 1,
      params: { brightness: 0.15, contrast: 0.15, saturation: 0.1, hue: 0.05 },
    };
  }
  if (normalizedType === "rotate") {
    return {
      type: "rotate",
      p: 1,
      params: { degrees: 8 },
    };
  }
  if (normalizedType === "vertical_flip") {
    return {
      type: "vertical_flip",
      p: 0.5,
      params: {},
    };
  }
  return {
    type: "horizontal_flip",
    p: 0.5,
    params: {},
  };
}

function normalizeAugmentationStep(step) {
  if (!step || typeof step !== "object") return createAugmentationStep();
  const fallback = createAugmentationStep(step.type);
  const rawProbability = Number(step.p);
  const probability = Number.isFinite(rawProbability) ? Math.max(0, Math.min(1, rawProbability)) : fallback.p;
  const params = step.params && typeof step.params === "object" ? { ...step.params } : {};
  return {
    type: fallback.type,
    p: probability,
    params,
  };
}

function readAugmentationProfile(config, task) {
  const rawProfile = config && typeof config === "object" ? config.augmentation_profile : null;
  if (typeof rawProfile === "string" && AUGMENTATION_PROFILE_LABELS.includes(rawProfile)) return rawProfile;
  return defaultAugmentationProfileForTask(task);
}

function readAugmentationSteps(config) {
  if (!config || typeof config !== "object" || !Array.isArray(config.augmentation_steps)) return [];
  return config.augmentation_steps.map(normalizeAugmentationStep);
}

function stampAugmentationSpec(config) {
  config.augmentation_spec_version = 1;
  return config;
}

function setAugmentationProfile(config, profile) {
  if (!config || typeof config !== "object") return config;
  config.augmentation_profile = AUGMENTATION_PROFILE_LABELS.includes(profile) ? profile : "none";
  return stampAugmentationSpec(config);
}

function replaceAugmentationSteps(config, steps) {
  if (!config || typeof config !== "object") return config;
  config.augmentation_steps = Array.isArray(steps) ? steps.map(normalizeAugmentationStep) : [];
  return stampAugmentationSpec(config);
}

function addAugmentationStep(config, type = "horizontal_flip") {
  const nextSteps = readAugmentationSteps(config);
  nextSteps.push(createAugmentationStep(type));
  return replaceAugmentationSteps(config, nextSteps);
}

function updateAugmentationStep(config, index, step) {
  const nextSteps = readAugmentationSteps(config);
  if (!Number.isInteger(index) || index < 0 || index >= nextSteps.length) return config;
  nextSteps[index] = normalizeAugmentationStep({ ...nextSteps[index], ...step });
  return replaceAugmentationSteps(config, nextSteps);
}

function removeAugmentationStep(config, index) {
  const nextSteps = readAugmentationSteps(config);
  if (!Number.isInteger(index) || index < 0 || index >= nextSteps.length) return config;
  nextSteps.splice(index, 1);
  return replaceAugmentationSteps(config, nextSteps);
}

function moveAugmentationStep(config, index, direction) {
  const nextSteps = readAugmentationSteps(config);
  if (!Number.isInteger(index) || index < 0 || index >= nextSteps.length) return config;
  const targetIndex = direction === "up" ? index - 1 : index + 1;
  if (targetIndex < 0 || targetIndex >= nextSteps.length) return config;
  const [step] = nextSteps.splice(index, 1);
  nextSteps.splice(targetIndex, 0, step);
  return replaceAugmentationSteps(config, nextSteps);
}

function augmentationCategoryValue(config) {
  const rawValue =
    config && typeof config === "object"
      ? (typeof config.augmentation_mode === "string" ? config.augmentation_mode : config.augmentation)
      : null;
  const normalized = AUGMENTATION_PROFILE_LABELS.includes(rawValue) ? rawValue : "none";
  return AUGMENTATION_PROFILE_LABELS.indexOf(normalized);
}

function augmentationCategoryLabel(value) {
  const index = Math.max(0, Math.min(AUGMENTATION_PROFILE_LABELS.length - 1, Math.round(Number(value) || 0)));
  return AUGMENTATION_PROFILE_LABELS[index];
}

module.exports = {
  AUGMENTATION_PROFILE_LABELS,
  defaultAugmentationProfileForTask,
  createAugmentationStep,
  normalizeAugmentationStep,
  readAugmentationProfile,
  readAugmentationSteps,
  setAugmentationProfile,
  replaceAugmentationSteps,
  addAugmentationStep,
  updateAugmentationStep,
  removeAugmentationStep,
  moveAugmentationStep,
  augmentationCategoryValue,
  augmentationCategoryLabel,
};
