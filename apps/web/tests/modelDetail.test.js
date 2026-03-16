const test = require("node:test");
const assert = require("node:assert/strict");

const {
  deriveModelDetailState,
  formatFamilyInputSizeHint,
  isAllowedSquareSize,
  mapDatasetVersionRecords,
  normalizeModelTask,
  tasksMatch,
} = require("../src/lib/workspace/modelDetail.js");

const FAMILIES_METADATA = {
  families: [
    {
      name: "resnet_classifier",
      task: "classification",
      allowed_backbones: ["resnet18", "resnet34"],
      input_size: { shape: "square", mode: "range", min_square_size: 32, step: 1, recommended_square_size: 224 },
    },
    {
      name: "ssdlite320_mobilenet_v3_large",
      task: "detection",
      allowed_backbones: ["mobilenet_v3_large"],
      input_size: { shape: "square", mode: "fixed", required_square_size: 320, recommended_square_size: 320 },
    },
  ],
};

test("normalizeModelTask and tasksMatch align legacy task labels", () => {
  assert.equal(normalizeModelTask("bbox"), "detection");
  assert.equal(normalizeModelTask("classification_single"), "classification");
  assert.equal(normalizeModelTask(" segmentation "), "segmentation");
  assert.equal(normalizeModelTask(""), null);
  assert.equal(tasksMatch("bbox", "detection"), true);
  assert.equal(tasksMatch("classification_single", "classification"), true);
  assert.equal(tasksMatch("classification", "segmentation"), false);
});

test("input-size helpers describe fixed and range family rules", () => {
  const rangeRule = FAMILIES_METADATA.families[0].input_size;
  const fixedRule = FAMILIES_METADATA.families[1].input_size;

  assert.equal(isAllowedSquareSize(rangeRule, 224), true);
  assert.equal(isAllowedSquareSize(rangeRule, 0), false);
  assert.equal(isAllowedSquareSize(fixedRule, 320), true);
  assert.equal(isAllowedSquareSize(fixedRule, 640), false);
  assert.equal(
    formatFamilyInputSizeHint(rangeRule),
    "Allowed for this family: any square >= 32. Recommended: 224 x 224.",
  );
  assert.equal(
    formatFamilyInputSizeHint(fixedRule),
    "Required for this family: 320 x 320.",
  );
});

test("mapDatasetVersionRecords reads API envelopes into model-detail options", () => {
  const versions = mapDatasetVersionRecords([
    {
      version: {
        dataset_version_id: "ds-1",
        name: "Classification v1",
        task: "classification_single",
        label_mode: "single_label",
        num_classes: 2,
        class_order: ["cat", "dog"],
        class_names: { cat: "Cat", dog: "Dog" },
      },
    },
    {
      version: {},
    },
  ]);

  assert.equal(versions.length, 1);
  assert.deepEqual(versions[0], {
    id: "ds-1",
    name: "Classification v1",
    task: "classification_single",
    label_mode: "single_label",
    num_classes: 2,
    class_order: ["cat", "dog"],
    class_names: { cat: "Cat", dog: "Dog" },
  });
});

test("deriveModelDetailState computes dataset/family/input/export selections", () => {
  const state = deriveModelDetailState({
    draftConfig: {
      source_dataset: { manifest_id: "ds-2" },
      input: {
        input_size: [320, 320],
        resize_policy: "stretch",
        normalization: { type: "none" },
      },
      architecture: {
        family: "ssdlite320_mobilenet_v3_large",
        backbone: { name: "mobilenet_v3_large", pretrained: true },
      },
      outputs: {
        aux: [{ type: "embedding", name: "embedding", projection: { out_dim: 512, normalize: "none" } }],
      },
      export: {
        onnx: {
          enabled: true,
          opset: 18,
          dynamic_shapes: { batch: true, height_width: false },
        },
      },
    },
    allDatasetVersions: [
      {
        id: "ds-1",
        name: "Cls v1",
        task: "classification",
        label_mode: "single_label",
        num_classes: 2,
        class_order: ["cat", "dog"],
        class_names: { cat: "Cat", dog: "Dog" },
      },
      {
        id: "ds-2",
        name: "Det v2",
        task: "bbox",
        label_mode: "single_label",
        num_classes: 3,
        class_order: ["car", "bus", "bike"],
        class_names: { car: "Car", bus: "Bus", bike: "Bike" },
      },
    ],
    familiesMetadata: FAMILIES_METADATA,
    validationErrors: [{ path: "$.input.input_size", keyword: "familyInputSize", message: "must match family" }],
  });

  assert.equal(state.currentManifestId, "ds-2");
  assert.equal(state.currentTask, "bbox");
  assert.equal(state.currentVersionFromManifest.name, "Det v2");
  assert.equal(state.currentFamilyName, "ssdlite320_mobilenet_v3_large");
  assert.deepEqual(state.allowedBackbones, ["mobilenet_v3_large"]);
  assert.deepEqual(state.allowedPresetSizes, [320]);
  assert.equal(state.customInputAllowed, false);
  assert.equal(state.inputSizePresetValue, "320");
  assert.equal(state.customInputSizeValue, "320");
  assert.equal(state.resizePolicy, "stretch");
  assert.equal(state.normalizationType, "none");
  assert.equal(state.embeddingEnabled, true);
  assert.equal(state.embeddingOutDim, 512);
  assert.equal(state.embeddingNormalize, "none");
  assert.equal(state.onnxEnabled, true);
  assert.equal(state.onnxOpset, 18);
  assert.equal(state.dynamicBatch, true);
  assert.equal(state.dynamicHeightWidth, false);
  assert.equal(state.inputSizeIssue.message, "must match family");
});
