const test = require("node:test");
const assert = require("node:assert/strict");

const {
  createEmbeddingAuxOutput,
  isModelConfigDirty,
  setDynamicShapeFlags,
  setEmbeddingAuxEnabled,
  setEmbeddingProjection,
  setSquareInputSize,
  setSourceDataset,
  setArchitectureFamily,
  setBackbone,
} = require("../src/lib/workspace/modelConfigEditor.js");

const FAMILIES_METADATA = {
  schema_version: "1",
  families: [
    {
      name: "deeplabv3",
      task: "segmentation",
      allowed_backbones: ["resnet50", "resnet101"],
    },
    {
      name: "resnet_classifier",
      task: "classification",
      allowed_backbones: ["resnet18", "resnet34", "resnet50", "resnet101", "mobilenet_v3_large", "mobilenet_v3_small"],
    },
    {
      name: "retinanet",
      task: "detection",
      allowed_backbones: ["resnet50", "resnet101"],
    },
  ],
};

test("isModelConfigDirty compares deeply with stable object key ordering", () => {
  const saved = {
    b: 1,
    a: {
      y: true,
      x: [3, 2, 1],
    },
  };
  const draft = {
    a: {
      x: [3, 2, 1],
      y: true,
    },
    b: 1,
  };
  assert.equal(isModelConfigDirty(saved, draft), false);
});

test("setSquareInputSize updates width and height to the same value", () => {
  const next = setSquareInputSize({ input: { input_size: [640, 640] } }, 512);
  assert.deepEqual(next.input.input_size, [512, 512]);
});

test("setEmbeddingAuxEnabled toggles the fixed embedding auxiliary output", () => {
  const enabled = setEmbeddingAuxEnabled({ outputs: { aux: [] } }, true);
  assert.equal(enabled.outputs.aux.length, 1);
  assert.deepEqual(enabled.outputs.aux[0], createEmbeddingAuxOutput());

  const disabled = setEmbeddingAuxEnabled(enabled, false);
  assert.deepEqual(disabled.outputs.aux, []);
});

test("setEmbeddingProjection updates embedding output dimension and normalization", () => {
  const base = setEmbeddingAuxEnabled({ outputs: { aux: [] } }, true);
  const next = setEmbeddingProjection(base, 512, "none");
  assert.equal(next.outputs.aux[0].projection.out_dim, 512);
  assert.equal(next.outputs.aux[0].projection.normalize, "none");
});

// --- setSourceDataset ---

test("setSourceDataset patches manifest_id, num_classes, class_order, class_names, and head.num_classes", () => {
  const config = {
    source_dataset: { manifest_id: "old-id", num_classes: 1 },
    architecture: { backbone: { name: "resnet50" }, head: { num_classes: 1, dropout: 0.5 } },
  };
  const summary = {
    id: "ds-123",
    manifest_id: "manifest-456",
    num_classes: 5,
    class_order: ["cat", "dog", "bird", "fish", "rabbit"],
    class_names: { cat: "Cat", dog: "Dog", bird: "Bird", fish: "Fish", rabbit: "Rabbit" },
  };
  const next = setSourceDataset(config, summary);
  assert.equal(next.source_dataset.manifest_id, "manifest-456");
  assert.equal(next.source_dataset.task, "classification");
  assert.equal(next.source_dataset.num_classes, 5);
  assert.deepEqual(next.source_dataset.class_order, summary.class_order);
  assert.deepEqual(next.source_dataset.class_names, ["Cat", "Dog", "Bird", "Fish", "Rabbit"]);
  assert.equal(next.architecture.head.num_classes, 5);
});

test("setSourceDataset normalizes bbox task names and preserves label mode", () => {
  const next = setSourceDataset(
    { source_dataset: {}, architecture: { head: { num_classes: 1 } } },
    {
      id: "ds-bbox",
      task: "bbox",
      label_mode: "multi_label",
      num_classes: 2,
      class_order: ["flower", "bee"],
      class_names: { flower: "Flower", bee: "Bee" },
    },
  );
  assert.equal(next.source_dataset.task, "detection");
  assert.equal(next.source_dataset.label_mode, "multi_label");
  assert.deepEqual(next.source_dataset.class_names, ["Flower", "Bee"]);
});

test("setSourceDataset falls back to id when manifest_id is absent", () => {
  const config = { source_dataset: {}, architecture: { head: { num_classes: 1 } } };
  const summary = { id: "ds-999", num_classes: 3, class_order: [], class_names: {} };
  const next = setSourceDataset(config, summary);
  assert.equal(next.source_dataset.manifest_id, "ds-999");
});

test("setSourceDataset does not mutate the input config", () => {
  const config = {
    source_dataset: { manifest_id: "orig", num_classes: 2 },
    architecture: { head: { num_classes: 2 } },
  };
  const frozen = JSON.parse(JSON.stringify(config));
  setSourceDataset(config, { id: "new", num_classes: 10, class_order: [], class_names: {} });
  assert.deepEqual(config, frozen);
});

// --- setArchitectureFamily ---

test("setArchitectureFamily regenerates architecture, head, loss, and outputs for the new family", () => {
  const config = {
    source_dataset: { manifest_id: "ds-1", task: "detection", num_classes: 3 },
    architecture: { backbone: { name: "resnet50" }, head: { num_classes: 3 } },
    loss: { type: "classification_cross_entropy" },
    outputs: { primary: { name: "old", type: "task_output", task: "classification", format: "classification_logits" }, aux: [] },
  };
  const next = setArchitectureFamily(config, "retinanet", FAMILIES_METADATA);
  assert.equal(next.loss.type, "retinanet_default");
  assert.equal(next.outputs.primary.task, "detection");
  assert.equal(next.outputs.primary.format, "coco_detections");
  assert.equal(next.architecture.head.type, "retinanet");
  assert.equal(next.architecture.head.num_classes, 3);
  assert.equal(next.architecture.framework, "torchvision");
  assert.equal(next.architecture.family, "retinanet");
});

test("setArchitectureFamily uses BCE loss for multi-label classification datasets", () => {
  const next = setArchitectureFamily(
    {
      source_dataset: { task: "classification", label_mode: "multi_label", num_classes: 4 },
      architecture: { backbone: { name: "resnet18" }, head: { num_classes: 4 } },
    },
    "resnet_classifier",
    FAMILIES_METADATA,
  );
  assert.equal(next.loss.type, "classification_bce_with_logits");
  assert.equal(next.architecture.head.num_classes, 4);
  assert.equal(next.outputs.primary.format, "classification_logits");
});

test("setArchitectureFamily keeps existing backbone when it is in allowed_backbones", () => {
  const config = {
    architecture: { backbone: { name: "resnet101" }, head: { num_classes: 2 } },
  };
  const next = setArchitectureFamily(config, "retinanet", FAMILIES_METADATA);
  assert.equal(next.architecture.backbone.name, "resnet101");
  assert.equal(next.architecture.family, "retinanet");
});

test("setArchitectureFamily resets backbone to first allowed when current backbone is not in allowed_backbones", () => {
  const config = {
    architecture: { backbone: { name: "mobilenet_v3_large" }, head: { num_classes: 2 } },
  };
  const next = setArchitectureFamily(config, "retinanet", FAMILIES_METADATA);
  assert.equal(next.architecture.backbone.name, "resnet50");
  assert.equal(next.architecture.family, "retinanet");
});

test("setArchitectureFamily does not mutate the input config", () => {
  const config = {
    architecture: { backbone: { name: "resnet50" }, head: { num_classes: 2 } },
    loss: { type: "classification_cross_entropy" },
    outputs: { primary: { name: "classification_logits", type: "task_output", task: "classification", format: "classification_logits" }, aux: [] },
  };
  const frozen = JSON.parse(JSON.stringify(config));
  setArchitectureFamily(config, "retinanet", FAMILIES_METADATA);
  assert.deepEqual(config, frozen);
});

// --- setBackbone ---

test("setBackbone patches backbone.name and preserves other backbone fields", () => {
  const config = {
    architecture: { backbone: { name: "resnet50", pretrained: true }, head: { num_classes: 5 } },
  };
  const next = setBackbone(config, "resnet101");
  assert.equal(next.architecture.backbone.name, "resnet101");
  assert.equal(next.architecture.backbone.pretrained, true);
  assert.equal(next.architecture.head.num_classes, 5);
});

test("setBackbone does not mutate the input config", () => {
  const config = {
    architecture: { backbone: { name: "resnet50" }, head: { num_classes: 2 } },
  };
  const frozen = JSON.parse(JSON.stringify(config));
  setBackbone(config, "resnet18");
  assert.deepEqual(config, frozen);
});

test("setDynamicShapeFlags derives dynamic_shapes.enabled from batch/height_width", () => {
  const base = {
    export: {
      onnx: {
        dynamic_shapes: {
          enabled: false,
          batch: false,
          height_width: false,
        },
      },
    },
  };
  const next = setDynamicShapeFlags(base, true, false);
  assert.equal(next.export.onnx.dynamic_shapes.batch, true);
  assert.equal(next.export.onnx.dynamic_shapes.height_width, false);
  assert.equal(next.export.onnx.dynamic_shapes.enabled, true);
});
