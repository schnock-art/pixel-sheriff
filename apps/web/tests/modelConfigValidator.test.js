const test = require("node:test");
const assert = require("node:assert/strict");

const { validateModelConfigDraft } = require("../src/lib/schema/validator.js");

function buildValidConfig() {
  return {
    schema_version: "1.0",
    name: "demo-model",
    created_at: "2026-02-28T12:00:00Z",
    source_dataset: {
      manifest_id: "manifest-1",
      task: "classification",
      num_classes: 2,
      class_order: ["class-1", "class-2"],
      class_names: ["cat", "dog"],
    },
    input: {
      image_channels: 3,
      input_size: [640, 640],
      resize_policy: "letterbox",
      normalization: {
        type: "imagenet",
      },
    },
    architecture: {
      family: "resnet_classifier",
      framework: "torchvision",
      precision: "fp32",
      backbone: {
        name: "resnet18",
        pretrained: true,
      },
      neck: {
        type: "none",
      },
      head: {
        type: "linear",
        num_classes: 2,
      },
    },
    loss: {
      type: "classification_cross_entropy",
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
    export: {
      onnx: {
        enabled: true,
        opset: 17,
        dynamic_shapes: {
          enabled: true,
          batch: true,
          height_width: false,
        },
        output_names: ["classification_logits"],
      },
    },
  };
}

test("validateModelConfigDraft returns valid=true for schema-compliant config", () => {
  const result = validateModelConfigDraft(buildValidConfig());
  assert.equal(result.isValid, true);
  assert.deepEqual(result.errors, []);
});

test("validateModelConfigDraft returns normalized AJV issues for invalid config", () => {
  const invalid = buildValidConfig();
  invalid.schema_version = "2.0";
  invalid.export.onnx.opset = 1;

  const result = validateModelConfigDraft(invalid);
  assert.equal(result.isValid, false);
  assert.ok(result.errors.length >= 1);
  assert.equal(typeof result.errors[0].path, "string");
  assert.equal(typeof result.errors[0].message, "string");
});

test("validateModelConfigDraft accepts multi-label classification loss type", () => {
  const config = buildValidConfig();
  config.source_dataset.label_mode = "multi_label";
  config.loss.type = "classification_bce_with_logits";

  const result = validateModelConfigDraft(config);
  assert.equal(result.isValid, true);
});
