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

test("validateModelConfigDraft accepts EfficientNetV2 classification configs", () => {
  const config = buildValidConfig();
  config.input.input_size = [384, 384];
  config.architecture.family = "efficientnet_v2_classifier";
  config.architecture.backbone.name = "efficientnet_v2_s";

  const result = validateModelConfigDraft(config);
  assert.equal(result.isValid, true);
});

test("validateModelConfigDraft accepts SSD Lite detection configs", () => {
  const config = buildValidConfig();
  config.source_dataset.task = "detection";
  config.source_dataset.num_classes = 3;
  config.source_dataset.class_order = ["flower", "bee", "bird"];
  config.source_dataset.class_names = ["Flower", "Bee", "Bird"];
  config.input.input_size = [320, 320];
  config.architecture.family = "ssdlite320_mobilenet_v3_large";
  config.architecture.backbone.name = "mobilenet_v3_large";
  config.architecture.neck.type = "none";
  config.architecture.head.type = "ssdlite";
  config.architecture.head.num_classes = 3;
  config.loss.type = "ssdlite_default";
  config.outputs.primary.name = "coco_detections";
  config.outputs.primary.task = "detection";
  config.outputs.primary.format = "coco_detections";
  config.export.onnx.output_names = ["coco_detections"];

  const result = validateModelConfigDraft(config);
  assert.equal(result.isValid, true);
});

test("validateModelConfigDraft rejects SSD Lite configs with non-required image size", () => {
  const config = buildValidConfig();
  config.source_dataset.task = "detection";
  config.source_dataset.num_classes = 3;
  config.source_dataset.class_order = ["flower", "bee", "bird"];
  config.source_dataset.class_names = ["Flower", "Bee", "Bird"];
  config.input.input_size = [224, 224];
  config.architecture.family = "ssdlite320_mobilenet_v3_large";
  config.architecture.backbone.name = "mobilenet_v3_large";
  config.architecture.neck.type = "none";
  config.architecture.head.type = "ssdlite";
  config.architecture.head.num_classes = 3;
  config.loss.type = "ssdlite_default";
  config.outputs.primary.name = "coco_detections";
  config.outputs.primary.task = "detection";
  config.outputs.primary.format = "coco_detections";
  config.export.onnx.output_names = ["coco_detections"];

  const result = validateModelConfigDraft(config);
  assert.equal(result.isValid, false);
  assert.ok(result.errors.some((issue) => issue.path === "$.input.input_size" && /requires input_size \[320, 320\]/.test(issue.message)));
});

test("validateModelConfigDraft rejects detection families outside their square size contract", () => {
  const config = buildValidConfig();
  config.source_dataset.task = "detection";
  config.input.input_size = [250, 250];
  config.architecture.family = "retinanet";
  config.architecture.head.type = "retinanet";
  config.loss.type = "retinanet_default";
  config.outputs.primary.name = "coco_detections";
  config.outputs.primary.task = "detection";
  config.outputs.primary.format = "coco_detections";
  config.export.onnx.output_names = ["coco_detections"];

  const result = validateModelConfigDraft(config);
  assert.equal(result.isValid, false);
  assert.ok(result.errors.some((issue) => issue.path === "$.input.input_size" && /increments of 32 from 224/.test(issue.message)));
});
