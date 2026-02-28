const test = require("node:test");
const assert = require("node:assert/strict");

const { readModelSummary } = require("../src/lib/workspace/modelSummary.js");

test("readModelSummary extracts key model config fields", () => {
  const summary = readModelSummary({
    source_dataset: {
      task: "detection",
      num_classes: 3,
      class_names: ["rock", "lake", "sky"],
    },
    input: {
      input_size: [640, 640],
      resize_policy: "letterbox",
      normalization: { type: "imagenet" },
    },
    architecture: {
      family: "retinanet",
      backbone: { name: "resnet50" },
      neck: { type: "fpn" },
      head: { type: "retinanet" },
    },
    outputs: {
      primary: { format: "coco_detections" },
    },
    export: {
      onnx: {
        enabled: true,
        opset: 17,
        dynamic_shapes: { batch: true, height_width: false },
      },
    },
  });

  assert.equal(summary.task, "detection");
  assert.equal(summary.numClasses, 3);
  assert.equal(summary.classNamesText, "rock, lake, sky");
  assert.equal(summary.inputSizeText, "640 x 640");
  assert.equal(summary.architectureFamily, "retinanet");
  assert.equal(summary.backboneName, "resnet50");
  assert.equal(summary.onnxEnabled, true);
  assert.equal(summary.onnxOpset, 17);
});

test("readModelSummary falls back safely for missing fields", () => {
  const summary = readModelSummary({});
  assert.equal(summary.task, "-");
  assert.equal(summary.numClasses, 0);
  assert.equal(summary.classNamesText, "-");
  assert.equal(summary.inputSizeText, "-");
  assert.equal(summary.onnxEnabled, false);
});

