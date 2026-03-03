const test = require("node:test");
const assert = require("node:assert/strict");

const {
  onnxStatusLabel,
  onnxInputShapeText,
  onnxClassNamesText,
  onnxValidationText,
} = require("../src/lib/workspace/experimentOnnx.js");

test("onnxStatusLabel reflects exported, failed, and pending states", () => {
  assert.equal(onnxStatusLabel({ status: "exported" }, "completed"), "Exported");
  assert.equal(onnxStatusLabel({ status: "failed" }, "completed"), "Failed");
  assert.equal(onnxStatusLabel(null, "completed"), "Pending");
  assert.equal(onnxStatusLabel(null, "running"), "Pending");
});

test("onnxInputShapeText and onnxClassNamesText render safe summaries", () => {
  assert.equal(onnxInputShapeText({ input_shape: [3, 224, 224] }), "3 x 224 x 224");
  assert.equal(onnxInputShapeText({ input_shape: [] }), "-");

  assert.equal(onnxClassNamesText({ class_names: ["cat", "dog"] }), "cat, dog");
  assert.equal(
    onnxClassNamesText({ class_names: ["a", "b", "c", "d"] }, { maxNames: 2 }),
    "a, b (+2)",
  );
});

test("onnxValidationText renders validation badge text", () => {
  assert.equal(onnxValidationText({ validation: { status: "passed" } }), "Validated with ONNX Runtime");
  assert.equal(onnxValidationText({ validation: { status: "failed" } }), "ONNX Runtime validation failed");
  assert.equal(onnxValidationText({ validation: {} }), "-");
});
