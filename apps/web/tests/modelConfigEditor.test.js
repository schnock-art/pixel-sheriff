const test = require("node:test");
const assert = require("node:assert/strict");

const {
  createEmbeddingAuxOutput,
  isModelConfigDirty,
  setDynamicShapeFlags,
  setEmbeddingAuxEnabled,
  setEmbeddingProjection,
  setSquareInputSize,
} = require("../src/lib/workspace/modelConfigEditor.js");

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
