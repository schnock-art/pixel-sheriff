const test = require("node:test");
const assert = require("node:assert/strict");

const { describeQatSupport, describeQatVariant } = require("../src/lib/workspace/experimentVariants.js");

test("describeQatSupport reports production, experimental, and unsupported states", () => {
  assert.deepEqual(describeQatSupport({
    qat_supported: true,
    qat_mode: "fake_quant",
    qat_experimental: false,
  }), {
    state: "supported",
    message: null,
    mode: "fake_quant",
    experimental: false,
  });

  assert.deepEqual(describeQatSupport({
    qat_supported: true,
    qat_mode: "fake_quant",
    qat_experimental: true,
    qat_warning: "SSDLite QAT is experimental.",
  }), {
    state: "experimental",
    message: "SSDLite QAT is experimental.",
    mode: "fake_quant",
    experimental: true,
  });

  assert.deepEqual(describeQatSupport({
    qat_supported: false,
    qat_reason: "RetinaNet is not supported.",
  }), {
    state: "unsupported",
    message: "RetinaNet is not supported.",
    mode: null,
    experimental: false,
  });
});

test("describeQatVariant surfaces experimental variant warnings", () => {
  assert.equal(describeQatVariant({
    qat: { experimental: true, warning: "SSDLite QAT is experimental." },
  }), "SSDLite QAT is experimental.");

  assert.equal(describeQatVariant({
    qat: { experimental: false, warning: "ignored" },
  }), null);
});
