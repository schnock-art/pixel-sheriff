const test = require("node:test");
const assert = require("node:assert/strict");

const {
  mergeLogChunk,
  normalizeDeviceLabel,
  runtimeBadgeLabel,
} = require("../src/lib/workspace/experimentRuntime.js");

test("normalizeDeviceLabel and runtimeBadgeLabel format runtime device badges", () => {
  assert.equal(normalizeDeviceLabel("cuda"), "CUDA");
  assert.equal(normalizeDeviceLabel("cpu"), "CPU");
  assert.equal(normalizeDeviceLabel("mps"), "MPS");
  assert.equal(runtimeBadgeLabel({ device_selected: "cuda" }), "CUDA");
  assert.equal(runtimeBadgeLabel({ device_selected: "unknown" }), null);
});

test("mergeLogChunk appends content and advances cursor", () => {
  const first = mergeLogChunk("", { from_byte: 0, to_byte: 12, content: "line-a\n" });
  assert.equal(first.content.includes("line-a"), true);
  assert.equal(first.cursor, 12);

  const second = mergeLogChunk(first.content, { from_byte: 12, to_byte: 24, content: "line-b\n" });
  assert.equal(second.content.includes("line-a"), true);
  assert.equal(second.content.includes("line-b"), true);
  assert.equal(second.cursor, 24);
});

test("mergeLogChunk resets content when server resets from_byte to 0", () => {
  const seeded = mergeLogChunk("old-content", { from_byte: 10, to_byte: 20, content: "new\n" });
  assert.equal(seeded.content.includes("old-content"), true);

  const reset = mergeLogChunk(seeded.content, { from_byte: 0, to_byte: 5, content: "fresh\n" });
  assert.equal(reset.content.includes("old-content"), false);
  assert.equal(reset.content.includes("fresh"), true);
  assert.equal(reset.cursor, 5);
});
