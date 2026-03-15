const test = require("node:test");
const assert = require("node:assert/strict");

const { resolvePrelabelBBox, resolvePrelabelCategoryId } = require("../src/lib/workspace/prelabelGeometry.js");

test("resolvePrelabelBBox prefers reviewed geometry when present", () => {
  assert.deepEqual(
    resolvePrelabelBBox({
      bbox: [10, 12, 30, 40],
      reviewed_bbox: [11, 13, 31, 41],
    }),
    [11, 13, 31, 41],
  );
});

test("resolvePrelabelBBox falls back to original geometry", () => {
  assert.deepEqual(
    resolvePrelabelBBox({
      bbox: [10, 12, 30, 40],
      reviewed_bbox: null,
    }),
    [10, 12, 30, 40],
  );
});

test("resolvePrelabelCategoryId prefers reviewed category when present", () => {
  assert.equal(
    resolvePrelabelCategoryId({
      category_id: "cat-original",
      reviewed_category_id: "cat-reviewed",
    }),
    "cat-reviewed",
  );
});

test("resolvePrelabelCategoryId falls back to original category", () => {
  assert.equal(
    resolvePrelabelCategoryId({
      category_id: "cat-original",
      reviewed_category_id: null,
    }),
    "cat-original",
  );
});
