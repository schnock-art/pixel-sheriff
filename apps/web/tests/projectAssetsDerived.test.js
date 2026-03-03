const test = require("node:test");
const assert = require("node:assert/strict");

const {
  buildVisibleTreeEntries,
  buildAssetReviewStateById,
  buildFolderReviewStatusByPath,
  buildFolderDirtyByPath,
  deriveMessageTone,
} = require("../src/lib/workspace/projectAssetsDerived.js");

test("buildVisibleTreeEntries hides descendants of collapsed folders", () => {
  const treeEntries = [
    { key: "folder:a", kind: "folder", path: "a", depth: 0, name: "a" },
    { key: "folder:a/b", kind: "folder", path: "a/b", depth: 1, name: "b" },
    { key: "file:1", kind: "file", path: "x.jpg", folderPath: "a/b", depth: 2, name: "x.jpg", assetId: "1" },
    { key: "folder:c", kind: "folder", path: "c", depth: 0, name: "c" },
    { key: "file:2", kind: "file", path: "y.jpg", folderPath: "c", depth: 1, name: "y.jpg", assetId: "2" },
  ];

  const visible = buildVisibleTreeEntries(treeEntries, { a: true });
  assert.deepEqual(
    visible.map((entry) => entry.key),
    ["folder:a", "folder:c", "file:2"],
  );
});

test("buildAssetReviewStateById applies pending precedence and dirty flags", () => {
  const orderedAssetRows = [{ id: "1" }, { id: "2" }, { id: "3" }];
  const pendingAnnotations = {
    "1": { status: "unlabeled", objects: [{ id: "obj-1" }] },
    "2": { status: "unlabeled", objects: [] },
  };
  const annotationByAssetId = new Map([
    ["1", { asset_id: "1", status: "unlabeled" }],
    ["2", { asset_id: "2", status: "approved" }],
    ["3", { asset_id: "3", status: "labeled" }],
  ]);

  const result = buildAssetReviewStateById({ orderedAssetRows, pendingAnnotations, annotationByAssetId });

  assert.deepEqual(result.get("1"), { status: "labeled", isDirty: true });
  assert.deepEqual(result.get("2"), { status: "unlabeled", isDirty: true });
  assert.deepEqual(result.get("3"), { status: "labeled", isDirty: false });
});

test("folder review status and dirty aggregation are derived from asset states", () => {
  const assetReviewStateById = new Map([
    ["1", { status: "labeled", isDirty: true }],
    ["2", { status: "unlabeled", isDirty: false }],
    ["3", { status: "labeled", isDirty: false }],
  ]);
  const folderAssetIds = {
    alpha: ["1", "2"],
    beta: ["3"],
    empty: [],
  };

  const status = buildFolderReviewStatusByPath({ folderAssetIds, assetReviewStateById });
  const dirty = buildFolderDirtyByPath({ folderAssetIds, assetReviewStateById });

  assert.deepEqual(status, {
    alpha: "has_unlabeled",
    beta: "all_labeled",
    empty: "empty",
  });
  assert.deepEqual(dirty, {
    alpha: true,
    beta: false,
    empty: false,
  });
});

test("deriveMessageTone classifies errors and success", () => {
  assert.equal(deriveMessageTone(null), "info");
  assert.equal(deriveMessageTone("Export ready."), "success");
  assert.equal(deriveMessageTone("Import failed: bad request"), "error");
  assert.equal(deriveMessageTone("An ERROR occurred"), "error");
});
