const test = require("node:test");
const assert = require("node:assert/strict");

const {
  buildDescendantsByPath,
  contentUrlForAsset,
  folderCheckState,
  toggleFolderPathSelection,
  toggleStatusSelection,
} = require("../src/lib/workspace/datasetPage.js");

test("toggleStatusSelection keeps include/exclude mutually exclusive", () => {
  const next = toggleStatusSelection({
    selected: ["labeled"],
    otherSelected: ["approved", "needs_review"],
    status: "approved",
    checked: true,
  });

  assert.deepEqual(next.selected.sort(), ["approved", "labeled"]);
  assert.deepEqual(next.otherSelected.sort(), ["needs_review"]);
});

test("toggleFolderPathSelection removes overlapping descendants from opposing set", () => {
  const descendantsByPath = buildDescendantsByPath(["a", "a/b", "a/b/c", "x"]);
  const next = toggleFolderPathSelection({
    selectedPaths: [],
    opposingSelectedPaths: ["a", "a/b/c", "x"],
    path: "a",
    checked: true,
    descendantsByPath,
  });

  assert.deepEqual(next.selectedPaths, ["a", "a/b", "a/b/c"]);
  assert.deepEqual(next.opposingSelectedPaths, ["x"]);
});

test("folderCheckState reports indeterminate for partial subtree selection", () => {
  const descendantsByPath = buildDescendantsByPath(["a", "a/b", "a/c"]);
  const state = folderCheckState("a", ["a/b"], descendantsByPath);
  assert.equal(state, "indeterminate");
});

test("contentUrlForAsset returns stable API route", () => {
  assert.equal(contentUrlForAsset("asset-123"), "/api/v1/assets/asset-123/content");
});
