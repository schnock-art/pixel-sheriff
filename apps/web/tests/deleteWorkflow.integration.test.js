const test = require("node:test");
const assert = require("node:assert/strict");

const { buildTreeEntries } = require("../src/lib/workspace/tree.js");
const {
  clearSelectedDeleteAssets,
  pruneCollapsedFoldersForDeletedPath,
  pruneSelectedDeleteAssets,
  selectScopeDeleteAssets,
  shouldResetSelectedFolderAfterDeletion,
  toggleSelectedDeleteAsset,
} = require("../src/lib/workspace/deleteState.js");

test("multi-delete selection flow supports scope-select, toggle, and post-delete cleanup", () => {
  const scopeRows = [{ id: "a" }, { id: "b" }, { id: "c" }];

  let selected = selectScopeDeleteAssets(scopeRows);
  assert.deepEqual(selected, { a: true, b: true, c: true });

  selected = toggleSelectedDeleteAsset(selected, "b");
  assert.deepEqual(selected, { a: true, c: true });

  selected = clearSelectedDeleteAssets(selected, ["a"]);
  assert.deepEqual(selected, { c: true });

  const pruned = pruneSelectedDeleteAssets({ c: true, missing: true }, (assetId) => assetId === "c");
  assert.deepEqual(pruned, { c: true });
});

test("folder/subfolder delete flow resolves subtree IDs, scope reset, and collapse pruning", () => {
  const tree = buildTreeEntries([
    { id: "a", uri: "/a", metadata_json: { relative_path: "root/sub/a.jpg" } },
    { id: "b", uri: "/b", metadata_json: { relative_path: "root/sub/nested/b.jpg" } },
    { id: "c", uri: "/c", metadata_json: { relative_path: "root/other/c.jpg" } },
  ]);

  const subtreeIds = (tree.folderAssetIds["root/sub"] ?? []).slice().sort();
  assert.deepEqual(subtreeIds, ["a", "b"]);

  assert.equal(shouldResetSelectedFolderAfterDeletion("root/sub/nested", "root/sub"), true);
  assert.equal(shouldResetSelectedFolderAfterDeletion("root/other", "root/sub"), false);

  const collapsed = {
    root: true,
    "root/sub": true,
    "root/sub/nested": false,
    "root/other": true,
  };
  assert.deepEqual(pruneCollapsedFoldersForDeletedPath(collapsed, "root/sub"), { root: true, "root/other": true });
});
