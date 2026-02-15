const test = require("node:test");
const assert = require("node:assert/strict");

const { buildPageTokens, estimateMaxVisiblePages } = require("../src/lib/workspace/pagination.js");
const { asRelativePath, buildTreeEntries, collectFolderPathsFromRelativePaths, folderChain } = require("../src/lib/workspace/tree.js");

test("collectFolderPathsFromRelativePaths builds sorted unique folder list", () => {
  const folders = collectFolderPathsFromRelativePaths([
    "a/b/file-1.jpg",
    "a/c/file-2.jpg",
    "z/file-3.jpg",
    "a/b/file-4.jpg",
  ]);

  assert.deepEqual(folders, ["a", "a/b", "a/c", "z"]);
});

test("asRelativePath prefers metadata relative path and then original filename", () => {
  assert.equal(asRelativePath({ uri: "/fallback/1", metadata_json: { relative_path: "foo/bar.jpg" } }), "foo/bar.jpg");
  assert.equal(asRelativePath({ uri: "/fallback/2", metadata_json: { original_filename: "orig/name.jpg" } }), "orig/name.jpg");
  assert.equal(asRelativePath({ uri: "/fallback/3", metadata_json: {} }), "/fallback/3");
});

test("buildTreeEntries preserves folder hierarchy and deterministic ordering", () => {
  const tree = buildTreeEntries([
    { id: "3", uri: "/x3", metadata_json: { relative_path: "zeta/c.jpg" } },
    { id: "1", uri: "/x1", metadata_json: { relative_path: "alpha/b.jpg" } },
    { id: "2", uri: "/x2", metadata_json: { relative_path: "alpha/a.jpg" } },
  ]);

  assert.deepEqual(tree.orderedAssetIds, ["2", "1", "3"]);
  assert.deepEqual(tree.folderAssetIds.alpha, ["2", "1"]);
  assert.deepEqual(tree.folderAssetIds.zeta, ["3"]);
});

test("folderChain returns ancestor chain from shallow to deep", () => {
  assert.deepEqual(folderChain("a/b/c"), ["a", "a/b", "a/b/c"]);
  assert.deepEqual(folderChain("single"), ["single"]);
});

test("estimateMaxVisiblePages respects bounds and width", () => {
  assert.equal(estimateMaxVisiblePages(0, 300), 0);
  assert.equal(estimateMaxVisiblePages(5, 0), 5);
  assert.equal(estimateMaxVisiblePages(20, 380), 9);
});

test("buildPageTokens includes first/last pages and ellipsis as needed", () => {
  const tokens = buildPageTokens(20, 10, 9);
  assert.deepEqual(tokens[0], { type: "page", page: 1 });
  assert.deepEqual(tokens[tokens.length - 1], { type: "page", page: 20 });
  assert.equal(tokens.some((token) => token.type === "ellipsis"), true);
});

test("buildPageTokens returns all pages when total fits visibility", () => {
  const tokens = buildPageTokens(5, 2, 9);
  assert.deepEqual(
    tokens,
    [
      { type: "page", page: 1 },
      { type: "page", page: 2 },
      { type: "page", page: 3 },
      { type: "page", page: 4 },
      { type: "page", page: 5 },
    ],
  );
});
