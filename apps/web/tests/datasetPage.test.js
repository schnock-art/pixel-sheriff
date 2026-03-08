const test = require("node:test");
const assert = require("node:assert/strict");

const {
  classDisplayName,
  classNamesFromVersion,
  buildDescendantsByPath,
  contentUrlForAsset,
  datasetVersionIdOf,
  fallbackClassName,
  filterPreviewAssets,
  folderCheckState,
  previewAssetPrimaryCategoryId,
  selectedVersionName,
  summaryFromVersion,
  toggleFolderPathSelection,
  toggleStatusSelection,
  resolvePreviewAssetsSection,
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

test("previewAssetPrimaryCategoryId returns the primary category id when present", () => {
  assert.equal(previewAssetPrimaryCategoryId({ label_summary: { primary_category_id: "class-1" } }), "class-1");
  assert.equal(previewAssetPrimaryCategoryId({ label_summary: {} }), null);
});

test("filterPreviewAssets applies split, status, class and search filters", () => {
  const items = [
    {
      asset_id: "asset-1",
      filename: "rose.jpg",
      relative_path: "flowers/rose.jpg",
      status: "labeled",
      split: "train",
      label_summary: { primary_category_id: "flower" },
    },
    {
      asset_id: "asset-2",
      filename: "bee.jpg",
      relative_path: "insects/bee.jpg",
      status: "approved",
      split: "val",
      label_summary: { primary_category_id: "bee" },
    },
    {
      asset_id: "asset-3",
      filename: "tree.jpg",
      relative_path: "plants/tree.jpg",
      status: "approved",
      split: "train",
      label_summary: { primary_category_id: "tree" },
    },
  ];

  const filtered = filterPreviewAssets(items, {
    splitFilter: "train",
    statusFilter: "approved",
    classFilter: "tree",
    searchText: "plants",
  });

  assert.deepEqual(filtered.map((item) => item.asset_id), ["asset-3"]);
});

test("datasetVersionIdOf returns the dataset version id from the envelope", () => {
  assert.equal(datasetVersionIdOf({ version: { dataset_version_id: "version-123" } }), "version-123");
  assert.equal(datasetVersionIdOf({ version: {} }), "");
});

test("summaryFromVersion normalizes counts and warnings from saved version stats", () => {
  const summary = summaryFromVersion({
    version: {
      stats: {
        asset_count: 12,
        class_counts: { rose: 5, bee: 3, ignored: "x" },
        split_counts: { train: 8, val: 2, test: 2 },
        warnings: ["stratify fallback"],
      },
    },
  });

  assert.deepEqual(summary, {
    total: 12,
    class_counts: { rose: 5, bee: 3 },
    split_counts: { train: 8, val: 2, test: 2 },
    warnings: ["stratify fallback"],
  });
});

test("selectedVersionName prefers name then id then fallback", () => {
  assert.equal(selectedVersionName({ version: { name: "Dataset v2", dataset_version_id: "version-2" } }), "Dataset v2");
  assert.equal(selectedVersionName({ version: { dataset_version_id: "version-2" } }), "version-2");
  assert.equal(selectedVersionName({ version: {} }), "(unnamed version)");
});

test("classNamesFromVersion extracts category names from label schema", () => {
  const mapping = classNamesFromVersion({
    version: {
      labels: {
        label_schema: {
          classes: [
            { category_id: "rose", name: "Rose" },
            { category_id: "bee", name: "Bee" },
            { category_id: "", name: "Ignored" },
          ],
        },
      },
    },
  });

  assert.deepEqual(mapping, { rose: "Rose", bee: "Bee" });
});

test("classDisplayName resolves summary, version, fallback and missing class labels", () => {
  assert.equal(
    classDisplayName("rose", {
      summaryClassNames: { rose: "Rose summary" },
      versionClassNames: { rose: "Rose version" },
      categoryNameById: { rose: "Rose category" },
    }),
    "Rose summary",
  );
  assert.equal(
    classDisplayName("bee", {
      summaryClassNames: {},
      versionClassNames: { bee: "Bee version" },
      categoryNameById: {},
    }),
    "Bee version",
  );
  assert.equal(
    classDisplayName("__missing__", {
      summaryClassNames: {},
      versionClassNames: {},
      categoryNameById: {},
    }),
    "Unlabeled / missing primary",
  );
  assert.equal(fallbackClassName("6d62685c-f0ef-4d02-9fb0-2cd6a2ae79a8"), "Deleted category (6d62685c)");
});

// resolvePreviewAssetsSection

test("resolvePreviewAssetsSection returns browse for non-draft mode", () => {
  assert.deepEqual(resolvePreviewAssetsSection("browse", null), { kind: "browse" });
  assert.deepEqual(resolvePreviewAssetsSection("browse", { sample_asset_ids: ["a"] }), { kind: "browse" });
});

test("resolvePreviewAssetsSection returns prompt when draft has no preview yet", () => {
  assert.deepEqual(resolvePreviewAssetsSection("draft", null), { kind: "prompt" });
});

test("resolvePreviewAssetsSection returns empty when preview matched no assets", () => {
  assert.deepEqual(resolvePreviewAssetsSection("draft", { sample_asset_ids: [] }), { kind: "empty" });
});

test("resolvePreviewAssetsSection returns samples with asset IDs after a successful preview", () => {
  const result = resolvePreviewAssetsSection("draft", { sample_asset_ids: ["id-1", "id-2", "id-3"] });
  assert.equal(result.kind, "samples");
  assert.deepEqual(result.assetIds, ["id-1", "id-2", "id-3"]);
});
