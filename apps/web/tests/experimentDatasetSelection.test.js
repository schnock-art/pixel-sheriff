const test = require("node:test");
const assert = require("node:assert/strict");

const { buildDatasetVersionOptions } = require("../src/lib/workspace/experimentDatasetSelection.js");

test("buildDatasetVersionOptions maps dataset versions to id/name options", () => {
  const items = [
    { version: { dataset_version_id: "ver-1", name: "Dataset v1" } },
    { version: { dataset_version_id: "ver-2", name: "Dataset v2" } },
  ];

  const result = buildDatasetVersionOptions(items, "ver-1");
  assert.deepEqual(result, [
    { id: "ver-1", name: "Dataset v1" },
    { id: "ver-2", name: "Dataset v2" },
  ]);
});

test("buildDatasetVersionOptions prepends missing configured dataset id", () => {
  const items = [{ version: { dataset_version_id: "ver-1", name: "Dataset v1" } }];

  const result = buildDatasetVersionOptions(items, "deleted-ver");
  assert.deepEqual(result, [
    { id: "deleted-ver", name: "deleted-ver (missing)" },
    { id: "ver-1", name: "Dataset v1" },
  ]);
});
