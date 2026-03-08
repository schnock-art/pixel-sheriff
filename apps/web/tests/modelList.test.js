const test = require("node:test");
const assert = require("node:assert/strict");

const { deriveModelDatasetVersion, deriveModelStatus } = require("../src/lib/workspace/modelList.js");

test("deriveModelDatasetVersion reads manifest id and resolves version name", () => {
  const result = deriveModelDatasetVersion(
    {
      source_dataset: {
        manifest_id: "dataset-v2",
      },
    },
    { "dataset-v2": "Dataset v2" },
  );

  assert.deepEqual(result, {
    datasetVersionId: "dataset-v2",
    datasetVersionName: "Dataset v2",
    hasSourceDataset: true,
  });
});

test("deriveModelStatus maps experiment state and source dataset fallback", () => {
  const experiments = [
    {
      id: "exp-1",
      model_id: "model-a",
      status: "completed",
      created_at: "2026-03-01T10:00:00Z",
      updated_at: "2026-03-01T12:00:00Z",
    },
    {
      id: "exp-2",
      model_id: "model-b",
      status: "running",
      created_at: "2026-03-02T10:00:00Z",
      updated_at: "2026-03-02T12:00:00Z",
    },
    {
      id: "exp-3",
      model_id: "model-c",
      status: "failed",
      created_at: "2026-03-03T10:00:00Z",
      updated_at: "2026-03-03T12:00:00Z",
    },
  ];

  assert.equal(deriveModelStatus(experiments, "model-a", true), "completed");
  assert.equal(deriveModelStatus(experiments, "model-b", true), "training");
  assert.equal(deriveModelStatus(experiments, "model-c", true), "failed");
  assert.equal(deriveModelStatus(experiments, "model-d", true), "ready");
  assert.equal(deriveModelStatus(experiments, "model-e", false), "draft");
});
