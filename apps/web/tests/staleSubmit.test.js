const test = require("node:test");
const assert = require("node:assert/strict");

const {
  isAnnotationSubmitNotFoundError,
  prunePendingAnnotationsForKnownAssets,
} = require("../src/lib/workspace/staleSubmit.js");

test("isAnnotationSubmitNotFoundError detects stale annotation submit 404", () => {
  const projectId = "project-1";
  const staleError = {
    status: 404,
    url: `http://localhost:8010/api/v1/projects/${projectId}/annotations`,
  };
  const other404 = { status: 404, url: "http://localhost:8010/api/v1/projects/project-1/assets" };
  const otherStatus = { status: 422, url: `http://localhost:8010/api/v1/projects/${projectId}/annotations` };

  assert.equal(isAnnotationSubmitNotFoundError(staleError, projectId), true);
  assert.equal(isAnnotationSubmitNotFoundError(other404, projectId), false);
  assert.equal(isAnnotationSubmitNotFoundError(otherStatus, projectId), false);
});

test("prunePendingAnnotationsForKnownAssets removes stale pending asset entries", () => {
  const pendingAnnotations = {
    "asset-1": { labelIds: [1], status: "labeled", objects: [], imageBasis: null },
    "asset-2": { labelIds: [2], status: "labeled", objects: [], imageBasis: null },
    "asset-3": { labelIds: [], status: "unlabeled", objects: [], imageBasis: null },
  };
  const knownAssetIds = ["asset-1", "asset-3"];

  const pruned = prunePendingAnnotationsForKnownAssets(pendingAnnotations, knownAssetIds);
  assert.deepEqual(Object.keys(pruned.nextPendingAnnotations).sort(), ["asset-1", "asset-3"]);
  assert.deepEqual(pruned.removedAssetIds, ["asset-2"]);
});

