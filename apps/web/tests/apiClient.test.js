const test = require("node:test");
const assert = require("node:assert/strict");

const { ApiError, apiPost, requestJson, resolveAssetUri } = require("../src/lib/api/client.js");
const {
  buildDatasetVersionAssetsPath,
  buildDatasetVersionsPath,
  buildExperimentEventsUrl,
  buildExperimentLogsPath,
} = require("../src/lib/api/paths.js");

test("requestJson wraps network failures in ApiError", async () => {
  const originalFetch = global.fetch;
  global.fetch = async () => {
    throw new Error("socket hang up");
  };

  await assert.rejects(
    () => requestJson("/projects", { method: "GET" }),
    (error) =>
      error instanceof ApiError &&
      error.method === "GET" &&
      error.url.endsWith("/api/v1/projects") &&
      error.responseBody === "socket hang up",
  );

  global.fetch = originalFetch;
});

test("apiPost sends JSON body and throws ApiError for non-ok responses", async () => {
  const originalFetch = global.fetch;
  let capturedInit = null;
  global.fetch = async (_url, init) => {
    capturedInit = init;
    return {
      ok: false,
      status: 422,
      text: async () => '{"error":{"message":"bad input"}}',
    };
  };

  await assert.rejects(
    () => apiPost("/projects", { name: "demo" }),
    (error) => error instanceof ApiError && error.status === 422 && error.responseBody.includes("bad input"),
  );
  assert.equal(capturedInit.method, "POST");
  assert.equal(capturedInit.headers["Content-Type"], "application/json");
  assert.equal(capturedInit.body, JSON.stringify({ name: "demo" }));

  global.fetch = originalFetch;
});

test("resolveAssetUri preserves absolute urls and prefixes relative api paths", () => {
  assert.equal(resolveAssetUri("https://example.com/image.png"), "https://example.com/image.png");
  assert.equal(resolveAssetUri("/api/v1/assets/a/content"), "/api/v1/assets/a/content");
  assert.equal(resolveAssetUri("exports/hash.zip"), "/exports/hash.zip");
});

test("buildDatasetVersionsPath includes task query only when provided", () => {
  assert.equal(buildDatasetVersionsPath("project-1"), "/projects/project-1/datasets/versions");
  assert.equal(buildDatasetVersionsPath("project-1", "task-7"), "/projects/project-1/datasets/versions?task_id=task-7");
});

test("buildDatasetVersionAssetsPath normalizes paging and filter params", () => {
  assert.equal(
    buildDatasetVersionAssetsPath("project-1", "version-2", {
      page: 2.9,
      page_size: 50,
      split: "train",
      status: "approved",
      class_id: "class-1",
      search: "leaf",
    }),
    "/projects/project-1/datasets/versions/version-2/assets?page=2&page_size=50&split=train&status=approved&class_id=class-1&search=leaf",
  );
});

test("buildExperimentLogsPath and event url keep query parameters stable", () => {
  assert.equal(
    buildExperimentLogsPath("project-1", "exp-2", { fromByte: 12.8, maxBytes: 4096 }),
    "/projects/project-1/experiments/exp-2/logs?from_byte=12&max_bytes=4096",
  );
  assert.equal(
    buildExperimentEventsUrl("http://localhost:8010/", "project 1", "exp/2", { fromLine: 5.7, attempt: 3 }),
    "http://localhost:8010/api/v1/projects/project%201/experiments/exp%2F2/events?from_line=5&attempt=3",
  );
});
