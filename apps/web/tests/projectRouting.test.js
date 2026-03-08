const test = require("node:test");
const assert = require("node:assert/strict");

const {
  deriveProjectSectionFromPathname,
  buildProjectSectionHref,
  buildModelBuilderHref,
  buildModelCreateHref,
  normalizeSection,
} = require("../src/lib/workspace/projectRouting.js");

test("deriveProjectSectionFromPathname resolves route section", () => {
  assert.equal(deriveProjectSectionFromPathname("/projects/abc/datasets"), "datasets");
  assert.equal(deriveProjectSectionFromPathname("/projects/abc/dataset"), "dataset");
  assert.equal(deriveProjectSectionFromPathname("/projects/abc/models"), "models");
  assert.equal(deriveProjectSectionFromPathname("/projects/abc/experiments"), "experiments");
  assert.equal(deriveProjectSectionFromPathname("/projects/abc/deploy"), "deploy");
});

test("deriveProjectSectionFromPathname falls back from detail routes and unknown paths", () => {
  assert.equal(deriveProjectSectionFromPathname("/projects/abc/models/new"), "models");
  assert.equal(deriveProjectSectionFromPathname("/projects/abc/models/model-1"), "models");
  assert.equal(deriveProjectSectionFromPathname("/projects/abc/experiments/exp-1"), "experiments");
  assert.equal(deriveProjectSectionFromPathname("/"), "datasets");
});

test("buildProjectSectionHref and builder href create project scoped links", () => {
  assert.equal(buildProjectSectionHref("abc", "datasets"), "/projects/abc/datasets");
  assert.equal(buildProjectSectionHref("abc", "dataset"), "/projects/abc/dataset");
  assert.equal(buildProjectSectionHref("abc", "models"), "/projects/abc/models");
  assert.equal(buildProjectSectionHref("abc", "deploy"), "/projects/abc/deploy");
  assert.equal(buildProjectSectionHref("abc", "unknown"), "/projects/abc/datasets");
  assert.equal(buildProjectSectionHref("", "models"), "/projects");

  assert.equal(buildModelBuilderHref("abc", null), "/projects/abc/models/new");
  assert.equal(buildModelBuilderHref("abc", "retinanet_v1"), "/projects/abc/models/retinanet_v1");
  assert.equal(buildModelCreateHref("abc"), "/projects/abc/models/new");
  assert.equal(buildModelCreateHref("abc", { taskId: "task-1" }), "/projects/abc/models/new?taskId=task-1");
  assert.equal(
    buildModelCreateHref("abc", { taskId: "task-1", datasetVersionId: "dataset-v2" }),
    "/projects/abc/models/new?taskId=task-1&datasetVersionId=dataset-v2",
  );
});

test("normalizeSection accepts only supported sections", () => {
  assert.equal(normalizeSection("datasets"), "datasets");
  assert.equal(normalizeSection("dataset"), "dataset");
  assert.equal(normalizeSection("models"), "models");
  assert.equal(normalizeSection("experiments"), "experiments");
  assert.equal(normalizeSection("deploy"), "deploy");
});

