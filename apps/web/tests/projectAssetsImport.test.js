const test = require("node:test");
const assert = require("node:assert/strict");

const {
  makeTimestampedAssetName,
  resolveImportRootName,
  createImportProgress,
  setActiveImportFile,
  advanceImportProgress,
  formatImportFailure,
  mergeImportedFolderOptions,
  buildImportResultMessage,
} = require("../src/lib/workspace/projectAssetsImport.js");

test("project assets import helpers derive names and progress state", () => {
  const named = makeTimestampedAssetName("video", new Date("2026-03-15T09:10:11.000Z"));
  assert.equal(named, "video_2026-03-15T09-10-11");

  const rootName = resolveImportRootName([{ webkitRelativePath: "incoming/cat.jpg" }], "Fallback");
  assert.equal(rootName, "incoming");
  assert.equal(resolveImportRootName([], "Fallback"), "Fallback");

  const progress = createImportProgress([{ size: 10 }, { size: 15 }], 123);
  assert.deepEqual(progress, {
    totalFiles: 2,
    completedFiles: 0,
    uploadedFiles: 0,
    failedFiles: 0,
    totalBytes: 25,
    processedBytes: 0,
    startedAtMs: 123,
    activeFileName: null,
  });

  assert.deepEqual(setActiveImportFile(progress, "cat.jpg"), {
    ...progress,
    activeFileName: "cat.jpg",
  });
  assert.equal(setActiveImportFile(null, "cat.jpg"), null);

  assert.deepEqual(advanceImportProgress(progress, 10, "uploaded"), {
    ...progress,
    completedFiles: 1,
    uploadedFiles: 1,
    processedBytes: 10,
    activeFileName: null,
  });
  assert.deepEqual(advanceImportProgress(progress, 15, "failed"), {
    ...progress,
    completedFiles: 1,
    failedFiles: 1,
    processedBytes: 15,
    activeFileName: null,
  });
});

test("project assets import helpers format failures, folder options, and summaries", () => {
  assert.equal(
    formatImportFailure("cat.jpg", { message: "Bad request", responseBody: "{\"error\":\"bad\"}" }),
    'cat.jpg: Bad request ({"error":"bad"})',
  );
  assert.equal(formatImportFailure("cat.jpg", new Error("Timed out")), "cat.jpg: Timed out");

  assert.deepEqual(
    mergeImportedFolderOptions(["train", "train/cats"], ["train/cats/a.jpg", "train/dogs/b.jpg"]),
    ["train", "train/cats", "train/dogs"],
  );

  assert.equal(
    buildImportResultMessage({
      uploadedCount: 0,
      totalFiles: 2,
      targetProjectName: "Vision",
      folderName: "train",
      failuresCount: 2,
    }),
    'Import failed: no files uploaded to "train".',
  );
  assert.equal(
    buildImportResultMessage({
      uploadedCount: 1,
      totalFiles: 2,
      targetProjectName: "Vision",
      folderName: "train",
      failuresCount: 1,
    }),
    'Imported 1/2 images into "Vision/train".',
  );
  assert.equal(
    buildImportResultMessage({
      uploadedCount: 2,
      totalFiles: 2,
      targetProjectName: "Vision",
      folderName: "train",
      failuresCount: 0,
    }),
    'Imported 2 images into "Vision/train".',
  );
});
