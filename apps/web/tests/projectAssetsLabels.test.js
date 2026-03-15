const test = require("node:test");
const assert = require("node:assert/strict");

const { formatDeleteLabelErrorMessage } = require("../src/lib/workspace/projectAssetsLabels.js");

test("formatDeleteLabelErrorMessage prefers in-use details and API messages", () => {
  assert.equal(
    formatDeleteLabelErrorMessage("cat", {
      message: "request failed",
      responseBody: JSON.stringify({
        error: {
          code: "category_in_use",
          details: { annotation_references: 2 },
        },
      }),
    }),
    'Cannot delete "cat": 2 annotations still reference this class. Clear those annotations and submit before deleting.',
  );

  assert.equal(
    formatDeleteLabelErrorMessage("cat", {
      message: "request failed",
      responseBody: JSON.stringify({
        error: {
          code: "unknown",
          message: "Custom backend error",
        },
      }),
    }),
    "Failed to delete label: Custom backend error",
  );
});

test("formatDeleteLabelErrorMessage falls back safely for generic failures", () => {
  assert.equal(
    formatDeleteLabelErrorMessage("cat", new Error("Network down")),
    "Failed to delete label: Network down",
  );
  assert.equal(
    formatDeleteLabelErrorMessage("cat", { responseBody: "not-json", message: "Bad response" }),
    "Failed to delete label: Bad response",
  );
});
