const test = require("node:test");
const assert = require("node:assert/strict");

const { shouldAllowNavigation } = require("../src/lib/workspace/navigationGuard.js");

test("shouldAllowNavigation allows when no unsaved drafts", () => {
  assert.equal(
    shouldAllowNavigation({
      hasUnsavedDrafts: false,
      confirmDiscard: () => {
        throw new Error("should not prompt");
      },
    }),
    true,
  );
});

test("shouldAllowNavigation prompts and allows on confirm", () => {
  let called = 0;
  const allowed = shouldAllowNavigation({
    hasUnsavedDrafts: true,
    confirmDiscard: () => {
      called += 1;
      return true;
    },
  });
  assert.equal(allowed, true);
  assert.equal(called, 1);
});

test("shouldAllowNavigation blocks on cancel or missing confirm", () => {
  assert.equal(
    shouldAllowNavigation({
      hasUnsavedDrafts: true,
      confirmDiscard: () => false,
    }),
    false,
  );
  assert.equal(shouldAllowNavigation({ hasUnsavedDrafts: true }), false);
});

