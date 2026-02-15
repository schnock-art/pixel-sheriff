const test = require("node:test");
const assert = require("node:assert/strict");

const { parseLabelShortcutDigit, resolveWorkspaceHotkeyAction } = require("../src/lib/workspace/hotkeys.js");

test("parseLabelShortcutDigit resolves top-row and numpad digits", () => {
  assert.equal(parseLabelShortcutDigit({ key: "1", code: "Digit1" }), 1);
  assert.equal(parseLabelShortcutDigit({ key: "2", code: "Numpad2" }), 2);
  assert.equal(parseLabelShortcutDigit({ key: "0", code: "Digit0" }), null);
});

test("resolveWorkspaceHotkeyAction handles navigation and label shortcuts", () => {
  assert.deepEqual(
    resolveWorkspaceHotkeyAction({ key: "ArrowLeft", code: "ArrowLeft" }, { activeLabelCount: 3 }),
    { type: "navigate_prev" },
  );
  assert.deepEqual(
    resolveWorkspaceHotkeyAction({ key: "ArrowRight", code: "ArrowRight" }, { activeLabelCount: 3 }),
    { type: "navigate_next" },
  );
  assert.deepEqual(
    resolveWorkspaceHotkeyAction({ key: "1", code: "Digit1" }, { activeLabelCount: 3 }),
    { type: "toggle_label", labelIndex: 0 },
  );
  assert.deepEqual(
    resolveWorkspaceHotkeyAction({ key: "2", code: "Numpad2" }, { activeLabelCount: 3 }),
    { type: "toggle_label", labelIndex: 1 },
  );
});

test("resolveWorkspaceHotkeyAction ignores blocked contexts and out-of-range digits", () => {
  assert.equal(
    resolveWorkspaceHotkeyAction(
      { key: "1", code: "Digit1", ctrlKey: true, target: { tagName: "DIV", isContentEditable: false } },
      { activeLabelCount: 3 },
    ),
    null,
  );
  assert.equal(
    resolveWorkspaceHotkeyAction(
      { key: "1", code: "Digit1", target: { tagName: "INPUT", isContentEditable: false } },
      { activeLabelCount: 3 },
    ),
    null,
  );
  assert.equal(
    resolveWorkspaceHotkeyAction(
      { key: "9", code: "Digit9", target: { tagName: "DIV", isContentEditable: false } },
      { activeLabelCount: 2 },
    ),
    null,
  );
});
