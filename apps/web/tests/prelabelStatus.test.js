const test = require("node:test");
const assert = require("node:assert/strict");

const { derivePrelabelSessionStatus } = require("../src/lib/workspace/prelabelStatus.js");

test("derivePrelabelSessionStatus distinguishes failed sessions from empty completed runs", () => {
  const failed = derivePrelabelSessionStatus({
    source_label: "Florence-2",
    source_type: "florence2",
    status: "failed",
    generated_proposals: 0,
    skipped_unmatched: 0,
    error_message: "trainer timeout",
  });
  const empty = derivePrelabelSessionStatus({
    source_label: "Florence-2",
    source_type: "florence2",
    status: "completed",
    generated_proposals: 0,
    skipped_unmatched: 0,
    error_message: null,
  });

  assert.deepEqual(failed, {
    badgeTone: "failed",
    badgeLabel: "Failed",
    description: "Florence-2 • processing failed",
    emptyStateMessage: "trainer timeout",
  });
  assert.deepEqual(empty, {
    badgeTone: "empty",
    badgeLabel: "No Detections",
    description: "Florence-2 • completed without proposals",
    emptyStateMessage: "AI prelabels completed, but no proposals were generated on sampled frames.",
  });
});

test("derivePrelabelSessionStatus surfaces unmatched detections separately", () => {
  const result = derivePrelabelSessionStatus({
    source_label: "Florence-2",
    source_type: "florence2",
    status: "completed",
    generated_proposals: 0,
    skipped_unmatched: 3,
    error_message: null,
  });

  assert.deepEqual(result, {
    badgeTone: "empty",
    badgeLabel: "No Matching Labels",
    description: "Florence-2 • detections did not match task classes",
    emptyStateMessage: "AI prelabels completed, but detections did not match your task classes.",
  });
});

test("derivePrelabelSessionStatus marks successful proposal runs as review-ready", () => {
  const result = derivePrelabelSessionStatus({
    source_label: "Florence-2",
    source_type: "florence2",
    device_preference: "auto",
    status: "completed",
    generated_proposals: 4,
    skipped_unmatched: 1,
    error_message: null,
  });

  assert.deepEqual(result, {
    badgeTone: "completed",
    badgeLabel: "Ready To Review",
    description: "Florence-2 • pref auto • proposals are ready for review",
    emptyStateMessage: "No pending proposals on this frame.",
  });
});
