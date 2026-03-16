const test = require("node:test");
const assert = require("node:assert/strict");

const {
  asRecord,
  asYesNo,
  cloneConfig,
  configValidation,
  formatCheckpoint,
  formatDateTime,
  formatDurationSeconds,
  formatEtaClock,
  patchNumber,
} = require("../src/lib/workspace/experimentDetail.js");

test("asRecord and cloneConfig provide safe mutable config helpers", () => {
  assert.deepEqual(asRecord(null), {});
  assert.deepEqual(asRecord(["x"]), {});
  assert.deepEqual(asRecord({ optimizer: { lr: 0.001 } }), { optimizer: { lr: 0.001 } });

  const original = { optimizer: { lr: 0.001 } };
  const cloned = cloneConfig(original);
  cloned.optimizer.lr = 0.01;
  assert.equal(original.optimizer.lr, 0.001);
  assert.equal(cloned.optimizer.lr, 0.01);
});

test("configValidation reports invalid numeric training inputs", () => {
  const invalid = configValidation({ optimizer: { lr: 0 }, epochs: 0, batch_size: -1 });
  assert.equal(invalid.isValid, false);
  assert.deepEqual(invalid.issues, [
    "Learning rate must be > 0",
    "Epochs must be >= 1",
    "Batch size must be >= 1",
  ]);

  const valid = configValidation({ optimizer: { lr: 0.001 }, epochs: 5, batch_size: 16 });
  assert.equal(valid.isValid, true);
  assert.deepEqual(valid.issues, []);
});

test("format helpers handle fallback and human-readable output", () => {
  assert.equal(formatDateTime(null), "-");
  assert.equal(formatDateTime("not-a-date"), "-");
  assert.match(formatDateTime("2025-01-02T03:04:05Z", { locales: "en-US", timeZone: "UTC" }), /\d/);

  assert.equal(formatDurationSeconds(-1), "-");
  assert.equal(formatDurationSeconds(59.2), "59s");
  assert.equal(formatDurationSeconds(90), "1m 30s");
  assert.equal(formatDurationSeconds(3661), "1h 1m 1s");

  assert.equal(formatEtaClock(-1), "-");
  assert.equal(
    formatEtaClock(300, { nowMs: Date.UTC(2025, 0, 2, 3, 4, 0), locales: "en-US", timeZone: "UTC" }),
    "03:09 AM",
  );
});

test("checkpoint/number/bool helpers keep UI-ready formatting stable", () => {
  assert.equal(formatCheckpoint(null), "Not available yet");
  assert.equal(
    formatCheckpoint({ epoch: 7, metric_name: "val_accuracy", value: 0.81234 }),
    "epoch 7 | val_accuracy: 0.8123",
  );
  assert.equal(patchNumber(""), null);
  assert.equal(patchNumber("1.25"), 1.25);
  assert.equal(patchNumber("abc"), null);
  assert.equal(asYesNo(true), "yes");
  assert.equal(asYesNo(false), "no");
  assert.equal(asYesNo(null), "-");
});
