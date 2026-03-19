const test = require("node:test");
const assert = require("node:assert/strict");

const { createPreviewScheduler } = require("../src/lib/workspace/previewScheduler.js");

function createFakeTimers() {
  let now = 0;
  let nextHandleId = 1;
  let timers = [];

  return {
    now: () => now,
    setNow(value) {
      now = value;
    },
    scheduleTimeout(callback, delay) {
      const handle = { id: nextHandleId++, dueAt: now + delay, callback };
      timers.push(handle);
      timers.sort((left, right) => left.dueAt - right.dueAt || left.id - right.id);
      return handle;
    },
    clearScheduledTimeout(handle) {
      timers = timers.filter((item) => item !== handle);
    },
    async flushNextTimer() {
      const next = timers.shift();
      assert.ok(next, "expected a scheduled timer");
      now = next.dueAt;
      next.callback();
      await Promise.resolve();
    },
    nextDueAt() {
      return timers[0]?.dueAt ?? null;
    },
    timerCount() {
      return timers.length;
    },
  };
}

test("preview scheduler coalesces in-flight refresh requests into one trailing run", async () => {
  const fakeTimers = createFakeTimers();
  const pendingResolves = [];
  let runCount = 0;
  const scheduler = createPreviewScheduler({
    getMinIntervalMs: () => 0,
    now: fakeTimers.now,
    scheduleTimeout: fakeTimers.scheduleTimeout,
    clearScheduledTimeout: fakeTimers.clearScheduledTimeout,
    onRun: () =>
      new Promise((resolve) => {
        runCount += 1;
        pendingResolves.push(resolve);
      }),
  });

  scheduler.requestRun({ immediate: true });
  await fakeTimers.flushNextTimer();
  assert.equal(runCount, 1);

  scheduler.requestRun();
  scheduler.requestRun();
  scheduler.requestRun({ immediate: true });
  assert.equal(runCount, 1);
  assert.equal(fakeTimers.timerCount(), 0);

  pendingResolves.shift()();
  await Promise.resolve();
  assert.equal(fakeTimers.timerCount(), 1);

  await fakeTimers.flushNextTimer();
  assert.equal(runCount, 2);

  scheduler.dispose();
});

test("preview scheduler respects min intervals for steady refreshes but allows immediate refreshes to jump ahead", async () => {
  const fakeTimers = createFakeTimers();
  let runCount = 0;
  const scheduler = createPreviewScheduler({
    getMinIntervalMs: () => 300,
    now: fakeTimers.now,
    scheduleTimeout: fakeTimers.scheduleTimeout,
    clearScheduledTimeout: fakeTimers.clearScheduledTimeout,
    onRun: async () => {
      runCount += 1;
    },
  });

  scheduler.requestRun({ immediate: true });
  await fakeTimers.flushNextTimer();
  assert.equal(runCount, 1);

  scheduler.requestRun();
  assert.equal(fakeTimers.nextDueAt(), 300);

  fakeTimers.setNow(100);
  scheduler.requestRun({ immediate: true });
  assert.equal(fakeTimers.nextDueAt(), 100);

  await fakeTimers.flushNextTimer();
  assert.equal(runCount, 2);

  scheduler.dispose();
});
