const test = require("node:test");
const assert = require("node:assert/strict");

const { getClassColor, normalizedHueForClassId } = require("../src/lib/workspace/classColors.js");

test("normalizedHueForClassId is deterministic and bounded", () => {
  const first = normalizedHueForClassId(17);
  const second = normalizedHueForClassId(17);
  const negative = normalizedHueForClassId(-17);
  const fallback = normalizedHueForClassId(Number.NaN);

  assert.equal(first, second);
  assert.ok(first >= 0 && first < 360);
  assert.ok(negative >= 0 && negative < 360);
  assert.equal(fallback, 210);
});

test("getClassColor returns deterministic css-ready tokens for class id", () => {
  const a = getClassColor(5);
  const b = getClassColor(5);

  assert.deepEqual(a, b);
  assert.match(a.chipBackground, /^hsl\(\d+ 88% 95%\)$/);
  assert.match(a.chipBorder, /^hsl\(\d+ 68% 58%\)$/);
  assert.match(a.chipText, /^hsl\(\d+ 46% 28%\)$/);
  assert.match(a.chipActiveBackground, /^hsl\(\d+ 92% 88%\)$/);
  assert.match(a.overlayStroke, /^hsl\(\d+ 70% 48%\)$/);
  assert.match(a.overlayFill, /^hsl\(\d+ 85% 55% \/ 0\.22\)$/);
});
