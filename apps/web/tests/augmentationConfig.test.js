const test = require("node:test");
const assert = require("node:assert/strict");

const {
  addAugmentationStep,
  augmentationCategoryLabel,
  augmentationCategoryValue,
  defaultAugmentationProfileForTask,
  moveAugmentationStep,
  normalizeRotateParams,
  readAugmentationSteps,
  removeAugmentationStep,
  setAugmentationProfile,
} = require("../src/lib/workspace/augmentationConfig.js");

test("defaultAugmentationProfileForTask is task-aware", () => {
  assert.equal(defaultAugmentationProfileForTask("classification"), "light");
  assert.equal(defaultAugmentationProfileForTask("detection"), "none");
  assert.equal(defaultAugmentationProfileForTask("segmentation"), "none");
});

test("augmentation config editing preserves steps when profile changes", () => {
  const config = { task: "classification" };
  setAugmentationProfile(config, "custom");
  addAugmentationStep(config, "horizontal_flip");
  addAugmentationStep(config, "rotate");
  assert.equal(config.augmentation_spec_version, 1);
  assert.equal(readAugmentationSteps(config).length, 2);

  setAugmentationProfile(config, "light");
  assert.equal(config.augmentation_profile, "light");
  assert.equal(readAugmentationSteps(config).length, 2);

  moveAugmentationStep(config, 1, "up");
  assert.equal(readAugmentationSteps(config)[0].type, "rotate");

  removeAugmentationStep(config, 1);
  assert.equal(readAugmentationSteps(config).length, 1);
});

test("augmentation analytics labels include custom bucket", () => {
  assert.equal(augmentationCategoryValue({ augmentation_mode: "custom" }), 4);
  assert.equal(augmentationCategoryLabel(4), "custom");
});

test("rotate augmentation normalizes legacy and canonical degree ranges", () => {
  assert.deepEqual(normalizeRotateParams({ degrees: 8 }), { min_degrees: -8, max_degrees: 8 });
  assert.deepEqual(normalizeRotateParams({ min_degrees: 5, max_degrees: -3 }), { min_degrees: -3, max_degrees: 5 });

  const steps = readAugmentationSteps({
    augmentation_steps: [
      { type: "rotate", p: 1, params: { degrees: 6 } },
      { type: "rotate", p: 1, params: { min_degrees: -4, max_degrees: 9 } },
    ],
  });
  assert.deepEqual(steps[0].params, { min_degrees: -6, max_degrees: 6 });
  assert.deepEqual(steps[1].params, { min_degrees: -4, max_degrees: 9 });
});
