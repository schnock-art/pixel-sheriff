import { test } from "@playwright/test";

import {
  bootstrapDemo,
  saveViewportScreenshot,
  selectAsset,
  selectGeometryObject,
  waitForBuilderReady,
  waitForDatasetReady,
  waitForLabelingReady,
  waitForModelsReady,
} from "./demoHarness.mjs";

test("captures deterministic README screenshots", async ({ page }) => {
  const demo = await bootstrapDemo();

  await page.goto(demo.urls.labeling, { waitUntil: "domcontentloaded" });
  await waitForLabelingReady(page);
  await saveViewportScreenshot(page, "screenshot-01-assets.png");

  const heroAsset = demo.assets.find((asset) => asset.objectId === demo.hero.objectId);
  if (!heroAsset) {
    throw new Error(`Hero asset ${demo.hero.objectId} is missing from seeded demo metadata`);
  }
  await selectAsset(page, demo.hero.assetRelativePath, demo.hero.objectId, demo.hero.categoryId);
  await selectGeometryObject(page, heroAsset);
  await saveViewportScreenshot(page, "screenshot-02-labeling.png");

  await page.goto(demo.urls.dataset, { waitUntil: "domcontentloaded" });
  await waitForDatasetReady(page, demo.datasetVersionId);
  await saveViewportScreenshot(page, "screenshot-03-dataset.png");

  await page.goto(demo.urls.models, { waitUntil: "domcontentloaded" });
  await waitForModelsReady(page);
  await saveViewportScreenshot(page, "screenshot-04-models.png");

  await page.goto(demo.urls.modelBuilder, { waitUntil: "domcontentloaded" });
  await waitForBuilderReady(page);
  await saveViewportScreenshot(page, "screenshot-05-builder.png");
});
