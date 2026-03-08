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

  await selectAsset(page, demo.hero.assetRelativePath);
  await selectGeometryObject(page, demo.hero.objectId);
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

