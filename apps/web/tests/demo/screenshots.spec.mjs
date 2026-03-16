import { test } from "@playwright/test";

import {
  bootstrapDemo,
  ensureDemoDeployment,
  experimentUrlsForDemo,
  getHeroAsset,
  saveViewportScreenshot,
  selectAsset,
  selectGeometryObject,
  waitForBuilderReady,
  waitForDatasetReady,
  waitForDeployReady,
  waitForExperimentDetailReady,
  waitForExperimentsReady,
  waitForLabelingReady,
  waitForModelsReady,
  stubDemoPredictForHeroAsset,
} from "./demoHarness.mjs";

test("captures deterministic README screenshots", async ({ page }) => {
  const demo = await bootstrapDemo();
  const experimentUrls = experimentUrlsForDemo(demo);

  await page.goto(demo.urls.labeling, { waitUntil: "domcontentloaded" });
  await waitForLabelingReady(page);
  await saveViewportScreenshot(page, "screenshot-01-assets.png");

  const heroAsset = getHeroAsset(demo);
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

  await page.goto(experimentUrls.experiments, { waitUntil: "domcontentloaded" });
  await waitForExperimentsReady(page);
  await saveViewportScreenshot(page, "screenshot-06-experiments.png");

  await page.goto(experimentUrls.experimentDetail, { waitUntil: "domcontentloaded" });
  await waitForExperimentDetailReady(page);
  await saveViewportScreenshot(page, "screenshot-07-experiment-run.png");

  await ensureDemoDeployment(page, demo);
  await waitForDeployReady(page);
  await saveViewportScreenshot(page, "screenshot-08-deploy.png");

  await stubDemoPredictForHeroAsset(page, demo);
  await page.goto(demo.urls.labeling, { waitUntil: "domcontentloaded" });
  await waitForLabelingReady(page);
  await selectAsset(page, demo.hero.assetRelativePath, demo.hero.objectId, demo.hero.categoryId);
  await page.getByRole("button", { name: "Suggest" }).click();
  await page.locator("[data-testid='pending-deployment-prediction']").first().waitFor();
  await saveViewportScreenshot(page, "screenshot-09-mal.png");
});
