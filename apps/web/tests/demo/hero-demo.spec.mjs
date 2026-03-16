import { test } from "@playwright/test";

import { recordingVideoDir } from "../../../../scripts/demo/common.mjs";
import {
  DEMO_VIEWPORT,
  bootstrapDemo,
  ensureDemoDeployment,
  experimentUrlsForDemo,
  getHeroAsset,
  pause,
  saveHeroVideo,
  selectAsset,
  selectGeometryObject,
  smoothClick,
  waitForBuilderReady,
  waitForDatasetReady,
  waitForDeployReady,
  waitForExperimentDetailReady,
  waitForExperimentsReady,
  waitForLabelingReady,
  waitForModelsReady,
  stubDemoPredictForHeroAsset,
} from "./demoHarness.mjs";

test.setTimeout(300000);

test("records the README hero walkthrough", async ({ browser }) => {
  const demo = await bootstrapDemo();
  const experimentUrls = experimentUrlsForDemo(demo);
  const heroAsset = getHeroAsset(demo);
  const context = await browser.newContext({
    baseURL: demo.webBaseUrl,
    viewport: DEMO_VIEWPORT,
    recordVideo: {
      dir: recordingVideoDir,
      size: DEMO_VIEWPORT,
    },
  });
  const page = await context.newPage();
  const video = page.video();

  await page.goto(demo.urls.labeling, { waitUntil: "domcontentloaded" });
  await waitForLabelingReady(page);
  await pause(page, 1100);

  await page.locator("[data-testid='project-ribbon']").hover();
  await pause(page, 650);

  for (const asset of demo.assets) {
    await selectAsset(page, asset.relativePath, asset.objectId, asset.categoryId);
    await pause(page, 550);
    await selectGeometryObject(page, asset);
    await pause(page, 950);
  }

  await smoothClick(page, page.locator("[data-testid='create-dataset-button']"));
  await page.waitForURL(/\/dataset(\?|$)/, { timeout: 15000 }).catch(async () => {
    await page.goto(demo.urls.dataset, { waitUntil: "domcontentloaded" });
  });
  await waitForDatasetReady(page, demo.datasetVersionId);
  await pause(page, 1400);

  await smoothClick(page, page.locator("[data-testid='workflow-tab-models']"));
  await page.waitForURL(/\/models(\?|$)/, { timeout: 15000 }).catch(async () => {
    await page.goto(demo.urls.models, { waitUntil: "domcontentloaded" });
  });
  await waitForModelsReady(page);
  await pause(page, 1300);

  await smoothClick(page, page.locator("[data-testid='model-row']").first().locator("a"));
  await page.waitForURL(/\/models\/[^/?#]+/, { timeout: 15000 }).catch(async () => {
    await page.goto(demo.urls.modelBuilder, { waitUntil: "domcontentloaded" });
  });
  await waitForBuilderReady(page);
  await pause(page, 1800);

  await smoothClick(page, page.locator("[data-testid='model-train-button']"));
  await page.locator("[data-testid='train-flow-modal']").waitFor();
  await pause(page, 500);
  await smoothClick(page, page.locator("[data-testid='model-train-continue-button']"));
  await page.waitForURL(/\/experiments\/[^/?#]+/, { timeout: 15000 }).catch(async () => {
    await page.goto(experimentUrls.experimentDetail, { waitUntil: "domcontentloaded" });
  });
  await waitForExperimentDetailReady(page);
  await pause(page, 1500);

  await smoothClick(page, page.getByRole("link", { name: "Back to Experiments" }));
  await page.waitForURL(/\/experiments(\?|$)/, { timeout: 15000 }).catch(async () => {
    await page.goto(experimentUrls.experiments, { waitUntil: "domcontentloaded" });
  });
  await waitForExperimentsReady(page);
  await pause(page, 1300);

  await smoothClick(page, page.locator("[data-testid='experiment-row']").first().locator("a"));
  await page.waitForURL(/\/experiments\/[^/?#]+/, { timeout: 15000 }).catch(async () => {
    await page.goto(experimentUrls.experimentDetail, { waitUntil: "domcontentloaded" });
  });
  await waitForExperimentDetailReady(page);
  await pause(page, 1600);

  await ensureDemoDeployment(page, demo);
  await waitForDeployReady(page);
  await pause(page, 1300);

  await stubDemoPredictForHeroAsset(page, demo);
  await page.goto(demo.urls.labeling, { waitUntil: "domcontentloaded" });
  await waitForLabelingReady(page);
  await selectAsset(page, demo.hero.assetRelativePath, heroAsset.objectId, heroAsset.categoryId);
  await pause(page, 500);
  await smoothClick(page, page.getByRole("button", { name: "Suggest" }));
  await page.locator("[data-testid='pending-deployment-prediction']").first().waitFor();
  await pause(page, 1600);
  await smoothClick(page, page.locator("[data-testid='prediction-review-accept']"));
  await page.locator("[data-testid='pending-deployment-prediction']").first().waitFor({ state: "detached" });
  await pause(page, 1400);

  await context.close();
  await saveHeroVideo(video);
});
