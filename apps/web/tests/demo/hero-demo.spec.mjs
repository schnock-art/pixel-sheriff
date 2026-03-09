import { test } from "@playwright/test";

import { recordingVideoDir } from "../../../../scripts/demo/common.mjs";
import {
  DEMO_VIEWPORT,
  bootstrapDemo,
  pause,
  saveHeroVideo,
  selectAsset,
  selectGeometryObject,
  smoothClick,
  waitForBuilderReady,
  waitForDatasetReady,
  waitForLabelingReady,
  waitForModelsReady,
} from "./demoHarness.mjs";

test.setTimeout(300000);

test("records the README hero walkthrough", async ({ browser }) => {
  const demo = await bootstrapDemo();
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
    await selectAsset(page, asset.relativePath);
    await pause(page, 550);
    await selectGeometryObject(page, asset.objectId);
    await pause(page, 700);
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

  await context.close();
  await saveHeroVideo(video);
});
