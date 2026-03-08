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
  await pause(page, 600);

  await page.locator("[data-testid='project-ribbon']").hover();
  await pause(page, 300);

  await selectAsset(page, demo.hero.assetRelativePath);
  await pause(page, 500);

  await selectGeometryObject(page, demo.hero.objectId);
  await pause(page, 800);

  await smoothClick(page, page.locator("[data-testid='create-dataset-button']"));
  await waitForDatasetReady(page, demo.datasetVersionId);
  await pause(page, 900);

  await smoothClick(page, page.locator("[data-testid='workflow-tab-models']"));
  await waitForModelsReady(page);
  await pause(page, 800);

  await smoothClick(page, page.locator("[data-testid='model-row']").first().locator("a"));
  await waitForBuilderReady(page);
  await pause(page, 1400);

  await context.close();
  await saveHeroVideo(video);
});
