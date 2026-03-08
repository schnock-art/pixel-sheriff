import path from "node:path";

import { copyFileEnsured, docsDemoDir, heroRawVideoPath, heroWebmPath } from "../../../../scripts/demo/common.mjs";
import { seedDemoProject } from "../../../../scripts/demo/seed-demo-project.mjs";

export const DEMO_VIEWPORT = { width: 1440, height: 900 };

export function attrSelector(testId, attributeName, value) {
  return `[data-testid="${testId}"][${attributeName}=${JSON.stringify(value)}]`;
}

export async function bootstrapDemo() {
  return seedDemoProject();
}

export async function pause(page, milliseconds = 300) {
  await page.waitForTimeout(milliseconds);
}

export async function waitForImageReady(page) {
  await page.locator("[data-testid='viewer-image']").waitFor();
  await page.waitForFunction(() => {
    const image = document.querySelector("[data-testid='viewer-image']");
    return Boolean(image && image.complete);
  });
}

export async function waitForLabelingReady(page) {
  await page.locator("[data-testid='project-ribbon']").waitFor();
  await page.locator("[data-testid='asset-browser']").waitFor();
  await page.locator("[data-testid='label-panel']").waitFor();
  await page.locator("[data-testid='folder-tree-asset']").first().waitFor();
  await waitForImageReady(page);
}

export async function waitForDatasetReady(page, datasetVersionId) {
  await page.locator("[data-testid='dataset-summary-panel']").waitFor();
  await page.locator(attrSelector("dataset-version-item", "data-dataset-version-id", datasetVersionId)).waitFor();
}

export async function waitForModelsReady(page) {
  await page.locator("[data-testid='models-table']").waitFor();
}

export async function waitForBuilderReady(page) {
  await page.locator("[data-testid='model-builder-grid']").waitFor();
  await page.locator("[data-testid='model-step-dataset']").waitFor();
}

export async function moveMouseToCenter(page, locator, steps = 18) {
  const box = await locator.boundingBox();
  if (!box) return;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps });
}

export async function smoothClick(page, locator, pauseBeforeClick = 140) {
  await locator.scrollIntoViewIfNeeded();
  await moveMouseToCenter(page, locator);
  await pause(page, pauseBeforeClick);
  await locator.click();
}

export async function selectAsset(page, relativePath) {
  const assetButton = page.locator(attrSelector("folder-tree-asset", "data-demo-path", relativePath));
  await smoothClick(page, assetButton);
  await waitForImageReady(page);
}

export async function selectGeometryObject(page, objectId) {
  const objectButton = page.locator(attrSelector("geometry-object-item", "data-object-id", objectId));
  await objectButton.scrollIntoViewIfNeeded();
  await moveMouseToCenter(page, objectButton);
  await pause(page, 120);
  await objectButton.click({ force: true });
  await page.locator(attrSelector("geometry-object", "data-object-id", objectId)).waitFor();
}

export async function saveViewportScreenshot(page, fileName) {
  const screenshotPath = path.join(docsDemoDir, fileName);
  await page.screenshot({
    path: screenshotPath,
    animations: "disabled",
  });
  return screenshotPath;
}

export async function saveHeroVideo(video) {
  if (!video) {
    throw new Error("Playwright did not produce a video artifact for the hero demo");
  }
  const recordedVideoPath = await video.path();
  await copyFileEnsured(recordedVideoPath, heroRawVideoPath);
  await copyFileEnsured(recordedVideoPath, heroWebmPath);
  return {
    raw: heroRawVideoPath,
    webm: heroWebmPath,
  };
}
