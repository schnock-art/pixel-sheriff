import path from "node:path";

import { copyFileEnsured, docsDemoDir, heroRawVideoPath, heroWebmPath, resolveDemoApiBaseUrl } from "../../../../scripts/demo/common.mjs";
import { seedDemoProject } from "../../../../scripts/demo/seed-demo-project.mjs";

export const DEMO_VIEWPORT = { width: 1440, height: 900 };

export function attrSelector(testId, attributeName, value) {
  return `[data-testid="${testId}"][${attributeName}=${JSON.stringify(value)}]`;
}

export async function bootstrapDemo() {
  return seedDemoProject();
}

function apiUrl(apiBaseUrl, routePath) {
  return `${apiBaseUrl}/api/v1${routePath}`;
}

async function readResponse(response) {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export async function demoApiRequest(demo, routePath, init = {}) {
  const apiBaseUrl = demo.apiBaseUrl ?? resolveDemoApiBaseUrl();
  const response = await fetch(apiUrl(apiBaseUrl, routePath), init);
  if (!response.ok) {
    const body = await readResponse(response);
    throw new Error(`Demo API request failed (${response.status}) ${routePath}: ${typeof body === "string" ? body : JSON.stringify(body)}`);
  }
  if (response.status === 204) return null;
  return readResponse(response);
}

function requireExperimentDemo(demo) {
  if (!demo.experimentDemo || typeof demo.experimentDemo !== "object") {
    throw new Error("experimentDemo metadata is missing. Run the Docker demo asset pipeline so experiment seed data is injected.");
  }
  return demo.experimentDemo;
}

export function experimentUrlsForDemo(demo) {
  const experimentDemo = requireExperimentDemo(demo);
  const taskQuery = demo.taskId ? `?taskId=${encodeURIComponent(demo.taskId)}` : "";
  return {
    experiments:
      demo.urls?.experiments ??
      `${demo.webBaseUrl}/projects/${encodeURIComponent(demo.projectId)}/experiments${taskQuery}`,
    experimentDetail:
      demo.urls?.experimentDetail ??
      `${demo.webBaseUrl}/projects/${encodeURIComponent(demo.projectId)}/experiments/${encodeURIComponent(experimentDemo.experimentId)}`,
    deploy:
      demo.urls?.deploy ??
      `${demo.webBaseUrl}/projects/${encodeURIComponent(demo.projectId)}/deploy${taskQuery}`,
  };
}

export function getHeroAsset(demo) {
  const heroAsset = demo.assets.find((asset) => asset.relativePath === demo.hero.assetRelativePath);
  if (!heroAsset) {
    throw new Error(`Hero asset ${demo.hero.assetRelativePath} is missing from seeded demo metadata`);
  }
  return heroAsset;
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

export async function waitForAssetAnnotationReady(page, objectId, categoryId) {
  await page.locator(attrSelector("geometry-object-item", "data-object-id", objectId)).waitFor();
  await page.locator(attrSelector("geometry-object", "data-object-id", objectId)).waitFor();
  await page.locator(attrSelector("label-chip", "data-category-id", categoryId)).waitFor();
}

async function clickCanvasPoint(page, imagePoint, imageSize) {
  const canvas = page.locator("[data-testid='viewer-canvas']");
  const box = await canvas.boundingBox();
  if (!box) {
    throw new Error("Viewer canvas is not visible");
  }

  const scale = Math.min(box.width / imageSize.width, box.height / imageSize.height);
  const renderedWidth = imageSize.width * scale;
  const renderedHeight = imageSize.height * scale;
  const offsetX = box.x + (box.width - renderedWidth) / 2;
  const offsetY = box.y + (box.height - renderedHeight) / 2;
  const clickX = offsetX + imagePoint.x * scale;
  const clickY = offsetY + imagePoint.y * scale;
  await page.mouse.move(clickX, clickY, { steps: 12 });
  await page.mouse.click(clickX, clickY);
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

export async function waitForExperimentsReady(page) {
  await page.locator("[data-testid='experiments-page']").waitFor();
  await page.locator("[data-testid='experiments-table']").waitFor();
  await page.locator("[data-testid='experiment-row']").first().waitFor();
}

export async function waitForExperimentDetailReady(page) {
  await page.locator("[data-testid='experiment-detail-page']").waitFor();
  await page.locator("[data-testid='experiment-card-runtime-logs']").waitFor();
  await page.locator("[data-testid='experiment-card-onnx']").waitFor();
  await page.locator("[data-testid='experiment-deploy-model-button']").waitFor();
}

export async function waitForDeployReady(page) {
  await page.locator("[data-testid='deploy-page']").waitFor();
  await page.locator("[data-testid='deploy-active-model-section']").waitFor();
  await page.locator("[data-testid='deploy-all-deployments-section']").waitFor();
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

export async function selectAsset(page, relativePath, objectId = null, categoryId = null) {
  const assetButton = page.locator(attrSelector("folder-tree-asset", "data-demo-path", relativePath));
  await smoothClick(page, assetButton);
  await waitForImageReady(page);
  if (objectId && categoryId) {
    await waitForAssetAnnotationReady(page, objectId, categoryId);
  }
}

export async function selectGeometryObject(page, asset) {
  const objectButton = page.locator(attrSelector("geometry-object-item", "data-object-id", asset.objectId));
  await objectButton.waitFor();
  await clickCanvasPoint(
    page,
    {
      x: asset.bbox[0] + asset.bbox[2] / 2,
      y: asset.bbox[1] + asset.bbox[3] / 2,
    },
    {
      width: asset.width,
      height: asset.height,
    },
  );
  await page.locator(
    `${attrSelector("geometry-object-item", "data-object-id", asset.objectId)}[data-selected="true"]`,
  ).waitFor();
  const labelChip = page.locator(attrSelector("label-chip", "data-category-id", asset.categoryId));
  await labelChip.waitFor();
  await labelChip.click({ force: true });
  await page.locator(`${attrSelector("label-chip", "data-category-id", asset.categoryId)}[data-selected="true"]`).waitFor();
}

export async function ensureDemoDeployment(page, demo) {
  const urls = experimentUrlsForDemo(demo);
  const listing = await demoApiRequest(demo, `/projects/${demo.projectId}/deployments`);
  const available = Array.isArray(listing?.items) ? listing.items.find((item) => item?.status === "available") ?? null : null;
  if (available) {
    await page.goto(urls.deploy, { waitUntil: "domcontentloaded" });
    await waitForDeployReady(page);
    return available;
  }

  await page.goto(urls.experimentDetail, { waitUntil: "domcontentloaded" });
  await waitForExperimentDetailReady(page);
  await smoothClick(page, page.locator("[data-testid='experiment-deploy-model-button']"));
  await page.waitForURL(/\/deploy(\?|$)/, { timeout: 15000 }).catch(async () => {
    await page.goto(urls.deploy, { waitUntil: "domcontentloaded" });
  });
  await waitForDeployReady(page);
  await page.locator("[data-testid='deployment-row']").first().waitFor({ timeout: 15000 });

  const refreshed = await demoApiRequest(demo, `/projects/${demo.projectId}/deployments`);
  const created = Array.isArray(refreshed?.items) ? refreshed.items.find((item) => item?.status === "available") ?? null : null;
  if (!created) {
    throw new Error("Expected a deployment to exist after clicking Deploy Model");
  }
  return created;
}

export async function stubDemoPredictForHeroAsset(page, demo) {
  const experimentDemo = requireExperimentDemo(demo);
  const heroAsset = getHeroAsset(demo);
  const apiBaseUrl = demo.apiBaseUrl ?? resolveDemoApiBaseUrl();
  const predictUrl = `${apiBaseUrl}/api/v1/projects/${demo.projectId}/predict`;
  const deploymentName = `${experimentDemo.experimentName} run ${experimentDemo.attempt}`;

  await page.route((url) => url.toString() === predictUrl, async (route) => {
    if (route.request().method().toUpperCase() !== "POST") {
      await route.continue();
      return;
    }

    let payload = null;
    try {
      payload = route.request().postDataJSON();
    } catch {
      payload = null;
    }

    if (!payload || payload.asset_id !== heroAsset.id) {
      await route.continue();
      return;
    }

    const deploymentId =
      typeof payload.deployment_id === "string" && payload.deployment_id.trim()
        ? payload.deployment_id.trim()
        : "demo-deployment";
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        asset_id: heroAsset.id,
        deployment_id: deploymentId,
        task: "bbox",
        device_selected: "cpu",
        deployment_name: deploymentName,
        device_preference: "auto",
        boxes: [
          {
            class_index: 0,
            class_id: demo.categoryIdsByName.Cat,
            class_name: "Cat",
            score: 0.94,
            bbox: [420, 240, 520, 830],
          },
          {
            class_index: 1,
            class_id: demo.categoryIdsByName.Dog,
            class_name: "Dog",
            score: 0.73,
            bbox: [1100, 1020, 380, 540],
          },
        ],
      }),
    });
  });
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
