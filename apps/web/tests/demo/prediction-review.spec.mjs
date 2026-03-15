import fs from "node:fs/promises";
import path from "node:path";

import { expect, test } from "@playwright/test";

import { repoRoot, resolveDemoApiBaseUrl, resolveDemoWebBaseUrl } from "../../../../scripts/demo/common.mjs";
import { attrSelector, bootstrapDemo, waitForImageReady, waitForLabelingReady } from "./demoHarness.mjs";

const apiBaseUrl = resolveDemoApiBaseUrl();
const webBaseUrl = resolveDemoWebBaseUrl();
const browserWebBaseUrl = process.env.DEMO_BROWSER_WEB_BASE_URL ?? webBaseUrl;
const browserApiBaseUrl = process.env.DEMO_BROWSER_API_BASE_URL ?? apiBaseUrl;
const browserRequestApiBaseUrl = process.env.DEMO_BROWSER_API_SOURCE_BASE_URL ?? apiBaseUrl;

function apiUrl(routePath) {
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

async function apiRequest(routePath, init = {}) {
  const response = await fetch(apiUrl(routePath), init);
  if (!response.ok) {
    throw new Error(`API request failed (${response.status}) ${routePath}: ${JSON.stringify(await readResponse(response))}`);
  }
  return readResponse(response);
}

async function uploadAsset(projectId, relativePath, sourceFile) {
  const bytes = await fs.readFile(sourceFile);
  const form = new FormData();
  form.append("file", new Blob([bytes], { type: "image/jpeg" }), path.basename(sourceFile));
  form.append("relative_path", relativePath);
  return apiRequest(`/projects/${projectId}/assets/upload`, { method: "POST", body: form });
}

function buildClassificationPayload(categoryId) {
  return {
    version: "2.0",
    category_id: categoryId,
    category_ids: [categoryId],
    classification: {
      primary_category_id: categoryId,
      category_ids: [categoryId],
    },
  };
}

async function createClassificationReviewProject() {
  const project = await apiRequest("/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: `Prediction Review Classification ${Date.now()}`,
      task_type: "classification_single",
    }),
  });

  const taskId = project.default_task_id;
  const catCategory = await apiRequest(`/projects/${project.id}/categories`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task_id: taskId, name: "Cat", display_order: 0 }),
  });
  const dogCategory = await apiRequest(`/projects/${project.id}/categories`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task_id: taskId, name: "Dog", display_order: 1 }),
  });

  const relativePath = "classification/review-cat.jpg";
  const uploadedAsset = await uploadAsset(
    project.id,
    relativePath,
    path.join(repoRoot, "docs", "demo", "source-images", "cat_01.jpg"),
  );

  await apiRequest(`/projects/${project.id}/annotations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      task_id: taskId,
      asset_id: uploadedAsset.id,
      status: "approved",
      annotated_by: "playwright-prediction-review",
      payload_json: buildClassificationPayload(catCategory.id),
    }),
  });

  return {
    projectId: project.id,
    taskId,
    assetId: uploadedAsset.id,
    relativePath,
    categoryIdsByName: {
      Cat: catCategory.id,
      Dog: dogCategory.id,
    },
    urls: {
      labeling: buildLabelingUrl(project.id, taskId),
    },
  };
}

async function listAnnotations(projectId, taskId) {
  return apiRequest(`/projects/${projectId}/annotations?task_id=${encodeURIComponent(taskId)}`);
}

function buildLabelingUrl(projectId, taskId) {
  return `${browserWebBaseUrl}/projects/${encodeURIComponent(projectId)}/datasets?taskId=${encodeURIComponent(taskId)}`;
}

async function routeBrowserApiToHost(page) {
  if (browserApiBaseUrl === browserRequestApiBaseUrl) return;
  await page.route((url) => url.toString().startsWith(`${browserRequestApiBaseUrl}/`), async (route) => {
    const request = route.request();
    const pageOrigin = new URL(browserWebBaseUrl).origin;
    const corsHeaders = {
      "access-control-allow-origin": pageOrigin,
      "access-control-allow-credentials": "true",
      "access-control-allow-methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
      "access-control-allow-headers":
        request.headerValue("access-control-request-headers") ?? "authorization,content-type",
      vary: "Origin",
    };
    if (request.method() === "OPTIONS") {
      await route.fulfill({
        status: 204,
        headers: corsHeaders,
      });
      return;
    }
    const nextUrl = route.request().url().replace(browserRequestApiBaseUrl, browserApiBaseUrl);
    const response = await route.fetch({ url: nextUrl });
    await route.fulfill({
      response,
      headers: {
        ...response.headers(),
        ...corsHeaders,
      },
    });
  });
}

async function stubDeploymentRoutes(page, { projectId, taskId, task, predictResponse, predictBatchResponse = null }) {
  const deploymentsUrl = `${browserRequestApiBaseUrl}/api/v1/projects/${projectId}/deployments`;
  const predictUrl = `${browserRequestApiBaseUrl}/api/v1/projects/${projectId}/predict`;
  const predictBatchUrl = `${browserRequestApiBaseUrl}/api/v1/projects/${projectId}/predict/batch`;

  await page.route((url) => url.toString() === deploymentsUrl, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        active_deployment_id: "demo-deployment-1",
        items: [
          {
            deployment_id: "demo-deployment-1",
            task_id: taskId,
            name: task === "bbox" ? "Demo Pet Detector" : "Demo Pet Classifier",
            device_preference: "auto",
            status: "available",
            task,
          },
        ],
      }),
    });
  });

  await page.route((url) => url.toString() === predictUrl, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(predictResponse),
    });
  });

  if (predictBatchResponse) {
    await page.route((url) => url.toString() === predictBatchUrl, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(predictBatchResponse),
      });
    });
  }
}

test("reviews deployed bbox predictions in the browser and saves accepted boxes", async ({ page }) => {
  const demo = await bootstrapDemo();
  const heroAsset = demo.assets.find((asset) => asset.relativePath === demo.hero.assetRelativePath);
  if (!heroAsset) throw new Error("Hero asset is missing from demo metadata.");
  const predictedBoxCount = 2;

  await routeBrowserApiToHost(page);
  await stubDeploymentRoutes(page, {
    projectId: demo.projectId,
    taskId: demo.taskId,
    task: "bbox",
    predictResponse: {
      asset_id: heroAsset.id,
      deployment_id: "demo-deployment-1",
      task: "bbox",
      device_selected: "cpu",
      deployment_name: "Demo Pet Detector",
      device_preference: "auto",
      boxes: [
        {
          class_index: 0,
          class_id: demo.categoryIdsByName.Cat,
          class_name: "Cat",
          score: 0.91,
          bbox: [420, 240, 520, 830],
        },
        {
          class_index: 1,
          class_id: demo.categoryIdsByName.Dog,
          class_name: "Dog",
          score: 0.74,
          bbox: [1100, 1020, 380, 540],
        },
      ],
    },
  });

  await page.goto(buildLabelingUrl(demo.projectId, demo.taskId), { waitUntil: "domcontentloaded" });
  await waitForLabelingReady(page);
  await page.locator(attrSelector("folder-tree-asset", "data-demo-path", demo.hero.assetRelativePath)).click();
  await waitForImageReady(page);

  const baselineObjectCount = await page.locator("[data-testid='geometry-object-item']").count();
  expect(baselineObjectCount).toBeGreaterThan(0);
  await page.getByRole("button", { name: "Suggest" }).click();

  await expect(page.locator("[data-testid='pending-deployment-prediction']")).toHaveCount(predictedBoxCount);
  await expect(page.locator("[data-testid='geometry-object-item']")).toHaveCount(baselineObjectCount);
  await expect(page.locator(attrSelector("label-chip", "data-category-id", demo.categoryIdsByName.Cat))).toBeDisabled();

  await page.locator("[data-testid='prediction-review-reject']").click();
  await expect(page.locator("[data-testid='pending-deployment-prediction']")).toHaveCount(0);
  await expect(page.locator("[data-testid='geometry-object-item']")).toHaveCount(baselineObjectCount);

  await page.getByRole("button", { name: "Suggest" }).click();
  await page.locator(`${attrSelector("prediction-review-item", "data-review-item-id", "prediction-bbox-2")}[data-selected="false"]`).click();
  await page.locator(`${attrSelector("prediction-review-item", "data-review-item-id", "prediction-bbox-2")}[data-selected="true"]`).waitFor();
  await page.locator("[data-testid='prediction-review-delete-selected']").click();
  await expect(page.locator("[data-testid='pending-deployment-prediction']")).toHaveCount(1);
  await page.locator(`${attrSelector("prediction-review-item", "data-review-item-id", "prediction-bbox-1")}[data-selected="true"]`).waitFor();
  await page.locator("[data-testid='prediction-review-accept']").click();

  await expect(page.locator("[data-testid='pending-deployment-prediction']")).toHaveCount(0);
  await expect(page.locator("[data-testid='geometry-object-item']")).toHaveCount(1);

  await page.locator("[data-testid='annotation-submit-button']").click();
  await expect(page.locator(".status-toast")).toContainText(/Submitted \d+ staged annotations\./);

  const annotations = await listAnnotations(demo.projectId, demo.taskId);
  const savedAnnotation = annotations.find((annotation) => annotation.asset_id === heroAsset.id);
  expect(savedAnnotation.payload_json.objects).toHaveLength(1);
  expect(savedAnnotation.payload_json.objects[0].provenance.origin_kind).toBe("deployment_prediction");
  expect(savedAnnotation.payload_json.objects[0].provenance.review_decision).toBe("accepted");
});

test("reviews deployed classification predictions in the browser and saves the accepted class", async ({ page }) => {
  const demo = await createClassificationReviewProject();

  await routeBrowserApiToHost(page);
  await stubDeploymentRoutes(page, {
    projectId: demo.projectId,
    taskId: demo.taskId,
    task: "classification",
    predictResponse: {
      asset_id: demo.assetId,
      deployment_id: "demo-deployment-1",
      task: "classification",
      device_selected: "cpu",
      deployment_name: "Demo Pet Classifier",
      device_preference: "auto",
      predictions: [
        {
          class_index: 0,
          class_id: demo.categoryIdsByName.Cat,
          class_name: "Cat",
          score: 0.96,
        },
        {
          class_index: 1,
          class_id: demo.categoryIdsByName.Dog,
          class_name: "Dog",
          score: 0.43,
        },
      ],
    },
  });

  await page.goto(demo.urls.labeling, { waitUntil: "domcontentloaded" });
  await waitForLabelingReady(page);
  await page.locator(attrSelector("folder-tree-asset", "data-demo-path", demo.relativePath)).click();
  await waitForImageReady(page);

  await page.locator(`${attrSelector("label-chip", "data-category-id", demo.categoryIdsByName.Cat)}[data-selected="true"]`).waitFor();
  await page.getByRole("button", { name: "Suggest" }).click();

  await expect(page.locator("[data-testid='prediction-review-item']")).toHaveCount(2);
  await page.locator(`${attrSelector("label-chip", "data-category-id", demo.categoryIdsByName.Cat)}[data-selected="true"]`).waitFor();
  await expect(page.locator(attrSelector("label-chip", "data-category-id", demo.categoryIdsByName.Dog))).toBeDisabled();

  await page.locator(`${attrSelector("prediction-review-item", "data-review-item-id", `prediction-class-${demo.categoryIdsByName.Dog}`)}[data-selected="false"]`).click();
  await page.locator(`${attrSelector("prediction-review-item", "data-review-item-id", `prediction-class-${demo.categoryIdsByName.Dog}`)}[data-selected="true"]`).waitFor();
  await page.locator("[data-testid='prediction-review-accept']").click();

  await page.locator(`${attrSelector("label-chip", "data-category-id", demo.categoryIdsByName.Dog)}[data-selected="true"]`).waitFor();
  await page.locator("[data-testid='annotation-submit-button']").click();
  await expect(page.locator(".status-toast")).toContainText(/Submitted \d+ staged annotations\./);

  const annotations = await listAnnotations(demo.projectId, demo.taskId);
  const savedAnnotation = annotations.find((annotation) => annotation.asset_id === demo.assetId);
  expect(savedAnnotation.payload_json.category_ids).toEqual([demo.categoryIdsByName.Dog]);
  expect(savedAnnotation.payload_json.prediction_review.selected_class_id).toBe(demo.categoryIdsByName.Dog);
  expect(savedAnnotation.payload_json.prediction_review.origin_kind).toBe("deployment_prediction");
});

test("reviews folder batch bbox predictions in the browser and advances across pending images", async ({ page }) => {
  const demo = await bootstrapDemo();
  const catAssets = demo.assets.filter((asset) => asset.relativePath.startsWith("cats/"));
  const [firstCatAsset, secondCatAsset, thirdCatAsset] = catAssets;
  if (!firstCatAsset || !secondCatAsset || !thirdCatAsset) {
    throw new Error("Expected at least three cat assets in the seeded demo project.");
  }

  await routeBrowserApiToHost(page);
  await stubDeploymentRoutes(page, {
    projectId: demo.projectId,
    taskId: demo.taskId,
    task: "bbox",
    predictResponse: {
      asset_id: firstCatAsset.id,
      deployment_id: "demo-deployment-1",
      task: "bbox",
      device_selected: "cpu",
      deployment_name: "Demo Pet Detector",
      device_preference: "auto",
      boxes: [],
    },
    predictBatchResponse: {
      deployment_id: "demo-deployment-1",
      task: "bbox",
      requested_count: catAssets.length,
      completed_count: catAssets.length,
      pending_review_count: 2,
      empty_count: 1,
      error_count: 0,
      deployment_name: "Demo Pet Detector",
      device_preference: "auto",
      predictions: [
        {
          asset_id: firstCatAsset.id,
          deployment_id: "demo-deployment-1",
          task: "bbox",
          device_selected: "cpu",
          deployment_name: "Demo Pet Detector",
          device_preference: "auto",
          boxes: [
            {
              class_index: 0,
              class_id: demo.categoryIdsByName.Cat,
              class_name: "Cat",
              score: 0.92,
              bbox: [390, 280, 980, 1820],
            },
            {
              class_index: 0,
              class_id: demo.categoryIdsByName.Cat,
              class_name: "Cat",
              score: 0.61,
              bbox: [1220, 1320, 310, 470],
            },
          ],
        },
        {
          asset_id: secondCatAsset.id,
          deployment_id: "demo-deployment-1",
          task: "bbox",
          device_selected: "cpu",
          deployment_name: "Demo Pet Detector",
          device_preference: "auto",
          boxes: [
            {
              class_index: 0,
              class_id: demo.categoryIdsByName.Cat,
              class_name: "Cat",
              score: 0.87,
              bbox: [1680, 900, 2040, 1160],
            },
          ],
        },
        {
          asset_id: thirdCatAsset.id,
          deployment_id: "demo-deployment-1",
          task: "bbox",
          device_selected: "cpu",
          deployment_name: "Demo Pet Detector",
          device_preference: "auto",
          boxes: [],
        },
      ],
      errors: [],
    },
  });

  await page.goto(buildLabelingUrl(demo.projectId, demo.taskId), { waitUntil: "domcontentloaded" });
  await waitForLabelingReady(page);
  await page.locator(attrSelector("folder-tree-folder", "data-demo-path", "cats")).click();
  await page.locator(".tree-scope-caption").getByText("cats").waitFor();
  await page.locator("[data-testid='prediction-review-batch-suggest']").click();

  await expect(page.getByText('Batch review: 2 pending, 0 accepted, 0 rejected, 1 empty, 0 failed.')).toBeVisible();
  await expect(page.locator("[data-testid='pending-deployment-prediction']")).toHaveCount(2);

  await page.locator("[data-testid='prediction-review-accept']").click();

  await expect(page.getByText('Batch review: 1 pending, 1 accepted, 0 rejected, 1 empty, 0 failed.')).toBeVisible();
  await expect(page.locator("[data-testid='pending-deployment-prediction']")).toHaveCount(1);
  await expect(page.getByText("Accepted 2 predicted boxes into the draft. Moved to the next pending image.")).toBeVisible();

  await page.locator("[data-testid='prediction-review-reject']").click();

  await expect(page.getByText('Batch review: 0 pending, 1 accepted, 1 rejected, 1 empty, 0 failed.')).toBeVisible();
  await expect(page.locator("[data-testid='pending-deployment-prediction']")).toHaveCount(0);
  await expect(page.getByText("Prediction rejected.")).toBeVisible();
  await expect(page.getByText("Current image review status: rejected.")).toBeVisible();

  await page.locator("[data-testid='annotation-submit-button']").click();
  await expect(page.locator(".status-toast")).toContainText(/Submitted \d+ staged annotations\./);

  const annotations = await listAnnotations(demo.projectId, demo.taskId);
  const acceptedAnnotation = annotations.find((annotation) => annotation.asset_id === firstCatAsset.id);
  const rejectedAnnotation = annotations.find((annotation) => annotation.asset_id === secondCatAsset.id);
  expect(acceptedAnnotation.payload_json.objects).toHaveLength(2);
  expect(acceptedAnnotation.payload_json.objects[0].provenance.origin_kind).toBe("deployment_prediction");
  expect(acceptedAnnotation.payload_json.objects[0].provenance.review_decision).toBe("accepted");
  expect(rejectedAnnotation.payload_json.objects).toHaveLength(1);
  expect(rejectedAnnotation.payload_json.objects[0].id).toBe(secondCatAsset.objectId);
});
