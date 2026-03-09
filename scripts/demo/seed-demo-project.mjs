import fs from "node:fs/promises";
import path from "node:path";

import {
  ensureDemoDirs,
  fixtureDir,
  repoRoot,
  readFixtureManifest,
  resolveDemoApiBaseUrl,
  resolveDemoWebBaseUrl,
  seedMetadataPath,
  writeJson,
} from "./common.mjs";

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

async function apiRequest(apiBaseUrl, routePath, init = {}) {
  const response = await fetch(apiUrl(apiBaseUrl, routePath), init);
  if (!response.ok) {
    const body = await readResponse(response);
    throw new Error(`API request failed (${response.status}) ${routePath}: ${typeof body === "string" ? body : JSON.stringify(body)}`);
  }
  if (response.status === 204) return null;
  return readResponse(response);
}

function resolveFixtureAssetPath(assetFixture) {
  if (typeof assetFixture.source_file === "string" && assetFixture.source_file.trim() !== "") {
    return path.join(repoRoot, assetFixture.source_file);
  }
  return path.join(fixtureDir, assetFixture.file);
}

async function uploadFixtureAsset(apiBaseUrl, projectId, assetFixture) {
  const filePath = resolveFixtureAssetPath(assetFixture);
  const bytes = await fs.readFile(filePath);
  const form = new FormData();
  form.append("file", new Blob([bytes], { type: assetFixture.mime_type ?? "image/png" }), path.basename(filePath));
  form.append("relative_path", assetFixture.relative_path);
  return apiRequest(apiBaseUrl, `/projects/${projectId}/assets/upload`, {
    method: "POST",
    body: form,
  });
}

function buildAnnotationPayload({ categoryId, objectId, bbox, width, height }) {
  return {
    category_id: categoryId,
    category_ids: [categoryId],
    classification: {
      primary_category_id: categoryId,
      category_ids: [categoryId],
    },
    image_basis: {
      width,
      height,
    },
    objects: [
      {
        id: objectId,
        kind: "bbox",
        category_id: categoryId,
        bbox,
      },
    ],
  };
}

export async function seedDemoProject(options = {}) {
  const apiBaseUrl = options.apiBaseUrl ?? resolveDemoApiBaseUrl();
  const webBaseUrl = options.webBaseUrl ?? resolveDemoWebBaseUrl();
  await ensureDemoDirs();

  const manifest = await readFixtureManifest();
  const projectName = options.projectName ?? manifest.project.name;

  const projects = await apiRequest(apiBaseUrl, "/projects");
  for (const project of projects.filter((item) => item.name === projectName)) {
    await apiRequest(apiBaseUrl, `/projects/${project.id}`, { method: "DELETE" });
  }

  const createdProject = await apiRequest(apiBaseUrl, "/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: projectName,
      task_type: manifest.project.task_type,
    }),
  });

  const projectId = createdProject.id;
  const taskId = createdProject.default_task_id;
  const categoryIdsByName = {};

  for (const category of manifest.categories) {
    const createdCategory = await apiRequest(apiBaseUrl, `/projects/${projectId}/categories`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task_id: taskId,
        name: category.name,
        display_order: category.display_order,
      }),
    });
    categoryIdsByName[category.name] = createdCategory.id;
  }

  const uploadedAssets = [];
  for (const assetFixture of manifest.assets) {
    const uploaded = await uploadFixtureAsset(apiBaseUrl, projectId, assetFixture);
    const assetRecord = {
      id: uploaded.id,
      relativePath: assetFixture.relative_path,
      filename: path.basename(assetFixture.relative_path),
      width: assetFixture.width,
      height: assetFixture.height,
      objectId: assetFixture.annotation.object_id,
      categoryName: assetFixture.annotation.category,
      categoryId: categoryIdsByName[assetFixture.annotation.category],
      bbox: assetFixture.annotation.bbox,
    };
    uploadedAssets.push(assetRecord);

    await apiRequest(apiBaseUrl, `/projects/${projectId}/annotations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task_id: taskId,
        asset_id: uploaded.id,
        status: assetFixture.status ?? "approved",
        annotated_by: "docs-demo-pipeline",
        payload_json: buildAnnotationPayload({
          categoryId: categoryIdsByName[assetFixture.annotation.category],
          objectId: assetFixture.annotation.object_id,
          bbox: assetFixture.annotation.bbox,
          width: assetFixture.width,
          height: assetFixture.height,
        }),
      }),
    });
  }

  const createdDataset = await apiRequest(apiBaseUrl, `/projects/${projectId}/datasets/versions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: manifest.dataset.name,
      created_by: "docs-demo-pipeline",
      task_id: taskId,
      selection: {
        mode: "explicit_asset_ids",
        explicit_asset_ids: uploadedAssets.map((asset) => asset.id),
      },
      split: manifest.dataset.split,
      set_active: true,
    }),
  });

  const datasetVersionId = createdDataset.version.dataset_version_id;
  const createdModel = await apiRequest(apiBaseUrl, `/projects/${projectId}/models`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: manifest.model.name,
      dataset_version_id: datasetVersionId,
    }),
  });

  const heroAsset = uploadedAssets.find((asset) => asset.relativePath === manifest.hero.asset_relative_path);
  if (!heroAsset) {
    throw new Error(`Hero asset "${manifest.hero.asset_relative_path}" not found in uploaded demo data`);
  }

  const metadata = {
    apiBaseUrl,
    webBaseUrl,
    projectName,
    projectId,
    taskId,
    datasetVersionId,
    modelId: createdModel.id,
    categoryIdsByName,
    assets: uploadedAssets,
    hero: {
      assetId: heroAsset.id,
      assetRelativePath: heroAsset.relativePath,
      objectId: heroAsset.objectId,
      categoryId: heroAsset.categoryId,
    },
    urls: {
      labeling: `${webBaseUrl}/projects/${encodeURIComponent(projectId)}/datasets?taskId=${encodeURIComponent(taskId)}`,
      dataset: `${webBaseUrl}/projects/${encodeURIComponent(projectId)}/dataset?taskId=${encodeURIComponent(taskId)}`,
      models: `${webBaseUrl}/projects/${encodeURIComponent(projectId)}/models?taskId=${encodeURIComponent(taskId)}`,
      modelBuilder: `${webBaseUrl}/projects/${encodeURIComponent(projectId)}/models/${encodeURIComponent(createdModel.id)}`,
    },
  };

  await writeJson(seedMetadataPath, metadata);
  return metadata;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const metadata = await seedDemoProject();
  console.log(JSON.stringify(metadata, null, 2));
}
