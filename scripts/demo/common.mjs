import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const moduleDir = path.dirname(fileURLToPath(import.meta.url));

export const repoRoot = path.resolve(moduleDir, "..", "..");
export const appsWebDir = path.join(repoRoot, "apps", "web");
export const docsDemoDir = path.join(repoRoot, "docs", "demo");
export const artifactsDemoDir = path.join(repoRoot, "artifacts", "demo");
export const rawVideoDir = path.join(artifactsDemoDir, "raw");
export const recordingVideoDir = path.join(artifactsDemoDir, "recordings");
export const metadataDir = path.join(artifactsDemoDir, "metadata");
export const demoCacheDir = path.join("/tmp", "pixel-sheriff-demo");
export const playwrightCacheDir = path.join(demoCacheDir, "playwright-transform-cache");
export const fixtureDir = path.join(repoRoot, "scripts", "demo", "fixtures");
export const fixtureManifestPath = path.join(fixtureDir, "manifest.json");
export const heroRawVideoPath = path.join(rawVideoDir, "hero-demo.webm");
export const heroWebmPath = path.join(docsDemoDir, "hero-demo.webm");
export const heroMp4Path = path.join(docsDemoDir, "hero-demo.mp4");
export const heroGifPath = path.join(docsDemoDir, "hero-demo.gif");
export const seedMetadataPath = path.join(metadataDir, "seed-demo-project.json");
export const screenshotPaths = [
  path.join(docsDemoDir, "screenshot-01-assets.png"),
  path.join(docsDemoDir, "screenshot-02-labeling.png"),
  path.join(docsDemoDir, "screenshot-03-dataset.png"),
  path.join(docsDemoDir, "screenshot-04-models.png"),
  path.join(docsDemoDir, "screenshot-05-builder.png"),
];

export function resolveDemoApiBaseUrl() {
  return process.env.DEMO_API_BASE_URL ?? "http://localhost:8010";
}

export function resolveDemoWebBaseUrl() {
  return process.env.DEMO_WEB_BASE_URL ?? "http://localhost:3010";
}

export async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
  return dirPath;
}

export async function ensureDemoDirs() {
  await Promise.all([
    docsDemoDir,
    artifactsDemoDir,
    rawVideoDir,
    recordingVideoDir,
    metadataDir,
    demoCacheDir,
    playwrightCacheDir,
  ].map((dirPath) => ensureDir(dirPath)));
}

export async function readFixtureManifest() {
  return JSON.parse(await fs.readFile(fixtureManifestPath, "utf8"));
}

export async function writeJson(filePath, value) {
  await ensureDir(path.dirname(filePath));
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

export async function copyFileEnsured(sourcePath, targetPath) {
  await ensureDir(path.dirname(targetPath));
  await fs.copyFile(sourcePath, targetPath);
  return targetPath;
}

export async function removeFileIfExists(filePath) {
  try {
    await fs.rm(filePath, { force: true });
  } catch {
    // Ignore cleanup errors so reruns stay resilient.
  }
}

export async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

export async function clearHeroOutputs() {
  await Promise.all([heroRawVideoPath, heroWebmPath, heroMp4Path, heroGifPath].map((filePath) => removeFileIfExists(filePath)));
  await fs.rm(recordingVideoDir, { recursive: true, force: true });
  await ensureDir(recordingVideoDir);
}

export async function clearScreenshotOutputs() {
  await Promise.all(screenshotPaths.map((filePath) => removeFileIfExists(filePath)));
}

export function runProcess(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    env: process.env,
    stdio: "inherit",
    ...options,
  });
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed with exit code ${result.status ?? "unknown"}`);
  }
}

export function runPlaywrightSpec(specFile) {
  const resolvedSpecFile = path.basename(specFile);
  runProcess(
    "npx",
    ["playwright", "test", resolvedSpecFile, "--config=playwright.demo.config.mjs"],
    {
      cwd: appsWebDir,
      env: {
        ...process.env,
        DEMO_API_BASE_URL: resolveDemoApiBaseUrl(),
        DEMO_WEB_BASE_URL: resolveDemoWebBaseUrl(),
        TMPDIR: process.env.TMPDIR ?? demoCacheDir,
        TEMP: process.env.TEMP ?? demoCacheDir,
        TMP: process.env.TMP ?? demoCacheDir,
        PWTEST_CACHE_DIR: process.env.PWTEST_CACHE_DIR ?? playwrightCacheDir,
      },
    },
  );
}

export async function validateRequiredFiles(filePaths) {
  const missing = [];
  for (const filePath of filePaths) {
    if (!(await fileExists(filePath))) missing.push(filePath);
  }
  if (missing.length > 0) {
    throw new Error(`Missing expected demo outputs:\n${missing.map((filePath) => `- ${path.relative(repoRoot, filePath)}`).join("\n")}`);
  }
}

export function printGeneratedPaths(filePaths) {
  const seen = new Set();
  console.log("Generated demo assets:");
  for (const filePath of filePaths) {
    if (!filePath || seen.has(filePath) || !existsSync(filePath)) continue;
    seen.add(filePath);
    console.log(`- ${path.relative(repoRoot, filePath)}`);
  }
}
