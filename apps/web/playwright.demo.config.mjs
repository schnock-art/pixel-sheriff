import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "@playwright/test";

const configDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(configDir, "..", "..");

export default defineConfig({
  testDir: "./tests/demo",
  timeout: 120000,
  expect: {
    timeout: 10000,
  },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  outputDir: path.join(repoRoot, "artifacts", "demo", "playwright-output"),
  use: {
    baseURL: process.env.DEMO_WEB_BASE_URL ?? "http://localhost:3010",
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    colorScheme: "light",
  },
});

