import { clearScreenshotOutputs, printGeneratedPaths, runPlaywrightSpec, screenshotPaths, validateRequiredFiles } from "./common.mjs";

export async function runScreenshotDemo() {
  await clearScreenshotOutputs();
  runPlaywrightSpec("tests/demo/screenshots.spec.mjs");
  await validateRequiredFiles(screenshotPaths);
  printGeneratedPaths(screenshotPaths);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await runScreenshotDemo();
}

