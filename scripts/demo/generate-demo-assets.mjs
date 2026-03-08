import { heroRawVideoPath, heroWebmPath, printGeneratedPaths, screenshotPaths, validateRequiredFiles } from "./common.mjs";
import { runHeroDemo } from "./run-hero.mjs";
import { runScreenshotDemo } from "./run-screenshots.mjs";

export async function generateDemoAssets() {
  await runHeroDemo();
  await runScreenshotDemo();
  await validateRequiredFiles([heroRawVideoPath, heroWebmPath, ...screenshotPaths]);
  printGeneratedPaths([heroRawVideoPath, heroWebmPath, ...screenshotPaths]);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await generateDemoAssets();
}

