import { clearHeroOutputs, heroRawVideoPath, heroWebmPath, printGeneratedPaths, runPlaywrightSpec, validateRequiredFiles } from "./common.mjs";
import { postprocessHeroVideo } from "./postprocess-video.mjs";

export async function runHeroDemo() {
  await clearHeroOutputs();
  runPlaywrightSpec("tests/demo/hero-demo.spec.mjs");
  const postprocessed = await postprocessHeroVideo();
  await validateRequiredFiles([heroRawVideoPath, heroWebmPath]);
  printGeneratedPaths([heroRawVideoPath, heroWebmPath, ...postprocessed]);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await runHeroDemo();
}

