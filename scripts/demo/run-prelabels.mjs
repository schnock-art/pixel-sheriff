import { runPlaywrightSpec } from "./common.mjs";

export async function runPrelabelsDemo() {
  runPlaywrightSpec("tests/demo/prelabels-review.spec.mjs");
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await runPrelabelsDemo();
}
