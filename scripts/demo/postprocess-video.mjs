import { spawnSync } from "node:child_process";

import {
  heroGifPath,
  heroMp4Path,
  heroWebmPath,
  removeFileIfExists,
  runProcess,
} from "./common.mjs";

function hasFfmpeg() {
  const result = spawnSync("ffmpeg", ["-version"], { stdio: "ignore" });
  return result.status === 0;
}

export async function postprocessHeroVideo() {
  await Promise.all([removeFileIfExists(heroMp4Path), removeFileIfExists(heroGifPath)]);

  if (!hasFfmpeg()) {
    console.log("ffmpeg not found; skipping MP4 and GIF generation.");
    return [];
  }

  runProcess("ffmpeg", [
    "-y",
    "-i",
    heroWebmPath,
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    heroMp4Path,
  ]);

  runProcess("ffmpeg", [
    "-y",
    "-i",
    heroWebmPath,
    "-vf",
    "fps=6,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
    heroGifPath,
  ]);

  return [heroMp4Path, heroGifPath];
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await postprocessHeroVideo();
}
