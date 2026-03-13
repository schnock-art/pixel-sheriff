import { expect, test } from "@playwright/test";

import { attrSelector, bootstrapDemo, waitForImageReady, waitForLabelingReady } from "./demoHarness.mjs";

test("reviews pending AI prelabels end to end", async ({ page }) => {
  const demo = await bootstrapDemo();
  if (!demo.prelabelDemo) {
    throw new Error("prelabelDemo metadata is missing. Seed the demo state before running this spec.");
  }

  const [frameOnePath, frameTwoPath] = demo.prelabelDemo.frameAssetPaths;
  const [proposalOneId, , proposalThreeId] = demo.prelabelDemo.proposalIds;

  await page.goto(demo.urls.labeling, { waitUntil: "domcontentloaded" });
  await waitForLabelingReady(page);

  await page.locator(attrSelector("folder-tree-asset", "data-demo-path", frameOnePath)).click();
  await waitForImageReady(page);
  await expect(page.locator("[data-testid='ai-prelabels-panel']")).toContainText("2 pending");

  const firstProposal = page.locator(
    `${attrSelector("ai-prelabels-proposal", "data-proposal-id", proposalOneId)}[data-selected="false"]`,
  );
  await firstProposal.click();
  await page.locator(`${attrSelector("ai-prelabels-proposal", "data-proposal-id", proposalOneId)}[data-selected="true"]`).waitFor();
  await page.locator(`${attrSelector("pending-prelabel-object", "data-proposal-id", proposalOneId)}[data-selected="true"]`).waitFor();

  await page.locator("[data-testid='ai-prelabels-edit-selected']").click();
  await page.locator(attrSelector("geometry-object-item", "data-object-id", `prelabel-${proposalOneId}`)).waitFor();
  await page.locator("[data-testid='annotation-submit-button']").click();
  await page.locator(attrSelector("geometry-object-item", "data-object-id", `prelabel-${proposalOneId}`)).waitFor();
  await page.locator(attrSelector("ai-prelabels-proposal", "data-proposal-id", proposalOneId)).waitFor({ state: "detached" });
  await expect(page.locator("[data-testid='ai-prelabels-panel']")).toContainText("1 pending");

  await page.locator("[data-testid='ai-prelabels-accept-frame']").click();
  await expect(page.locator("[data-testid='ai-prelabels-panel']")).toContainText("No pending proposals on this frame.");

  await page.locator("[data-testid='sequence-next-pending-button']").click();
  await page.locator(`${attrSelector("folder-tree-asset", "data-demo-path", frameTwoPath)}.active`).waitFor();
  await waitForImageReady(page);
  await expect(page.locator("[data-testid='ai-prelabels-panel']")).toContainText("1 pending");

  const finalProposal = page.locator(attrSelector("ai-prelabels-proposal", "data-proposal-id", proposalThreeId));
  await finalProposal.click();
  await page.locator("[data-testid='ai-prelabels-reject-selected']").click();
  await finalProposal.waitFor({ state: "detached" });
  await page.locator(attrSelector("pending-prelabel-object", "data-proposal-id", proposalThreeId)).waitFor({ state: "detached" });
  await expect(page.locator("[data-testid='ai-prelabels-panel']")).toContainText("No pending proposals on this frame.");
});
