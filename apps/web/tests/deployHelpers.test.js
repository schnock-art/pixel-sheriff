const test = require("node:test");
const assert = require("node:assert/strict");

const {
  buildPredictBatchPayload,
  buildPredictPayload,
  buildAcceptedPredictionReview,
  detectionBoxesToPreviewObjects,
  deploymentTaskForExperiment,
  detectionBoxesToGeometryObjects,
  deviceLabelToPreference,
  normalizePredictReview,
  resolveDefaultReviewItemId,
} = require("../src/lib/workspace/deployHelpers.js");

test("deviceLabelToPreference maps dropdown values to patch payload", () => {
  assert.equal(deviceLabelToPreference("Auto"), "auto");
  assert.equal(deviceLabelToPreference("CUDA"), "cuda");
  assert.equal(deviceLabelToPreference("CPU"), "cpu");
});

test("deploymentTaskForExperiment maps detection experiments to bbox deployments", () => {
  assert.equal(deploymentTaskForExperiment("classification"), "classification");
  assert.equal(deploymentTaskForExperiment("detection"), "bbox");
  assert.equal(deploymentTaskForExperiment("bbox"), "bbox");
});

test("buildPredictPayload sends score threshold only for bbox suggestions", () => {
  assert.deepEqual(buildPredictPayload({ assetId: "asset-1", deploymentId: "dep-1", task: "classification", scoreThreshold: 0.7 }), {
    asset_id: "asset-1",
    deployment_id: "dep-1",
    top_k: 5,
  });
  assert.deepEqual(buildPredictPayload({ assetId: "asset-1", deploymentId: "dep-1", task: "bbox", scoreThreshold: 0.7 }), {
    asset_id: "asset-1",
    deployment_id: "dep-1",
    score_threshold: 0.7,
  });
});

test("buildPredictBatchPayload sends asset ids and task-specific inference options", () => {
  assert.deepEqual(
    buildPredictBatchPayload({ assetIds: ["asset-1", "asset-2"], deploymentId: "dep-1", task: "classification", scoreThreshold: 0.7 }),
    {
      asset_ids: ["asset-1", "asset-2"],
      deployment_id: "dep-1",
      top_k: 5,
    },
  );
  assert.deepEqual(
    buildPredictBatchPayload({ assetIds: ["asset-1", "asset-2"], deploymentId: "dep-1", task: "bbox", scoreThreshold: 0.7 }),
    {
      asset_ids: ["asset-1", "asset-2"],
      deployment_id: "dep-1",
      score_threshold: 0.7,
    },
  );
});

test("detectionBoxesToPreviewObjects converts predict response boxes into preview overlay objects", () => {
  assert.deepEqual(
    detectionBoxesToPreviewObjects([{ review_item_id: "prediction-bbox-1", class_id: "cat-1", class_name: "cat", score: 0.9, bbox: [1, 2, 3, 4] }]),
    [{ id: "prediction-bbox-1", category_id: "cat-1", bbox: [1, 2, 3, 4], label_text: "cat", confidence: 0.9 }],
  );
});

test("detectionBoxesToGeometryObjects adds deployment provenance for accepted bbox predictions", () => {
  assert.deepEqual(
    detectionBoxesToGeometryObjects([{ review_item_id: "prediction-bbox-1", class_id: "cat-1", score: 0.9, bbox: [1, 2, 3, 4] }], {
      sourceModel: "truck-detector",
      reviewDecision: "accepted",
    }),
    [
      {
        id: "prediction-bbox-1",
        kind: "bbox",
        category_id: "cat-1",
        bbox: [1, 2, 3, 4],
        provenance: {
          origin_kind: "deployment_prediction",
          source_model: "truck-detector",
          confidence: 0.9,
          review_decision: "accepted",
        },
      },
    ],
  );
});

test("normalizePredictReview builds bbox review state with preview objects", () => {
  const review = normalizePredictReview(
    {
      asset_id: "asset-1",
      deployment_id: "dep-1",
      deployment_name: "detector",
      device_selected: "cpu",
      device_preference: "auto",
      task: "bbox",
      boxes: [{ class_index: 0, class_id: "cat-1", class_name: "cat", score: 0.85, bbox: [1, 2, 3, 4] }],
    },
    { scoreThreshold: 0.6 },
  );

  assert.equal(review.task, "bbox");
  assert.equal(review.score_threshold, 0.6);
  assert.equal(resolveDefaultReviewItemId(review), "prediction-bbox-1");
  assert.equal(review.preview_objects[0].label_text, "cat");
});

test("buildAcceptedPredictionReview returns classification metadata for the selected prediction", () => {
  const accepted = buildAcceptedPredictionReview(
    {
      task: "classification",
      asset_id: "asset-1",
      deployment_id: "dep-1",
      deployment_name: "cls-v1",
      device_selected: "cuda",
      device_preference: "auto",
      items: [
        { review_item_id: "prediction-class-cat", class_id: "cat", class_name: "Cat", score: 0.2 },
        { review_item_id: "prediction-class-dog", class_id: "dog", class_name: "Dog", score: 0.8 },
      ],
    },
    "prediction-class-dog",
  );
  assert.deepEqual(accepted, {
    task: "classification",
    categoryId: "dog",
    predictionReview: {
      origin_kind: "deployment_prediction",
      task: "classification",
      deployment_id: "dep-1",
      deployment_name: "cls-v1",
      device_selected: "cuda",
      device_preference: "auto",
      selected_class_id: "dog",
      selected_class_name: "Dog",
      score: 0.8,
    },
  });
});
