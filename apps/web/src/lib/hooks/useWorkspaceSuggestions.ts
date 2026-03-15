import { useEffect, useMemo, useState } from "react";

import { ApiError, listDeployments, predict, predictBatch, type TaskKind } from "../api";
import {
  buildAcceptedPredictionReview,
  buildPredictBatchPayload,
  buildPredictPayload,
  detectionBoxesToPreviewObjects,
  normalizePredictReview,
  resolveDefaultReviewItemId,
} from "../workspace/deployHelpers.js";
import type { GeometryBBoxObject, PredictionReviewMetadata } from "./useAnnotationWorkflow";

type DeploymentSummary = {
  deployment_id: string;
  task_id: string | null;
  name: string;
  device_preference: string;
  status: string;
  task: TaskKind;
};

export interface ClassificationSuggestionReviewItem {
  review_item_id: string;
  class_index: number;
  class_id: string;
  class_name: string;
  score: number;
}

export interface BBoxSuggestionReviewItem {
  review_item_id: string;
  class_index: number;
  class_id: string;
  class_name: string;
  score: number;
  bbox: number[];
}

export interface SuggestionPreviewObject {
  id: string;
  category_id: string;
  bbox: number[];
  label_text: string;
  confidence: number;
}

export interface PendingClassificationSuggestionReview {
  task: "classification";
  asset_id: string;
  deployment_id: string;
  deployment_name: string | null;
  device_selected: string | null;
  device_preference: string | null;
  items: ClassificationSuggestionReviewItem[];
}

export interface PendingBBoxSuggestionReview {
  task: "bbox";
  asset_id: string;
  deployment_id: string;
  deployment_name: string | null;
  device_selected: string | null;
  device_preference: string | null;
  score_threshold: number | null;
  items: BBoxSuggestionReviewItem[];
  preview_objects: SuggestionPreviewObject[];
}

export type PendingSuggestionReview = PendingClassificationSuggestionReview | PendingBBoxSuggestionReview;

type ReviewQueueStatus = "pending" | "accepted" | "rejected" | "empty" | "failed";

interface ReviewQueueEntry {
  status: ReviewQueueStatus;
  review: PendingSuggestionReview | null;
  errorMessage: string | null;
}

interface BatchPredictionScope {
  folderPath: string;
  assetIds: string[];
  deploymentId: string;
}

export function useWorkspaceSuggestions({
  selectedProjectId,
  selectedTaskId,
  currentAssetId,
  selectedTaskKind,
  selectedFolderPath,
  selectedFolderAssetIds,
  onSelectAsset,
  onAcceptDetectionReview,
  onAcceptClassificationReview,
  setMessage,
}: {
  selectedProjectId: string | null;
  selectedTaskId: string | null;
  currentAssetId: string | null;
  selectedTaskKind: TaskKind | null;
  selectedFolderPath: string | null;
  selectedFolderAssetIds: string[];
  onSelectAsset: (assetId: string, folderPath?: string) => void;
  onAcceptDetectionReview: (objects: GeometryBBoxObject[]) => void;
  onAcceptClassificationReview: (categoryId: string, predictionReview: PredictionReviewMetadata) => void;
  setMessage: (message: string | null) => void;
}) {
  const [deploymentsState, setDeploymentsState] = useState<{
    active_deployment_id: string | null;
    items: DeploymentSummary[];
  }>({ active_deployment_id: null, items: [] });
  const [suggestionScoreThreshold, setSuggestionScoreThreshold] = useState(0.3);
  const [selectedDeploymentId, setSelectedDeploymentId] = useState<string | null>(null);
  const [lastInferenceDeviceSelected, setLastInferenceDeviceSelected] = useState<string | null>(null);
  const [reviewQueueByAssetId, setReviewQueueByAssetId] = useState<Record<string, ReviewQueueEntry>>({});
  const [batchScope, setBatchScope] = useState<BatchPredictionScope | null>(null);
  const [selectedReviewItemId, setSelectedReviewItemId] = useState<string | null>(null);
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [isBatchSuggesting, setIsBatchSuggesting] = useState(false);

  const activeDeployment = useMemo(
    () => deploymentsState.items.find((item) => item.deployment_id === deploymentsState.active_deployment_id) ?? null,
    [deploymentsState.active_deployment_id, deploymentsState.items],
  );
  const availableDeployments = useMemo(
    () =>
      deploymentsState.items.filter((item) => {
        if (item.status !== "available") return false;
        if (selectedTaskKind && item.task !== selectedTaskKind) return false;
        if (selectedTaskId && item.task_id && item.task_id !== selectedTaskId) return false;
        return true;
      }),
    [deploymentsState.items, selectedTaskId, selectedTaskKind],
  );
  const selectedDeployment = useMemo(
    () => availableDeployments.find((item) => item.deployment_id === selectedDeploymentId) ?? null,
    [availableDeployments, selectedDeploymentId],
  );
  const currentReviewQueueEntry = useMemo(
    () => (currentAssetId ? reviewQueueByAssetId[currentAssetId] ?? null : null),
    [currentAssetId, reviewQueueByAssetId],
  );
  const pendingReview = currentReviewQueueEntry?.status === "pending" ? currentReviewQueueEntry.review : null;
  const pendingPreviewObjects = useMemo(
    () => (pendingReview?.task === "bbox" ? pendingReview.preview_objects : []),
    [pendingReview],
  );
  const hasPendingReview = pendingReview !== null;
  const currentAssetReviewStatus: ReviewQueueStatus | "none" = currentReviewQueueEntry?.status ?? "none";
  const batchPredictionSummary = useMemo(() => {
    if (!batchScope) return null;
    let pending = 0;
    let accepted = 0;
    let rejected = 0;
    let empty = 0;
    let failed = 0;
    for (const assetId of batchScope.assetIds) {
      const status = reviewQueueByAssetId[assetId]?.status;
      if (status === "pending") pending += 1;
      else if (status === "accepted") accepted += 1;
      else if (status === "rejected") rejected += 1;
      else if (status === "empty") empty += 1;
      else if (status === "failed") failed += 1;
    }
    return {
      folderPath: batchScope.folderPath,
      total: batchScope.assetIds.length,
      pending,
      accepted,
      rejected,
      empty,
      failed,
    };
  }, [batchScope, reviewQueueByAssetId]);

  useEffect(() => {
    let mounted = true;
    async function loadDeployments() {
      if (!selectedProjectId) {
        setDeploymentsState({ active_deployment_id: null, items: [] });
        return;
      }
      try {
        const response = await listDeployments(selectedProjectId);
        if (!mounted) return;
        setDeploymentsState(response);
      } catch {
        if (!mounted) return;
        setDeploymentsState({ active_deployment_id: null, items: [] });
      }
    }
    void loadDeployments();
    return () => {
      mounted = false;
    };
  }, [selectedProjectId]);

  useEffect(() => {
    setReviewQueueByAssetId({});
    setBatchScope(null);
    setSelectedReviewItemId(null);
    setLastInferenceDeviceSelected(null);
  }, [selectedProjectId, selectedTaskId, selectedTaskKind]);

  useEffect(() => {
    const activeCompatible =
      availableDeployments.find((item) => item.deployment_id === deploymentsState.active_deployment_id) ?? null;
    const currentStillAvailable =
      selectedDeploymentId && availableDeployments.some((item) => item.deployment_id === selectedDeploymentId);
    if (currentStillAvailable) return;
    setSelectedDeploymentId(activeCompatible?.deployment_id ?? availableDeployments[0]?.deployment_id ?? null);
  }, [availableDeployments, deploymentsState.active_deployment_id, selectedDeploymentId]);

  useEffect(() => {
    if (!selectedDeploymentId) {
      setReviewQueueByAssetId({});
      setBatchScope(null);
      setSelectedReviewItemId(null);
      setLastInferenceDeviceSelected(null);
      return;
    }
    const hasMismatchedReview = Object.values(reviewQueueByAssetId).some(
      (entry) => entry.review && entry.review.deployment_id !== selectedDeploymentId,
    );
    if ((batchScope && batchScope.deploymentId !== selectedDeploymentId) || hasMismatchedReview) {
      setReviewQueueByAssetId({});
      setBatchScope(null);
      setSelectedReviewItemId(null);
      setLastInferenceDeviceSelected(null);
    }
  }, [batchScope, reviewQueueByAssetId, selectedDeploymentId]);

  useEffect(() => {
    if (!pendingReview) {
      if (selectedReviewItemId !== null) setSelectedReviewItemId(null);
      return;
    }
    const validSelection = pendingReview.items.some((item) => item.review_item_id === selectedReviewItemId);
    if (validSelection) return;
    setSelectedReviewItemId(resolveDefaultReviewItemId(pendingReview));
  }, [pendingReview, selectedReviewItemId]);

  useEffect(() => {
    if (pendingReview?.device_selected) {
      setLastInferenceDeviceSelected(pendingReview.device_selected);
    }
  }, [pendingReview]);

  function findNextPendingBatchAssetId(currentId: string, nextEntries: Record<string, ReviewQueueEntry>) {
    if (!batchScope || batchScope.assetIds.length < 2) return null;
    const currentIndex = batchScope.assetIds.indexOf(currentId);
    if (currentIndex < 0) return null;
    for (let offset = 1; offset < batchScope.assetIds.length; offset += 1) {
      const assetId = batchScope.assetIds[(currentIndex + offset) % batchScope.assetIds.length];
      if (nextEntries[assetId]?.status === "pending") return assetId;
    }
    return null;
  }

  function updatePendingBBoxReview(mutator: (review: PendingBBoxSuggestionReview) => PendingBBoxSuggestionReview | null) {
    if (!currentAssetId) return null;
    let nextReview: PendingBBoxSuggestionReview | null = null;
    setReviewQueueByAssetId((previous) => {
      const entry = previous[currentAssetId];
      if (!entry?.review || entry.review.task !== "bbox" || entry.status !== "pending") {
        return previous;
      }
      nextReview = mutator(entry.review);
      if (!nextReview) {
        return {
          ...previous,
          [currentAssetId]: { status: "rejected", review: null, errorMessage: null },
        };
      }
      return {
        ...previous,
        [currentAssetId]: { status: "pending", review: nextReview, errorMessage: null },
      };
    });
    return nextReview;
  }

  async function handleSuggest() {
    if (!selectedProjectId || !currentAssetId) {
      setMessage("Select an image before requesting suggestions.");
      return;
    }
    if (!selectedDeployment) {
      if (activeDeployment && selectedTaskKind && activeDeployment.task !== selectedTaskKind) {
        setMessage("No deployed models are available for this task. Deploy one in Deploy.");
        return;
      }
      setMessage("No deployed models are available for this task. Deploy one in Deploy.");
      return;
    }
    try {
      setIsSuggesting(true);
      setMessage(null);
      setSelectedReviewItemId(null);
      const response = await predict(
        selectedProjectId,
        buildPredictPayload({
          assetId: currentAssetId,
          deploymentId: selectedDeployment.deployment_id,
          task: selectedDeployment.task,
          scoreThreshold: suggestionScoreThreshold,
        }),
      );
      const nextReview = normalizePredictReview(response, { scoreThreshold: suggestionScoreThreshold }) as PendingSuggestionReview | null;
      setReviewQueueByAssetId((previous) => ({
        ...previous,
        [currentAssetId]:
          nextReview && nextReview.items.length > 0
            ? { status: "pending", review: nextReview, errorMessage: null }
            : { status: "empty", review: null, errorMessage: null },
      }));
      setSelectedReviewItemId(resolveDefaultReviewItemId(nextReview));
      setLastInferenceDeviceSelected(response.device_selected ?? null);
      if (!nextReview || nextReview.items.length === 0) {
        setMessage("No predictions matched the current request.");
      }
    } catch (error) {
      if (error instanceof ApiError && error.responseBody) {
        setMessage(`Suggest failed: ${error.responseBody}`);
      } else {
        setMessage(error instanceof Error ? `Suggest failed: ${error.message}` : "Suggest failed.");
      }
    } finally {
      setIsSuggesting(false);
    }
  }

  async function handleSuggestFolder() {
    if (!selectedProjectId) {
      setMessage("Select a project before running folder inference.");
      return;
    }
    if (!selectedFolderPath) {
      setMessage("Select a folder before running batch inference.");
      return;
    }
    if (selectedFolderAssetIds.length === 0) {
      setMessage("The selected folder does not contain any images.");
      return;
    }
    if (!selectedDeployment) {
      setMessage("No deployed models are available for this task. Deploy one in Deploy.");
      return;
    }

    try {
      setIsBatchSuggesting(true);
      setMessage(null);
      setSelectedReviewItemId(null);
      const response = await predictBatch(
        selectedProjectId,
        buildPredictBatchPayload({
          assetIds: selectedFolderAssetIds,
          deploymentId: selectedDeployment.deployment_id,
          task: selectedDeployment.task,
          scoreThreshold: suggestionScoreThreshold,
        }),
      );

      const nextEntries: Record<string, ReviewQueueEntry> = {};
      for (const prediction of response.predictions) {
        const review = normalizePredictReview(prediction, { scoreThreshold: suggestionScoreThreshold }) as PendingSuggestionReview | null;
        nextEntries[prediction.asset_id] =
          review && review.items.length > 0
            ? { status: "pending", review, errorMessage: null }
            : { status: "empty", review: null, errorMessage: null };
      }
      for (const error of response.errors) {
        nextEntries[error.asset_id] = {
          status: "failed",
          review: null,
          errorMessage: error.message,
        };
      }

      setReviewQueueByAssetId((previous) => {
        const remaining = { ...previous };
        for (const assetId of selectedFolderAssetIds) delete remaining[assetId];
        return { ...remaining, ...nextEntries };
      });
      setBatchScope({
        folderPath: selectedFolderPath,
        assetIds: selectedFolderAssetIds.slice(),
        deploymentId: selectedDeployment.deployment_id,
      });

      const firstPendingAssetId =
        selectedFolderAssetIds.find((assetId) => nextEntries[assetId]?.status === "pending") ?? null;
      const currentDevice =
        (currentAssetId && nextEntries[currentAssetId]?.review?.device_selected) ??
        response.predictions[0]?.device_selected ??
        null;
      setLastInferenceDeviceSelected(currentDevice);

      if (firstPendingAssetId) {
        onSelectAsset(firstPendingAssetId, selectedFolderPath);
      }

      if (response.pending_review_count > 0) {
        setMessage(
          `Batch predictions ready for "${selectedFolderPath}": ${response.pending_review_count} pending, ${response.empty_count} empty, ${response.error_count} failed.`,
        );
      } else {
        setMessage(
          `Batch inference finished for "${selectedFolderPath}": no predictions to review${response.error_count > 0 ? `, ${response.error_count} failed` : ""}.`,
        );
      }
    } catch (error) {
      if (error instanceof ApiError && error.responseBody) {
        setMessage(`Batch inference failed: ${error.responseBody}`);
      } else {
        setMessage(error instanceof Error ? `Batch inference failed: ${error.message}` : "Batch inference failed.");
      }
    } finally {
      setIsBatchSuggesting(false);
    }
  }

  function rejectReview() {
    if (!pendingReview || !currentAssetId) return;
    const nextEntries = {
      ...reviewQueueByAssetId,
      [currentAssetId]: { status: "rejected" as const, review: null, errorMessage: null },
    };
    setReviewQueueByAssetId(nextEntries);
    setSelectedReviewItemId(null);
    const nextAssetId = findNextPendingBatchAssetId(currentAssetId, nextEntries);
    if (nextAssetId) {
      onSelectAsset(nextAssetId, batchScope?.folderPath);
      setMessage("Prediction rejected. Moved to the next pending image.");
      return;
    }
    setMessage("Prediction rejected.");
  }

  function updatePendingBBoxReviewItem(reviewItemId: string, bbox: number[]) {
    const nextReview = updatePendingBBoxReview((review) => {
      const nextItems = review.items.map((item) =>
        item.review_item_id === reviewItemId ? { ...item, bbox: bbox.slice() } : item,
      );
      return {
        ...review,
        items: nextItems,
        preview_objects: detectionBoxesToPreviewObjects(nextItems),
      };
    });
    if (nextReview) {
      setSelectedReviewItemId(reviewItemId);
    }
  }

  function deleteSelectedPendingBBoxReviewItem() {
    if (!pendingReview || pendingReview.task !== "bbox" || !currentAssetId || !selectedReviewItemId) return;

    const nextItems = pendingReview.items.filter((item) => item.review_item_id !== selectedReviewItemId);
    if (nextItems.length === pendingReview.items.length) return;

    if (nextItems.length === 0) {
      const nextEntries = {
        ...reviewQueueByAssetId,
        [currentAssetId]: { status: "rejected" as const, review: null, errorMessage: null },
      };
      setReviewQueueByAssetId(nextEntries);
      setSelectedReviewItemId(null);
      const nextAssetId = findNextPendingBatchAssetId(currentAssetId, nextEntries);
      if (nextAssetId) {
        onSelectAsset(nextAssetId, batchScope?.folderPath);
        setMessage("Removed the last predicted box. Review rejected and moved to the next pending image.");
        return;
      }
      setMessage("Removed the last predicted box. Review rejected.");
      return;
    }

    updatePendingBBoxReview((review) => ({
      ...review,
      items: nextItems,
      preview_objects: detectionBoxesToPreviewObjects(nextItems),
    }));
    setSelectedReviewItemId(nextItems[0]?.review_item_id ?? null);
    setMessage(`Removed predicted box. ${nextItems.length} remaining.`);
  }

  function acceptReview() {
    if (!pendingReview || !currentAssetId) return;
    const accepted = buildAcceptedPredictionReview(pendingReview, selectedReviewItemId) as
      | { task: "bbox"; objects: GeometryBBoxObject[] }
      | { task: "classification"; categoryId: string; predictionReview: PredictionReviewMetadata }
      | null;
    if (!accepted) {
      setMessage("No prediction is ready to accept.");
      return;
    }

    if (accepted.task === "bbox") {
      onAcceptDetectionReview(accepted.objects);
    } else {
      onAcceptClassificationReview(accepted.categoryId, accepted.predictionReview);
    }

    const nextEntries = {
      ...reviewQueueByAssetId,
      [currentAssetId]: { status: "accepted" as const, review: null, errorMessage: null },
    };
    setReviewQueueByAssetId(nextEntries);
    setSelectedReviewItemId(null);
    const acceptedMessage =
      accepted.task === "bbox"
        ? `Accepted ${accepted.objects.length} predicted box${accepted.objects.length === 1 ? "" : "es"} into the draft.`
        : "Accepted predicted class into the draft.";
    const nextAssetId = findNextPendingBatchAssetId(currentAssetId, nextEntries);
    if (nextAssetId) {
      onSelectAsset(nextAssetId, batchScope?.folderPath);
      setMessage(`${acceptedMessage} Moved to the next pending image.`);
      return;
    }
    setMessage(acceptedMessage);
  }

  return {
    availableDeployments,
    selectedDeployment,
    selectedDeploymentId,
    setSelectedDeploymentId,
    suggestionScoreThreshold,
    setSuggestionScoreThreshold,
    lastInferenceDeviceSelected,
    pendingReview,
    pendingPreviewObjects,
    selectedReviewItemId,
    setSelectedReviewItemId,
    hasPendingReview,
    isSuggesting,
    isBatchSuggesting,
    batchPredictionSummary,
    currentAssetReviewStatus,
    handleSuggest,
    handleSuggestFolder,
    updatePendingBBoxReviewItem,
    deleteSelectedPendingBBoxReviewItem,
    acceptReview,
    rejectReview,
  };
}
