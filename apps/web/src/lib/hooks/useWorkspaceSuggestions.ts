import { useEffect, useMemo, useState } from "react";

import { ApiError, listDeployments, predict, type TaskKind } from "../api";
import {
  buildAcceptedPredictionReview,
  buildPredictPayload,
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

export function useWorkspaceSuggestions({
  selectedProjectId,
  selectedTaskId,
  currentAssetId,
  selectedTaskKind,
  onAcceptDetectionReview,
  onAcceptClassificationReview,
  setMessage,
}: {
  selectedProjectId: string | null;
  selectedTaskId: string | null;
  currentAssetId: string | null;
  selectedTaskKind: TaskKind | null;
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
  const [pendingReview, setPendingReview] = useState<PendingSuggestionReview | null>(null);
  const [selectedReviewItemId, setSelectedReviewItemId] = useState<string | null>(null);
  const [isSuggesting, setIsSuggesting] = useState(false);

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
  const pendingPreviewObjects = useMemo(
    () => (pendingReview?.task === "bbox" ? pendingReview.preview_objects : []),
    [pendingReview],
  );
  const hasPendingReview = pendingReview !== null;

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
    setPendingReview(null);
    setSelectedReviewItemId(null);
    setLastInferenceDeviceSelected(null);
  }, [currentAssetId, selectedProjectId, selectedTaskId, selectedTaskKind]);

  useEffect(() => {
    const activeCompatible =
      availableDeployments.find((item) => item.deployment_id === deploymentsState.active_deployment_id) ?? null;
    const currentStillAvailable =
      selectedDeploymentId && availableDeployments.some((item) => item.deployment_id === selectedDeploymentId);
    if (currentStillAvailable) return;
    setSelectedDeploymentId(activeCompatible?.deployment_id ?? availableDeployments[0]?.deployment_id ?? null);
  }, [availableDeployments, deploymentsState.active_deployment_id, selectedDeploymentId]);

  useEffect(() => {
    if (!pendingReview) return;
    if (!selectedDeploymentId || pendingReview.deployment_id !== selectedDeploymentId) {
      setPendingReview(null);
      setSelectedReviewItemId(null);
    }
  }, [pendingReview, selectedDeploymentId]);

  useEffect(() => {
    if (!pendingReview) {
      if (selectedReviewItemId !== null) setSelectedReviewItemId(null);
      return;
    }
    const validSelection = pendingReview.items.some((item) => item.review_item_id === selectedReviewItemId);
    if (validSelection) return;
    setSelectedReviewItemId(resolveDefaultReviewItemId(pendingReview));
  }, [pendingReview, selectedReviewItemId]);

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
      setPendingReview(null);
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
      setPendingReview(nextReview);
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

  function rejectReview() {
    if (!pendingReview) return;
    setPendingReview(null);
    setSelectedReviewItemId(null);
    setMessage("Prediction rejected.");
  }

  function acceptReview() {
    if (!pendingReview) return;
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
      setMessage(`Accepted ${accepted.objects.length} predicted box${accepted.objects.length === 1 ? "" : "es"} into the draft.`);
    } else {
      onAcceptClassificationReview(accepted.categoryId, accepted.predictionReview);
      setMessage("Accepted predicted class into the draft.");
    }

    setPendingReview(null);
    setSelectedReviewItemId(null);
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
    handleSuggest,
    acceptReview,
    rejectReview,
  };
}
