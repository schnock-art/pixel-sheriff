import { useEffect, useMemo, useState } from "react";

import { ApiError, listDeployments, predict, type PredictDetectionBox, type TaskKind } from "../api";
import { buildPredictPayload, detectionBoxesToGeometryObjects } from "../workspace/deployHelpers.js";

type DeploymentSummary = {
  deployment_id: string;
  task_id: string | null;
  name: string;
  device_preference: string;
  status: string;
  task: TaskKind;
};

export function useWorkspaceSuggestions({
  selectedProjectId,
  selectedTaskId,
  currentAssetId,
  selectedTaskKind,
  onApplyDetectionSuggestions,
  setMessage,
}: {
  selectedProjectId: string | null;
  selectedTaskId: string | null;
  currentAssetId: string | null;
  selectedTaskKind: TaskKind | null;
  onApplyDetectionSuggestions: (objects: Array<{ id: string; kind: "bbox"; category_id: string; bbox: number[] }>) => void;
  setMessage: (message: string | null) => void;
}) {
  const [deploymentsState, setDeploymentsState] = useState<{
    active_deployment_id: string | null;
    items: DeploymentSummary[];
  }>({ active_deployment_id: null, items: [] });
  const [suggestionPredictions, setSuggestionPredictions] = useState<
    Array<{ class_id: string; class_name: string; score: number }>
  >([]);
  const [suggestionBoxes, setSuggestionBoxes] = useState<PredictDetectionBox[]>([]);
  const [suggestionScoreThreshold, setSuggestionScoreThreshold] = useState(0.3);
  const [selectedDeploymentId, setSelectedDeploymentId] = useState<string | null>(null);
  const [lastInferenceDeviceSelected, setLastInferenceDeviceSelected] = useState<string | null>(null);
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
    setSuggestionPredictions([]);
    setSuggestionBoxes([]);
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
      const response = await predict(
        selectedProjectId,
        buildPredictPayload({
          assetId: currentAssetId,
          deploymentId: selectedDeployment.deployment_id,
          task: selectedDeployment.task,
          scoreThreshold: suggestionScoreThreshold,
        }),
      );
      if (response.task === "bbox") {
        const boxes = response.boxes ?? [];
        setSuggestionPredictions([]);
        setSuggestionBoxes(boxes);
        onApplyDetectionSuggestions(
          detectionBoxesToGeometryObjects(boxes) as Array<{ id: string; kind: "bbox"; category_id: string; bbox: number[] }>,
        );
      } else {
        setSuggestionBoxes([]);
        setSuggestionPredictions(
          (response.predictions ?? []).map((row) => ({
            class_id: row.class_id,
            class_name: row.class_name,
            score: row.score,
          })),
        );
      }
      setLastInferenceDeviceSelected(response.device_selected ?? null);
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

  return {
    availableDeployments,
    selectedDeployment,
    selectedDeploymentId,
    setSelectedDeploymentId,
    suggestionPredictions,
    suggestionBoxes,
    suggestionScoreThreshold,
    setSuggestionScoreThreshold,
    lastInferenceDeviceSelected,
    isSuggesting,
    handleSuggest,
  };
}
