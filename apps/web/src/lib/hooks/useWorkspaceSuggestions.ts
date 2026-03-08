import { useEffect, useMemo, useState } from "react";

import { ApiError, listDeployments, predict } from "../api";

type DeploymentSummary = { deployment_id: string; name: string; device_preference: string; status: string };

export function useWorkspaceSuggestions({
  selectedProjectId,
  currentAssetId,
  setMessage,
}: {
  selectedProjectId: string | null;
  currentAssetId: string | null;
  setMessage: (message: string | null) => void;
}) {
  const [deploymentsState, setDeploymentsState] = useState<{
    active_deployment_id: string | null;
    items: DeploymentSummary[];
  }>({ active_deployment_id: null, items: [] });
  const [suggestionPredictions, setSuggestionPredictions] = useState<
    Array<{ class_id: string; class_name: string; score: number }>
  >([]);
  const [lastInferenceDeviceSelected, setLastInferenceDeviceSelected] = useState<string | null>(null);
  const [isSuggesting, setIsSuggesting] = useState(false);

  const activeDeployment = useMemo(
    () => deploymentsState.items.find((item) => item.deployment_id === deploymentsState.active_deployment_id) ?? null,
    [deploymentsState.active_deployment_id, deploymentsState.items],
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
    setLastInferenceDeviceSelected(null);
  }, [currentAssetId, selectedProjectId]);

  async function handleSuggest() {
    if (!selectedProjectId || !currentAssetId) {
      setMessage("Select an image before requesting suggestions.");
      return;
    }
    try {
      setIsSuggesting(true);
      const response = await predict(selectedProjectId, {
        asset_id: currentAssetId,
        deployment_id: null,
        top_k: 5,
      });
      setSuggestionPredictions(
        (response.predictions ?? []).map((row) => ({
          class_id: row.class_id,
          class_name: row.class_name,
          score: row.score,
        })),
      );
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
    activeDeployment,
    suggestionPredictions,
    lastInferenceDeviceSelected,
    isSuggesting,
    handleSuggest,
  };
}
