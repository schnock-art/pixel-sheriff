import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from "react";

import { upsertAnnotation, type Annotation, type AnnotationStatus } from "../api";
import {
  canSubmitWithStates,
  deriveNextAnnotationStatus,
  getCommittedSelectionState,
  normalizeLabelIds,
  readAnnotationLabelIds,
  resolvePendingAnnotation,
} from "../workspace/annotationState";

export interface PendingAnnotation {
  labelIds: number[];
  status: AnnotationStatus;
}

interface LabelRow {
  id: number;
  name: string;
}

interface UseAnnotationWorkflowParams {
  selectedProjectId: string | null;
  currentAsset: { id: string } | null;
  annotationByAssetId: Map<string, Annotation>;
  activeLabelRows: LabelRow[];
  multiLabelEnabled: boolean;
  editMode: boolean;
  setEditMode: Dispatch<SetStateAction<boolean>>;
  setAnnotations: Dispatch<SetStateAction<Annotation[]>>;
  setMessage: Dispatch<SetStateAction<string | null>>;
}

export function useAnnotationWorkflow({
  selectedProjectId,
  currentAsset,
  annotationByAssetId,
  activeLabelRows,
  multiLabelEnabled,
  editMode,
  setEditMode,
  setAnnotations,
  setMessage,
}: UseAnnotationWorkflowParams) {
  const [selectedLabelIds, setSelectedLabelIds] = useState<number[]>([]);
  const [currentStatus, setCurrentStatus] = useState<AnnotationStatus>("unlabeled");
  const [pendingAnnotations, setPendingAnnotations] = useState<Record<string, PendingAnnotation>>({});
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (!currentAsset) {
      setCurrentStatus("unlabeled");
      setSelectedLabelIds([]);
      return;
    }

    const pending = pendingAnnotations[currentAsset.id];
    if (pending) {
      setCurrentStatus(pending.status);
      setSelectedLabelIds(pending.labelIds);
      return;
    }

    const annotation = annotationByAssetId.get(currentAsset.id);
    if (!annotation) {
      setCurrentStatus("unlabeled");
      setSelectedLabelIds([]);
      return;
    }

    setCurrentStatus(annotation.status);
    setSelectedLabelIds(readAnnotationLabelIds(annotation.payload_json));
  }, [annotationByAssetId, currentAsset, pendingAnnotations]);

  useEffect(() => {
    if (multiLabelEnabled) return;

    if (selectedLabelIds.length > 1) {
      setSelectedLabelIds([selectedLabelIds[0]]);
    }

    setPendingAnnotations((previous) => {
      let changed = false;
      const next: Record<string, PendingAnnotation> = {};
      for (const [assetId, pending] of Object.entries(previous)) {
        if (pending.labelIds.length > 1) {
          changed = true;
          next[assetId] = { ...pending, labelIds: [pending.labelIds[0]] };
        } else {
          next[assetId] = pending;
        }
      }
      return changed ? next : previous;
    });
  }, [multiLabelEnabled, selectedLabelIds]);

  const pendingCount = Object.keys(pendingAnnotations).length;
  const currentCommittedSelectionState = useMemo(
    () => getCommittedSelectionState(currentAsset ? annotationByAssetId.get(currentAsset.id) : null),
    [annotationByAssetId, currentAsset],
  );
  const currentDraftSelectionState = useMemo(() => {
    const activeLabelIds = new Set(activeLabelRows.map((label) => label.id));
    const resolvedLabelIds = normalizeLabelIds(selectedLabelIds.filter((id) => activeLabelIds.has(id)));
    return {
      labelIds: resolvedLabelIds,
      status: deriveNextAnnotationStatus(currentStatus, resolvedLabelIds),
    };
  }, [activeLabelRows, currentStatus, selectedLabelIds]);

  const canSubmit = canSubmitWithStates({
    pendingCount,
    editMode,
    hasCurrentAsset: currentAsset !== null,
    draftState: currentDraftSelectionState,
    committedState: currentCommittedSelectionState,
  });

  function stageLabelSelection(nextLabelIds: number[]) {
    if (!currentAsset) return;
    const normalizedLabelIds = normalizeLabelIds(nextLabelIds);
    const draftState: PendingAnnotation = {
      labelIds: normalizedLabelIds,
      status: deriveNextAnnotationStatus(currentStatus, normalizedLabelIds),
    };
    const committedState = getCommittedSelectionState(annotationByAssetId.get(currentAsset.id));
    const nextPending = resolvePendingAnnotation(draftState, committedState);

    setSelectedLabelIds(draftState.labelIds);
    setCurrentStatus(draftState.status);
    setPendingAnnotations((previous) => {
      const next = { ...previous };
      if (nextPending) next[currentAsset.id] = nextPending;
      else delete next[currentAsset.id];
      return next;
    });
  }

  function getNextToggledLabels(labelId: number): number[] {
    if (multiLabelEnabled) {
      return selectedLabelIds.includes(labelId)
        ? selectedLabelIds.filter((value) => value !== labelId)
        : [...selectedLabelIds, labelId];
    }
    return selectedLabelIds.length === 1 && selectedLabelIds[0] === labelId ? [] : [labelId];
  }

  function handleToggleLabel(id: number) {
    if (!currentAsset) return;
    stageLabelSelection(getNextToggledLabels(id));
  }

  async function submitSingleAnnotation() {
    if (!selectedProjectId || !currentAsset) {
      setMessage("Select a dataset and asset before submitting.");
      return;
    }

    const activeLabelIds = new Set(activeLabelRows.map((label) => label.id));
    const resolvedLabelIds = normalizeLabelIds(selectedLabelIds.filter((id) => activeLabelIds.has(id)));
    const selectedLabel = activeLabelRows.find((label) => label.id === resolvedLabelIds[0]);
    const isUnlabeledSelection = resolvedLabelIds.length === 0;
    if (!isUnlabeledSelection && !selectedLabel) {
      setMessage("Selected label could not be resolved.");
      return;
    }

    const annotation = await upsertAnnotation(selectedProjectId, {
      asset_id: currentAsset.id,
      status: deriveNextAnnotationStatus(currentStatus, resolvedLabelIds),
      payload_json: isUnlabeledSelection
        ? {
            type: "classification",
            category_ids: [],
            coco: { image_id: currentAsset.id, category_id: null },
            source: "web-ui",
          }
        : {
            type: "classification",
            category_id: selectedLabel.id,
            category_ids: resolvedLabelIds,
            category_name: selectedLabel.name,
            coco: { image_id: currentAsset.id, category_id: selectedLabel.id },
            source: "web-ui",
          },
    });

    setAnnotations((previous) => {
      const others = previous.filter((item) => item.asset_id !== annotation.asset_id);
      return [...others, annotation];
    });
    setPendingAnnotations((previous) => {
      const next = { ...previous };
      delete next[currentAsset.id];
      return next;
    });
    setCurrentStatus(annotation.status);
    setMessage(isUnlabeledSelection ? "Cleared annotation labels." : "Saved annotation.");
  }

  async function submitPendingAnnotations() {
    if (!selectedProjectId) {
      setMessage("Select a dataset before submitting.");
      return;
    }

    const entries = Object.entries(pendingAnnotations);
    if (entries.length === 0) {
      setMessage("No staged edits to submit.");
      return;
    }

    const saved: Annotation[] = [];
    const activeLabelIds = new Set(activeLabelRows.map((label) => label.id));
    for (const [assetId, pending] of entries) {
      const selectedIds = normalizeLabelIds(pending.labelIds.filter((id) => activeLabelIds.has(id)));
      const label = activeLabelRows.find((item) => item.id === selectedIds[0]);
      const isUnlabeledSelection = selectedIds.length === 0;
      if (!isUnlabeledSelection && !label) continue;

      const annotation = await upsertAnnotation(selectedProjectId, {
        asset_id: assetId,
        status: deriveNextAnnotationStatus(pending.status, selectedIds),
        payload_json: isUnlabeledSelection
          ? {
              type: "classification",
              category_ids: [],
              coco: { image_id: assetId, category_id: null },
              source: "web-ui",
            }
          : {
              type: "classification",
              category_id: label.id,
              category_ids: selectedIds,
              category_name: label.name,
              coco: { image_id: assetId, category_id: label.id },
              source: "web-ui",
            },
      });
      saved.push(annotation);
    }

    setAnnotations((previous) => {
      const savedAssetIds = new Set(saved.map((item) => item.asset_id));
      const others = previous.filter((item) => !savedAssetIds.has(item.asset_id));
      return [...others, ...saved];
    });
    setPendingAnnotations({});
    setEditMode(false);
    setMessage(`Submitted ${saved.length} staged annotations.`);
  }

  async function handleSubmit() {
    try {
      setIsSaving(true);
      setMessage(null);
      if (pendingCount > 0) {
        await submitPendingAnnotations();
      } else {
        await submitSingleAnnotation();
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Failed to submit annotation.");
    } finally {
      setIsSaving(false);
    }
  }

  function resetAnnotationWorkflow() {
    setSelectedLabelIds([]);
    setCurrentStatus("unlabeled");
    setPendingAnnotations({});
  }

  return {
    selectedLabelIds,
    setSelectedLabelIds,
    currentStatus,
    setCurrentStatus,
    pendingAnnotations,
    setPendingAnnotations,
    pendingCount,
    canSubmit,
    isSaving,
    handleToggleLabel,
    handleSubmit,
    resetAnnotationWorkflow,
  };
}
