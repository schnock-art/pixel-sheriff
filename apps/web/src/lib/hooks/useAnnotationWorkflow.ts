import { useCallback, useEffect, useMemo, useState, type Dispatch, type SetStateAction } from "react";

import { upsertAnnotation, type Annotation, type AnnotationStatus } from "../api";
import {
  buildAnnotationUpsertInput,
  resolveActiveSelection,
} from "../workspace/annotationSubmission";
import {
  isAnnotationSubmitNotFoundError,
  prunePendingAnnotationsForKnownAssets,
} from "../workspace/staleSubmit";
import { resolveSelectionForAsset } from "../workspace/annotationWorkflowSelection";
import {
  canSubmitWithStates,
  deriveNextAnnotationStatus,
  getCommittedSelectionState,
  normalizeAnnotationObjects,
  normalizeImageBasis,
  normalizeLabelIds,
  resolvePendingAnnotation,
} from "../workspace/annotationState";

export interface GeometryBBoxObject {
  id: string;
  kind: "bbox";
  category_id: string;
  bbox: number[];
  provenance?: {
    origin_kind: string;
    session_id?: string;
    proposal_id?: string;
    source_model?: string;
    prompt_text?: string;
    confidence?: number;
    review_decision?: string;
  };
}

export interface GeometryPolygonObject {
  id: string;
  kind: "polygon";
  category_id: string;
  segmentation: number[][];
  provenance?: {
    origin_kind: string;
    session_id?: string;
    proposal_id?: string;
    source_model?: string;
    prompt_text?: string;
    confidence?: number;
    review_decision?: string;
  };
}

export type GeometryObject = GeometryBBoxObject | GeometryPolygonObject;

export interface ImageBasis {
  width: number;
  height: number;
}

export interface PendingAnnotation {
  labelIds: string[];
  status: AnnotationStatus;
  objects: GeometryObject[];
  imageBasis: ImageBasis | null;
}

interface LabelRow {
  id: string;
  name: string;
}

interface UseAnnotationWorkflowParams {
  selectedProjectId: string | null;
  selectedTaskId: string | null;
  currentAsset: { id: string; width: number | null; height: number | null } | null;
  availableAssetIds: string[];
  annotationByAssetId: Map<string, Annotation>;
  activeLabelRows: LabelRow[];
  multiLabelEnabled: boolean;
  editMode: boolean;
  setEditMode: Dispatch<SetStateAction<boolean>>;
  setAnnotations: Dispatch<SetStateAction<Annotation[]>>;
  setMessage: Dispatch<SetStateAction<string | null>>;
  onSubmitSuccess?: () => Promise<void> | void;
}

function resolveFallbackImageBasis(currentAsset: UseAnnotationWorkflowParams["currentAsset"]): ImageBasis | null {
  if (!currentAsset) return null;
  if (typeof currentAsset.width !== "number" || currentAsset.width <= 0) return null;
  if (typeof currentAsset.height !== "number" || currentAsset.height <= 0) return null;
  return { width: Math.round(currentAsset.width), height: Math.round(currentAsset.height) };
}

function normalizeGeometryObjectInput(objects: GeometryObject[] | undefined): GeometryObject[] {
  return normalizeAnnotationObjects(objects ?? []) as GeometryObject[];
}

export function useAnnotationWorkflow({
  selectedProjectId,
  selectedTaskId,
  currentAsset,
  availableAssetIds,
  annotationByAssetId,
  activeLabelRows,
  multiLabelEnabled,
  editMode,
  setEditMode,
  setAnnotations,
  setMessage,
  onSubmitSuccess,
}: UseAnnotationWorkflowParams) {
  const [selectedLabelIds, setSelectedLabelIds] = useState<string[]>([]);
  const [currentStatus, setCurrentStatus] = useState<AnnotationStatus>("unlabeled");
  const [pendingAnnotations, setPendingAnnotations] = useState<Record<string, PendingAnnotation>>({});
  const [isSaving, setIsSaving] = useState(false);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [currentImageBasis, setCurrentImageBasis] = useState<ImageBasis | null>(null);

  const currentCommittedSelectionState = useMemo(
    () => getCommittedSelectionState(currentAsset ? annotationByAssetId.get(currentAsset.id) : null),
    [annotationByAssetId, currentAsset],
  );
  const currentPendingAnnotation = currentAsset ? pendingAnnotations[currentAsset.id] ?? null : null;
  const fallbackImageBasis = useMemo(() => resolveFallbackImageBasis(currentAsset), [currentAsset]);

  const currentObjects = useMemo(
    () =>
      normalizeGeometryObjectInput(
        currentPendingAnnotation?.objects ?? (currentCommittedSelectionState.objects as GeometryObject[]) ?? [],
      ),
    [currentCommittedSelectionState.objects, currentPendingAnnotation],
  );
  const currentResolvedImageBasis = useMemo(
    () =>
      normalizeImageBasis(
        currentImageBasis ?? currentPendingAnnotation?.imageBasis ?? currentCommittedSelectionState.imageBasis ?? fallbackImageBasis,
      ) as ImageBasis | null,
    [currentCommittedSelectionState.imageBasis, currentImageBasis, currentPendingAnnotation, fallbackImageBasis],
  );

  useEffect(() => {
    const nextSelection = resolveSelectionForAsset({
      currentAssetId: currentAsset?.id ?? null,
      pendingAnnotations,
      annotationByAssetId,
    });
    setCurrentStatus(nextSelection.status);
    setSelectedLabelIds(nextSelection.labelIds);
  }, [annotationByAssetId, currentAsset, pendingAnnotations]);

  useEffect(() => {
    setCurrentImageBasis(null);
    setSelectedObjectId(null);
  }, [currentAsset?.id]);

  useEffect(() => {
    if (!selectedObjectId) return;
    if (!currentObjects.some((item) => item.id === selectedObjectId)) {
      setSelectedObjectId(null);
    }
  }, [currentObjects, selectedObjectId]);

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

  useEffect(() => {
    if (availableAssetIds.length === 0) return;
    setPendingAnnotations((previous) => {
      const { nextPendingAnnotations, removedAssetIds } = prunePendingAnnotationsForKnownAssets(previous, availableAssetIds);
      if (removedAssetIds.length === 0) return previous;
      return nextPendingAnnotations;
    });
  }, [availableAssetIds]);

  const pendingCount = Object.keys(pendingAnnotations).length;
  const currentDraftSelectionState = useMemo(() => {
    const resolvedLabelIds = resolveActiveSelection(selectedLabelIds, activeLabelRows);
    return {
      labelIds: resolvedLabelIds,
      status: deriveNextAnnotationStatus(currentStatus, resolvedLabelIds, currentObjects.length),
      objects: currentObjects,
      imageBasis: currentResolvedImageBasis,
    };
  }, [activeLabelRows, currentObjects, currentResolvedImageBasis, currentStatus, selectedLabelIds]);

  const canSubmit = canSubmitWithStates({
    pendingCount,
    editMode,
    hasCurrentAsset: currentAsset !== null,
    draftState: currentDraftSelectionState,
    committedState: currentCommittedSelectionState,
  });

  function stageCurrentDraft(nextDraft: PendingAnnotation) {
    if (!currentAsset) return;

    const committedRaw = getCommittedSelectionState(annotationByAssetId.get(currentAsset.id));
    const committedState: PendingAnnotation = {
      labelIds: committedRaw.labelIds,
      status: committedRaw.status,
      objects: normalizeGeometryObjectInput(committedRaw.objects as GeometryObject[]),
      imageBasis: (normalizeImageBasis(committedRaw.imageBasis ?? fallbackImageBasis) as ImageBasis | null) ?? null,
    };
    const nextPending = resolvePendingAnnotation(nextDraft, committedState) as PendingAnnotation | null;

    setPendingAnnotations((previous) => {
      const next = { ...previous };
      if (nextPending) next[currentAsset.id] = nextPending;
      else delete next[currentAsset.id];
      return next;
    });
  }

  function stageLabelSelection(nextLabelIds: string[]) {
    if (!currentAsset) return;
    const normalizedLabelIds = normalizeLabelIds(nextLabelIds);
    const draftState: PendingAnnotation = {
      labelIds: normalizedLabelIds,
      status: deriveNextAnnotationStatus(currentStatus, normalizedLabelIds, currentObjects.length),
      objects: currentObjects,
      imageBasis: currentResolvedImageBasis,
    };

    setSelectedLabelIds(draftState.labelIds);
    setCurrentStatus(draftState.status);
    stageCurrentDraft(draftState);
  }

  function getNextToggledLabels(labelId: string): string[] {
    if (multiLabelEnabled) {
      return selectedLabelIds.includes(labelId)
        ? selectedLabelIds.filter((value) => value !== labelId)
        : [...selectedLabelIds, labelId];
    }
    return selectedLabelIds.length === 1 && selectedLabelIds[0] === labelId ? [] : [labelId];
  }

  function handleToggleLabel(id: string) {
    if (!currentAsset) return;
    stageLabelSelection(getNextToggledLabels(id));
  }

  function clearSelectedLabels() {
    if (!currentAsset) return;
    stageLabelSelection([]);
  }

  function stageGeometry(nextObjects: GeometryObject[], nextImageBasis: ImageBasis | null = currentResolvedImageBasis) {
    if (!currentAsset) return;

    const resolvedLabelIds = resolveActiveSelection(selectedLabelIds, activeLabelRows);
    const normalizedObjects = normalizeGeometryObjectInput(nextObjects);
    const normalizedBasis = (normalizeImageBasis(nextImageBasis) as ImageBasis | null) ?? null;
    const draftState: PendingAnnotation = {
      labelIds: resolvedLabelIds,
      status: deriveNextAnnotationStatus(currentStatus, resolvedLabelIds, normalizedObjects.length),
      objects: normalizedObjects,
      imageBasis: normalizedBasis,
    };

    setSelectedLabelIds(resolvedLabelIds);
    setCurrentStatus(draftState.status);
    setCurrentImageBasis(normalizedBasis);
    stageCurrentDraft(draftState);
  }

  function upsertGeometryObject(object: GeometryObject) {
    const existingIndex = currentObjects.findIndex((item) => item.id === object.id);
    const next = currentObjects.slice();
    if (existingIndex >= 0) next[existingIndex] = object;
    else next.push(object);
    stageGeometry(next);
    setSelectedObjectId(object.id);
  }

  function replaceGeometryObjects(nextObjects: GeometryObject[]) {
    const normalizedObjects = normalizeGeometryObjectInput(nextObjects);
    stageGeometry(normalizedObjects);
    setSelectedObjectId(normalizedObjects[0]?.id ?? null);
  }

  function deleteSelectedGeometryObject() {
    if (!selectedObjectId) return;
    const next = currentObjects.filter((object) => object.id !== selectedObjectId);
    stageGeometry(next);
    setSelectedObjectId(null);
  }

  function assignSelectedGeometryCategory(categoryId: string) {
    if (!selectedObjectId) return;
    const next = currentObjects.map((object) =>
      object.id === selectedObjectId ? { ...object, category_id: categoryId } : object,
    );
    stageGeometry(next);
    setSelectedLabelIds([categoryId]);
  }

  async function submitSingleAnnotation() {
    if (!selectedProjectId || !selectedTaskId || !currentAsset) {
      setMessage("Select a dataset and asset before submitting.");
      return;
    }

    const upsertInput = buildAnnotationUpsertInput({
      assetId: currentAsset.id,
      currentStatus,
      selectedLabelIds,
      activeLabelRows,
      objects: currentObjects,
      imageBasis: currentResolvedImageBasis,
    });
    if (!upsertInput) {
      setMessage("Selected label could not be resolved.");
      return;
    }

    const annotation = await upsertAnnotation(selectedProjectId, {
      task_id: selectedTaskId,
      asset_id: currentAsset.id,
      status: upsertInput.status,
      payload_json: upsertInput.payload_json,
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
    await onSubmitSuccess?.();
    setMessage(upsertInput.isUnlabeledSelection ? "Cleared annotation labels." : "Saved annotation.");
  }

  async function submitPendingAnnotations() {
    if (!selectedProjectId || !selectedTaskId) {
      setMessage("Select a dataset before submitting.");
      return;
    }

    const {
      nextPendingAnnotations: prunedPendingAnnotations,
      removedAssetIds,
    } = prunePendingAnnotationsForKnownAssets(pendingAnnotations, availableAssetIds);
    if (removedAssetIds.length > 0) {
      setPendingAnnotations(prunedPendingAnnotations);
    }

    const entries = Object.entries(prunedPendingAnnotations);
    if (entries.length === 0) {
      setMessage(
        removedAssetIds.length > 0
          ? "No staged edits to submit. Some entries were removed because assets no longer exist."
          : "No staged edits to submit.",
      );
      return;
    }

    const saved: Annotation[] = [];
    for (const [assetId, pending] of entries) {
      const upsertInput = buildAnnotationUpsertInput({
        assetId,
        currentStatus: pending.status,
        selectedLabelIds: pending.labelIds,
        activeLabelRows,
        objects: pending.objects,
        imageBasis: pending.imageBasis,
      });
      if (!upsertInput) continue;

      const annotation = await upsertAnnotation(selectedProjectId, {
        task_id: selectedTaskId,
        asset_id: assetId,
        status: upsertInput.status,
        payload_json: upsertInput.payload_json,
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
    setSelectedObjectId(null);
    await onSubmitSuccess?.();
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
      if (isAnnotationSubmitNotFoundError(error, selectedProjectId)) {
        const {
          nextPendingAnnotations: prunedPendingAnnotations,
          removedAssetIds,
        } = prunePendingAnnotationsForKnownAssets(pendingAnnotations, availableAssetIds);
        if (removedAssetIds.length > 0) {
          setPendingAnnotations(prunedPendingAnnotations);
        }
        setMessage("Submit context is stale. Missing assets were removed from staged edits. Refresh and retry.");
        return;
      }
      setMessage(error instanceof Error ? error.message : "Failed to submit annotation.");
    } finally {
      setIsSaving(false);
    }
  }

  const updateCurrentImageBasis = useCallback(
    (nextImageBasis: ImageBasis | null) => {
      const normalizedBasis = (normalizeImageBasis(nextImageBasis) as ImageBasis | null) ?? null;
      setCurrentImageBasis(normalizedBasis);

      // Avoid restaging pure classification selections on image-basis updates.
      // This prevents selection flicker/reset in label mode while keeping geometry basis synced.
      if (!currentAsset) return;
      if (currentObjects.length === 0) return;
      stageGeometry(currentObjects, normalizedBasis);
    },
    [currentAsset, currentObjects, stageGeometry],
  );

  function resetAnnotationWorkflow() {
    setSelectedLabelIds([]);
    setCurrentStatus("unlabeled");
    setPendingAnnotations({});
    setSelectedObjectId(null);
    setCurrentImageBasis(null);
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
    currentObjects,
    selectedObjectId,
    setSelectedObjectId,
    currentImageBasis: currentResolvedImageBasis,
    setCurrentImageBasis: updateCurrentImageBasis,
    handleToggleLabel,
    clearSelectedLabels,
    assignSelectedGeometryCategory,
    upsertGeometryObject,
    replaceGeometryObjects,
    deleteSelectedGeometryObject,
    handleSubmit,
    resetAnnotationWorkflow,
  };
}
