import { useState, type Dispatch, type SetStateAction } from "react";

import { createCategory, deleteCategory, patchCategory } from "../api";
import { formatDeleteLabelErrorMessage } from "../workspace/projectAssetsLabels";

interface LabelRow {
  id: string;
  name: string;
  isActive: boolean;
  displayOrder: number;
}

interface UseProjectAssetsLabelManagementParams {
  selectedProjectId: string | null;
  selectedTaskId: string | null;
  allLabelRows: LabelRow[];
  annotationMode: "labels" | "bbox" | "segmentation";
  isTaskLabelsLocked: boolean;
  geometryCategoryId: string | null;
  setGeometryCategoryId: Dispatch<SetStateAction<string | null>>;
  setSelectedLabelIds: Dispatch<SetStateAction<string[]>>;
  refetchLabels: () => Promise<unknown>;
  setMessage: Dispatch<SetStateAction<string | null>>;
}

export function useProjectAssetsLabelManagement({
  selectedProjectId,
  selectedTaskId,
  allLabelRows,
  annotationMode,
  isTaskLabelsLocked,
  geometryCategoryId,
  setGeometryCategoryId,
  setSelectedLabelIds,
  refetchLabels,
  setMessage,
}: UseProjectAssetsLabelManagementParams) {
  const [isCreatingLabel, setIsCreatingLabel] = useState(false);
  const [isSavingLabelChanges, setIsSavingLabelChanges] = useState(false);
  const [deletingLabelId, setDeletingLabelId] = useState<string | null>(null);

  async function handleCreateLabel(name: string) {
    if (!selectedProjectId || !selectedTaskId) {
      setMessage("Select a project before creating labels.");
      return;
    }
    if (isTaskLabelsLocked) {
      setMessage("This task is locked because dataset versions already exist. Create a new task to change labels.");
      return;
    }

    try {
      setIsCreatingLabel(true);
      setMessage(null);
      const created = await createCategory(selectedProjectId, { task_id: selectedTaskId, name, display_order: allLabelRows.length });
      await refetchLabels();
      if (annotationMode !== "labels") setGeometryCategoryId(created.id);
      setSelectedLabelIds([created.id]);
      setMessage(`Created label "${created.name}".`);
    } catch (error) {
      setMessage(error instanceof Error ? `Failed to create label: ${error.message}` : "Failed to create label.");
    } finally {
      setIsCreatingLabel(false);
    }
  }

  async function handleSaveLabelChanges(
    changes: Array<{ id: string; name: string; isActive: boolean; displayOrder: number }>,
  ) {
    if (isTaskLabelsLocked) {
      setMessage("This task is locked because dataset versions already exist. Create a new task to change labels.");
      return;
    }
    try {
      setIsSavingLabelChanges(true);
      setMessage(null);
      for (const change of changes) {
        await patchCategory(change.id, {
          name: change.name.trim(),
          is_active: change.isActive,
          display_order: change.displayOrder,
        });
      }
      await refetchLabels();
      setMessage("Saved label configuration.");
    } catch (error) {
      setMessage(error instanceof Error ? `Failed to save labels: ${error.message}` : "Failed to save labels.");
    } finally {
      setIsSavingLabelChanges(false);
    }
  }

  async function handleDeleteLabel(labelId: string, labelName: string) {
    if (isTaskLabelsLocked) {
      setMessage("This task is locked because dataset versions already exist. Create a new task to change labels.");
      return;
    }
    if (!window.confirm(`Delete label "${labelName}"? This cannot be undone.`)) return;
    try {
      setDeletingLabelId(labelId);
      await deleteCategory(labelId);
      await refetchLabels();
      setSelectedLabelIds((previous) => previous.filter((id) => id !== labelId));
      if (geometryCategoryId === labelId) {
        setGeometryCategoryId(null);
      }
      setMessage(`Deleted label "${labelName}".`);
    } catch (error) {
      setMessage(formatDeleteLabelErrorMessage(labelName, error));
    } finally {
      setDeletingLabelId(null);
    }
  }

  return {
    isCreatingLabel,
    isSavingLabelChanges,
    deletingLabelId,
    handleCreateLabel,
    handleSaveLabelChanges,
    handleDeleteLabel,
  };
}
