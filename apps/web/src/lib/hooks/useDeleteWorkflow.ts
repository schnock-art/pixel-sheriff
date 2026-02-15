import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from "react";

import { deleteAsset, deleteProject, type Annotation } from "../api";
import { asRelativePath } from "../workspace/tree";
import type { PendingAnnotation } from "./useAnnotationWorkflow";

interface UseDeleteWorkflowParams {
  selectedProjectId: string | null;
  selectedDatasetName: string;
  selectedTreeFolderPath: string | null;
  setSelectedTreeFolderPath: Dispatch<SetStateAction<string | null>>;
  currentAsset: { id: string; uri: string; metadata_json: Record<string, unknown> } | null;
  assetRows: Array<{ id: string }>;
  assets: Array<{ id: string }>;
  treeFolderAssetIds: Record<string, string[]>;
  assetById: Map<string, { id: string }>;
  annotations: Annotation[];
  setAnnotations: Dispatch<SetStateAction<Annotation[]>>;
  pendingAnnotations: Record<string, PendingAnnotation>;
  setPendingAnnotations: Dispatch<SetStateAction<Record<string, PendingAnnotation>>>;
  setAssetIndex: Dispatch<SetStateAction<number>>;
  setCollapsedFolders: Dispatch<SetStateAction<Record<string, boolean>>>;
  setSelectedProjectId: Dispatch<SetStateAction<string | null>>;
  setProjectMultiLabelSettings: Dispatch<SetStateAction<Record<string, boolean>>>;
  setImportExistingProjectId: Dispatch<SetStateAction<string>>;
  setSelectedImportExistingFolder: Dispatch<SetStateAction<string>>;
  setImportFolderOptionsByProject: Dispatch<SetStateAction<Record<string, string[]>>>;
  setMessage: Dispatch<SetStateAction<string | null>>;
  resetAnnotationWorkflow: () => void;
  setEditMode: Dispatch<SetStateAction<boolean>>;
  refetchAssets: (projectIdOverride?: string | null) => Promise<unknown>;
  refetchProjects: () => Promise<unknown>;
}

export function useDeleteWorkflow({
  selectedProjectId,
  selectedDatasetName,
  selectedTreeFolderPath,
  setSelectedTreeFolderPath,
  currentAsset,
  assetRows,
  assets,
  treeFolderAssetIds,
  assetById,
  annotations,
  setAnnotations,
  pendingAnnotations,
  setPendingAnnotations,
  setAssetIndex,
  setCollapsedFolders,
  setSelectedProjectId,
  setProjectMultiLabelSettings,
  setImportExistingProjectId,
  setSelectedImportExistingFolder,
  setImportFolderOptionsByProject,
  setMessage,
  resetAnnotationWorkflow,
  setEditMode,
  refetchAssets,
  refetchProjects,
}: UseDeleteWorkflowParams) {
  const [isDeletingAssets, setIsDeletingAssets] = useState(false);
  const [isDeletingProject, setIsDeletingProject] = useState(false);
  const [bulkDeleteMode, setBulkDeleteMode] = useState(false);
  const [selectedDeleteAssets, setSelectedDeleteAssets] = useState<Record<string, boolean>>({});

  const selectedDeleteAssetIds = useMemo(
    () => Object.keys(selectedDeleteAssets).filter((assetId) => selectedDeleteAssets[assetId]),
    [selectedDeleteAssets],
  );

  useEffect(() => {
    setSelectedDeleteAssets((previous) => {
      const next: Record<string, boolean> = {};
      for (const assetId of Object.keys(previous)) {
        if (assetById.has(assetId) && previous[assetId]) next[assetId] = true;
      }
      const previousKeys = Object.keys(previous);
      const nextKeys = Object.keys(next);
      if (previousKeys.length === nextKeys.length && previousKeys.every((key) => next[key] === previous[key])) return previous;
      return next;
    });
  }, [assetById]);

  async function deleteAssetsWithSummary(assetIds: string[], contextLabel: string) {
    if (!selectedProjectId) {
      setMessage("Select a dataset before deleting assets.");
      return { removed: 0, failed: assetIds.length };
    }

    const uniqueIds = Array.from(new Set(assetIds)).filter((assetId) => assetById.has(assetId));
    if (uniqueIds.length === 0) {
      setMessage(`No images selected to remove in ${contextLabel}.`);
      return { removed: 0, failed: 0 };
    }

    const targetSet = new Set(uniqueIds);
    const removedAssetIds: string[] = [];
    let removed = 0;
    let failed = 0;

    try {
      setIsDeletingAssets(true);
      setMessage(null);

      for (const assetId of uniqueIds) {
        try {
          await deleteAsset(selectedProjectId, assetId);
          removed += 1;
          removedAssetIds.push(assetId);
        } catch {
          failed += 1;
        }
      }

      if (removed > 0) {
        const removedAssetSet = new Set(removedAssetIds);
        const removedAnnotationCount = annotations.reduce(
          (count, annotation) => (removedAssetSet.has(annotation.asset_id) ? count + 1 : count),
          0,
        );
        setPendingAnnotations((previous) => {
          const next = { ...previous };
          for (const assetId of uniqueIds) delete next[assetId];
          return next;
        });
        setAnnotations((previous) => previous.filter((annotation) => !targetSet.has(annotation.asset_id)));
        await refetchAssets(selectedProjectId);
        setMessage(
          `Deleted ${removed}/${uniqueIds.length} images from ${contextLabel} (annotations removed: ${removedAnnotationCount}${
            failed > 0 ? `, failed: ${failed}` : ""
          }).`,
        );
      } else {
        setMessage(`Deleted 0/${uniqueIds.length} images from ${contextLabel}${failed > 0 ? ` (failed: ${failed}).` : "."}`);
      }
      setSelectedDeleteAssets((previous) => {
        const next = { ...previous };
        for (const assetId of uniqueIds) delete next[assetId];
        return next;
      });
      return { removed, failed };
    } finally {
      setIsDeletingAssets(false);
    }
  }

  async function handleDeleteCurrentAsset() {
    if (!currentAsset) {
      setMessage("Select an image before removing it.");
      return;
    }

    const assetName = asRelativePath(currentAsset);
    const confirmed = window.confirm(`Remove image "${assetName}" from "${selectedDatasetName}"?`);
    if (!confirmed) return;

    await deleteAssetsWithSummary([currentAsset.id], `"${selectedDatasetName}"`);
  }

  function handleToggleBulkDeleteMode() {
    setBulkDeleteMode((previous) => {
      const next = !previous;
      if (!next) setSelectedDeleteAssets({});
      return next;
    });
  }

  function handleToggleDeleteSelection(assetId: string) {
    if (!bulkDeleteMode) return;
    setSelectedDeleteAssets((previous) => {
      const next = { ...previous };
      if (next[assetId]) delete next[assetId];
      else next[assetId] = true;
      return next;
    });
  }

  function handleSelectAllDeleteScope() {
    const inScopeIds = assetRows.map((asset) => asset.id);
    if (inScopeIds.length === 0) {
      setMessage("No images in current scope.");
      return;
    }
    setSelectedDeleteAssets(Object.fromEntries(inScopeIds.map((assetId) => [assetId, true])));
  }

  function handleClearDeleteSelection() {
    setSelectedDeleteAssets({});
  }

  async function handleDeleteSelectedAssets() {
    if (selectedDeleteAssetIds.length === 0) {
      setMessage("Select one or more images to remove.");
      return;
    }

    const scopeLabel = selectedTreeFolderPath ? `folder "${selectedTreeFolderPath}"` : `project "${selectedDatasetName}"`;
    const confirmed = window.confirm(`Remove ${selectedDeleteAssetIds.length} selected image(s) from ${scopeLabel}?`);
    if (!confirmed) return;

    await deleteAssetsWithSummary(selectedDeleteAssetIds, scopeLabel);
  }

  async function handleDeleteSelectedFolder() {
    if (!selectedTreeFolderPath) {
      setMessage("Select a folder before deleting it.");
      return;
    }
    const folderAssetIds = treeFolderAssetIds[selectedTreeFolderPath] ?? [];
    if (folderAssetIds.length === 0) {
      setMessage(`Folder "${selectedTreeFolderPath}" has no images to delete.`);
      return;
    }

    const confirmed = window.confirm(`Delete folder "${selectedTreeFolderPath}" and ${folderAssetIds.length} image(s) in this subtree?`);
    if (!confirmed) return;

    const result = await deleteAssetsWithSummary(folderAssetIds, `folder "${selectedTreeFolderPath}"`);
    if (result.removed > 0) {
      setSelectedTreeFolderPath(null);
      setAssetIndex(0);
    }
  }

  async function handleDeleteFolderPath(folderPath: string) {
    const folderAssetIds = treeFolderAssetIds[folderPath] ?? [];
    if (folderAssetIds.length === 0) {
      setMessage(`Folder "${folderPath}" has no images to delete.`);
      return;
    }

    const confirmed = window.confirm(`Delete folder "${folderPath}" and ${folderAssetIds.length} image(s) in this subtree?`);
    if (!confirmed) return;

    const result = await deleteAssetsWithSummary(folderAssetIds, `folder "${folderPath}"`);
    if (result.removed > 0) {
      if (selectedTreeFolderPath === folderPath || selectedTreeFolderPath?.startsWith(`${folderPath}/`)) {
        setSelectedTreeFolderPath(null);
        setAssetIndex(0);
      }
      setCollapsedFolders((previous) => {
        const next = { ...previous };
        for (const key of Object.keys(next)) {
          if (key === folderPath || key.startsWith(`${folderPath}/`)) {
            delete next[key];
          }
        }
        return next;
      });
    }
  }

  async function handleDeleteCurrentProject() {
    if (!selectedProjectId) {
      setMessage("Select a dataset before deleting it.");
      return;
    }

    const projectId = selectedProjectId;
    const projectName = selectedDatasetName;
    const confirmed = window.confirm(`Delete project "${projectName}" and all its assets/annotations?`);
    if (!confirmed) return;

    try {
      setIsDeletingProject(true);
      setMessage(null);
      const projectAssetCount = assets.length;
      const projectAnnotationCount = annotations.length;
      await deleteProject(projectId);

      setPendingAnnotations((previous) => {
        if (Object.keys(previous).length === 0) return previous;
        return {};
      });
      resetAnnotationWorkflow();
      setEditMode(false);
      setAssetIndex(0);
      setSelectedTreeFolderPath(null);
      setCollapsedFolders({});
      setImportExistingProjectId("");
      setSelectedImportExistingFolder("");
      setImportFolderOptionsByProject((previous) => {
        const next = { ...previous };
        delete next[projectId];
        return next;
      });
      setProjectMultiLabelSettings((previous) => {
        const next = { ...previous };
        delete next[projectId];
        return next;
      });
      setSelectedProjectId(null);
      await refetchProjects();
      await refetchAssets(null);
      setMessage(
        `Deleted project "${projectName}" (assets removed: ${projectAssetCount}, annotations removed: ${projectAnnotationCount}).`,
      );
    } catch (error) {
      setMessage(error instanceof Error ? `Failed to delete project: ${error.message}` : "Failed to delete project.");
    } finally {
      setIsDeletingProject(false);
    }
  }

  function resetDeleteWorkflow() {
    setBulkDeleteMode(false);
    setSelectedDeleteAssets({});
  }

  return {
    isDeletingAssets,
    isDeletingProject,
    bulkDeleteMode,
    selectedDeleteAssets,
    selectedDeleteAssetIds,
    setSelectedDeleteAssets,
    handleToggleBulkDeleteMode,
    handleToggleDeleteSelection,
    handleSelectAllDeleteScope,
    handleClearDeleteSelection,
    handleDeleteCurrentAsset,
    handleDeleteSelectedAssets,
    handleDeleteSelectedFolder,
    handleDeleteFolderPath,
    handleDeleteCurrentProject,
    resetDeleteWorkflow,
  };
}
