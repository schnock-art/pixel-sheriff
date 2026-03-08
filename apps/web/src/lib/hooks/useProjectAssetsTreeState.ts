import { useMemo, useState } from "react";

import type { Annotation, AnnotationStatus, Asset } from "../api";
import { buildVisibleTreeEntries } from "../workspace/projectAssetsDerived";
import { asRelativePath, buildTreeEntries, folderChain } from "../workspace/tree";

type WorkspaceAnnotationMode = "labels" | "bbox" | "segmentation";

export function useProjectAssetsTreeState({
  assets,
  annotations,
  assetById,
  projectAnnotationMode,
}: {
  assets: Asset[];
  annotations: Annotation[];
  assetById: Map<string, Asset>;
  projectAnnotationMode: WorkspaceAnnotationMode;
}) {
  const [assetIndex, setAssetIndex] = useState(0);
  const [filterStatus, setFilterStatus] = useState<"all" | AnnotationStatus>("all");
  const [filterCategoryId, setFilterCategoryId] = useState<string>("all");
  const [selectedTreeFolderPath, setSelectedTreeFolderPath] = useState<string | null>(null);
  const [collapsedFolders, setCollapsedFolders] = useState<Record<string, boolean>>({});

  const treeBuild = useMemo(() => buildTreeEntries(assets), [assets]);
  const treeEntries = treeBuild.entries;
  const treeFolderPaths = useMemo(() => Object.keys(treeBuild.folderAssetIds), [treeBuild.folderAssetIds]);
  const visibleTreeEntries = useMemo(() => buildVisibleTreeEntries(treeEntries, collapsedFolders), [collapsedFolders, treeEntries]);
  const orderedAssetRows = useMemo(
    () =>
      treeBuild.orderedAssetIds
        .map((assetId) => assetById.get(assetId))
        .filter((asset): asset is Asset => asset !== undefined),
    [assetById, treeBuild.orderedAssetIds],
  );

  const annotationByAssetId = useMemo(() => {
    const map = new Map<string, Annotation>();
    for (const annotation of annotations) {
      map.set(annotation.asset_id, annotation);
    }
    return map;
  }, [annotations]);

  const filteredAssetRows = useMemo(() => {
    if (!selectedTreeFolderPath) return orderedAssetRows;
    const prefix = `${selectedTreeFolderPath}/`;
    return orderedAssetRows.filter((asset) => asRelativePath(asset).replaceAll("\\", "/").startsWith(prefix));
  }, [orderedAssetRows, selectedTreeFolderPath]);

  const assetRows = useMemo(() => {
    let rows = filteredAssetRows;
    if (filterStatus !== "all") {
      rows = rows.filter((asset) => (annotationByAssetId.get(asset.id)?.status ?? "unlabeled") === filterStatus);
    }
    if (filterCategoryId !== "all") {
      rows = rows.filter((asset) => {
        const payload = annotationByAssetId.get(asset.id)?.payload_json as Record<string, unknown> | undefined;
        if (!payload) return false;
        if (projectAnnotationMode === "labels") {
          const classification = payload.classification as Record<string, unknown> | undefined;
          const categoryIds = classification?.category_ids as unknown[] | undefined;
          return Array.isArray(categoryIds) && categoryIds.includes(filterCategoryId);
        }
        const objects = payload.objects as Array<Record<string, unknown>> | undefined;
        return Array.isArray(objects) && objects.some((object) => object.category_id === filterCategoryId);
      });
    }
    return rows;
  }, [annotationByAssetId, filterCategoryId, filterStatus, filteredAssetRows, projectAnnotationMode]);

  const safeAssetIndex = Math.min(assetIndex, Math.max(assetRows.length - 1, 0));
  const currentAsset = assetRows[safeAssetIndex] ?? null;
  const selectedFolderAssetCount = useMemo(() => {
    if (!selectedTreeFolderPath) return 0;
    return treeBuild.folderAssetIds[selectedTreeFolderPath]?.length ?? 0;
  }, [selectedTreeFolderPath, treeBuild.folderAssetIds]);

  function handlePrevAsset() {
    setAssetIndex((previous) => (previous <= 0 ? 0 : previous - 1));
  }

  function handleNextAsset() {
    setAssetIndex((previous) => (previous >= assetRows.length - 1 ? previous : previous + 1));
  }

  function handleSelectFolderScope(folderPath: string | null) {
    if (folderPath) {
      setCollapsedFolders((previous) => {
        const next = { ...previous };
        for (const path of folderChain(folderPath)) next[path] = false;
        return next;
      });
    }
    setSelectedTreeFolderPath(folderPath);
    setFilterStatus("all");
    setFilterCategoryId("all");
    setAssetIndex(0);
  }

  function handleToggleFolderCollapsed(folderPath: string) {
    setCollapsedFolders((previous) => ({
      ...previous,
      [folderPath]: !Boolean(previous[folderPath]),
    }));
  }

  function handleCollapseAllFolders() {
    const next: Record<string, boolean> = {};
    for (const folderPath of treeFolderPaths) {
      next[folderPath] = true;
    }
    setCollapsedFolders(next);
  }

  function handleExpandAllFolders() {
    setCollapsedFolders({});
  }

  function handleSelectTreeAsset(assetId: string, folderPath?: string) {
    const targetFolderPath = folderPath ?? selectedTreeFolderPath;
    let scopedRows = targetFolderPath
      ? orderedAssetRows.filter((asset) => asRelativePath(asset).replaceAll("\\", "/").startsWith(`${targetFolderPath}/`))
      : orderedAssetRows;

    if (folderPath && folderPath !== selectedTreeFolderPath) {
      setSelectedTreeFolderPath(folderPath);
      setCollapsedFolders((previous) => {
        const next = { ...previous };
        for (const path of folderChain(folderPath)) next[path] = false;
        return next;
      });
    }

    setFilterStatus("all");
    setFilterCategoryId("all");

    let index = scopedRows.findIndex((item) => item.id === assetId);
    if (index < 0) {
      setSelectedTreeFolderPath(null);
      scopedRows = orderedAssetRows;
      index = scopedRows.findIndex((item) => item.id === assetId);
    }
    if (index >= 0) setAssetIndex(index);
  }

  function resetTreeState() {
    setSelectedTreeFolderPath(null);
    setCollapsedFolders({});
    setFilterStatus("all");
    setFilterCategoryId("all");
    setAssetIndex(0);
  }

  return {
    assetIndex,
    setAssetIndex,
    filterStatus,
    setFilterStatus,
    filterCategoryId,
    setFilterCategoryId,
    selectedTreeFolderPath,
    setSelectedTreeFolderPath,
    collapsedFolders,
    setCollapsedFolders,
    treeBuild,
    visibleTreeEntries,
    orderedAssetRows,
    annotationByAssetId,
    assetRows,
    safeAssetIndex,
    currentAsset,
    selectedFolderAssetCount,
    handlePrevAsset,
    handleNextAsset,
    handleSelectFolderScope,
    handleToggleFolderCollapsed,
    handleCollapseAllFolders,
    handleExpandAllFolders,
    handleSelectTreeAsset,
    resetTreeState,
  };
}
