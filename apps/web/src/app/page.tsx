"use client";

import { useEffect, useMemo, useState } from "react";

import { AssetGrid } from "../components/AssetGrid";
import { Filters } from "../components/Filters";
import { LabelPanel } from "../components/LabelPanel";
import { Viewer } from "../components/Viewer";
import {
  ApiError,
  deleteAsset,
  deleteProject,
  createExport,
  createCategory,
  createProject,
  listAssets,
  patchCategory,
  resolveAssetUri,
  uploadAsset,
  upsertAnnotation,
  type Annotation,
  type AnnotationStatus,
} from "../lib/api";
import { useAssets } from "../lib/hooks/useAssets";
import { useLabels } from "../lib/hooks/useLabels";
import { useProject } from "../lib/hooks/useProject";

const IMAGE_EXTENSIONS = new Set(["jpg", "jpeg", "png", "gif", "bmp", "webp", "tif", "tiff"]);
const PROJECT_MULTILABEL_STORAGE_KEY = "pixel-sheriff:project-multilabel:v1";
const LAST_PROJECT_STORAGE_KEY = "pixel-sheriff:last-project-id:v1";

interface PendingAnnotation {
  labelIds: number[];
  status: AnnotationStatus;
}

interface TreeEntry {
  key: string;
  name: string;
  depth: number;
  kind: "folder" | "file";
  path: string;
  assetId?: string;
  folderPath?: string;
}

interface TreeBuildResult {
  entries: TreeEntry[];
  orderedAssetIds: string[];
  folderAssetIds: Record<string, string[]>;
}

type FolderReviewStatus = "all_labeled" | "has_unlabeled" | "empty";

interface ImportDialogState {
  open: boolean;
  sourceFolderName: string;
  files: File[];
}

interface ImportProgressState {
  totalFiles: number;
  completedFiles: number;
  uploadedFiles: number;
  failedFiles: number;
  totalBytes: number;
  processedBytes: number;
  startedAtMs: number;
  activeFileName: string | null;
}

function isImageCandidate(file: File): boolean {
  if (file.type.toLowerCase().startsWith("image/")) return true;
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  return IMAGE_EXTENSIONS.has(extension);
}

function buildTargetRelativePath(file: File, targetFolder: string): string {
  const normalizedFolder = targetFolder.replaceAll("\\", "/").replace(/^\/+|\/+$/g, "");
  const relative = (file.webkitRelativePath || file.name).replaceAll("\\", "/");
  const parts = relative.split("/").filter(Boolean);
  const remainder = file.webkitRelativePath ? parts.slice(1).join("/") : file.name;
  return `${normalizedFolder}/${remainder || file.name}`;
}

function asRelativePath(asset: { uri: string; metadata_json: Record<string, unknown> }): string {
  const fromMetadata = asset.metadata_json.relative_path;
  if (typeof fromMetadata === "string" && fromMetadata.trim() !== "") return fromMetadata;
  const original = asset.metadata_json.original_filename;
  if (typeof original === "string" && original.trim() !== "") return original;
  return asset.uri;
}

function collectFolderPaths(assets: { uri: string; metadata_json: Record<string, unknown> }[]): string[] {
  return collectFolderPathsFromRelativePaths(assets.map((asset) => asRelativePath(asset)));
}

function collectFolderPathsFromRelativePaths(relativePaths: string[]): string[] {
  const paths = new Set<string>();
  for (const rawPath of relativePaths) {
    const rel = rawPath.replaceAll("\\", "/");
    const parts = rel.split("/").filter(Boolean);
    const folderParts = parts.slice(0, -1);
    let prefix = "";
    for (const part of folderParts) {
      prefix = prefix ? `${prefix}/${part}` : part;
      paths.add(prefix);
    }
  }
  return Array.from(paths).sort((a, b) => a.localeCompare(b));
}

function folderChain(path: string): string[] {
  const parts = path.split("/").filter(Boolean);
  const chain: string[] = [];
  let prefix = "";
  for (const part of parts) {
    prefix = prefix ? `${prefix}/${part}` : part;
    chain.push(prefix);
  }
  return chain;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0s";
  const rounded = Math.round(seconds);
  const mins = Math.floor(rounded / 60);
  const secs = rounded % 60;
  if (mins === 0) return `${secs}s`;
  return `${mins}m ${secs}s`;
}

function parseLabelShortcutDigit(event: KeyboardEvent): number | null {
  const digitCodeMatch = event.code.match(/^(Digit|Numpad)([1-9])$/);
  if (digitCodeMatch) return Number(digitCodeMatch[2]);
  if (/^[1-9]$/.test(event.key)) return Number(event.key);
  return null;
}

function buildTreeEntries(assets: { id: string; uri: string; metadata_json: Record<string, unknown> }[]): TreeBuildResult {
  type FolderNode = {
    name: string;
    path: string;
    folders: Map<string, FolderNode>;
    files: Array<{ id: string; name: string }>;
  };

  const root: FolderNode = { name: "", path: "", folders: new Map(), files: [] };

  for (const asset of assets) {
    const rel = asRelativePath(asset).replaceAll("\\", "/");
    const segments = rel.split("/").filter(Boolean);
    const filename = segments[segments.length - 1] ?? rel;
    const folderParts = segments.slice(0, -1);

    let cursor = root;
    let prefix = "";
    for (const part of folderParts) {
      prefix = prefix ? `${prefix}/${part}` : part;
      const existing = cursor.folders.get(part);
      if (existing) {
        cursor = existing;
      } else {
        const next: FolderNode = { name: part, path: prefix, folders: new Map(), files: [] };
        cursor.folders.set(part, next);
        cursor = next;
      }
    }

    cursor.files.push({ id: asset.id, name: filename });
  }

  const result: TreeEntry[] = [];
  const orderedAssetIds: string[] = [];
  const folderAssetIds = new Map<string, string[]>();

  function visit(node: FolderNode, depth: number): string[] {
    const subtreeAssetIds: string[] = [];
    const folders = Array.from(node.folders.values()).sort((a, b) => a.name.localeCompare(b.name));
    for (const folder of folders) {
      result.push({
        key: `folder:${folder.path}`,
        name: folder.name,
        depth,
        kind: "folder",
        path: folder.path,
      });
      const childAssetIds = visit(folder, depth + 1);
      folderAssetIds.set(folder.path, childAssetIds);
      subtreeAssetIds.push(...childAssetIds);
    }

    const files = node.files.slice().sort((a, b) => a.name.localeCompare(b.name));
    for (const file of files) {
      orderedAssetIds.push(file.id);
      subtreeAssetIds.push(file.id);
      result.push({
        key: `file:${file.id}`,
        name: file.name,
        depth,
        kind: "file",
        path: file.name,
        assetId: file.id,
        folderPath: node.path,
      });
    }
    return subtreeAssetIds;
  }

  visit(root, 0);
  const folderAssetIdsRecord: Record<string, string[]> = {};
  folderAssetIds.forEach((ids, folderPath) => {
    folderAssetIdsRecord[folderPath] = ids;
  });

  return { entries: result, orderedAssetIds, folderAssetIds: folderAssetIdsRecord };
}

export default function HomePage() {
  const { data: projects, refetch: refetchProjects } = useProject();
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [assetIndex, setAssetIndex] = useState(0);
  const [selectedLabelIds, setSelectedLabelIds] = useState<number[]>([]);
  const [currentStatus, setCurrentStatus] = useState<AnnotationStatus>("unlabeled");
  const [isSaving, setIsSaving] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [isDeletingAssets, setIsDeletingAssets] = useState(false);
  const [isDeletingProject, setIsDeletingProject] = useState(false);
  const [isCreatingLabel, setIsCreatingLabel] = useState(false);
  const [isSavingLabelChanges, setIsSavingLabelChanges] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [projectMultiLabelSettings, setProjectMultiLabelSettings] = useState<Record<string, boolean>>({});
  const [pendingAnnotations, setPendingAnnotations] = useState<Record<string, PendingAnnotation>>({});
  const [message, setMessage] = useState<string | null>(null);
  const [importFailures, setImportFailures] = useState<string[]>([]);
  const [importDialog, setImportDialog] = useState<ImportDialogState>({ open: false, sourceFolderName: "", files: [] });
  const [importMode, setImportMode] = useState<"existing" | "new">("existing");
  const [importExistingProjectId, setImportExistingProjectId] = useState<string>("");
  const [importNewProjectName, setImportNewProjectName] = useState("");
  const [importFolderName, setImportFolderName] = useState("");
  const [selectedTreeFolderPath, setSelectedTreeFolderPath] = useState<string | null>(null);
  const [collapsedFolders, setCollapsedFolders] = useState<Record<string, boolean>>({});
  const [importFolderOptionsByProject, setImportFolderOptionsByProject] = useState<Record<string, string[]>>({});
  const [selectedImportExistingFolder, setSelectedImportExistingFolder] = useState<string>("");
  const [importProgress, setImportProgress] = useState<ImportProgressState | null>(null);
  const [bulkDeleteMode, setBulkDeleteMode] = useState(false);
  const [selectedDeleteAssets, setSelectedDeleteAssets] = useState<Record<string, boolean>>({});
  const [preferredProjectId, setPreferredProjectId] = useState<string | null>(null);
  const [hasLoadedPreferredProject, setHasLoadedPreferredProject] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(LAST_PROJECT_STORAGE_KEY);
      setPreferredProjectId(raw && raw.trim() !== "" ? raw : null);
    } finally {
      setHasLoadedPreferredProject(true);
    }
  }, []);

  useEffect(() => {
    if (!hasLoadedPreferredProject) return;

    if (projects.length === 0) {
      if (selectedProjectId !== null) setSelectedProjectId(null);
      return;
    }

    if (selectedProjectId && projects.some((project) => project.id === selectedProjectId)) {
      return;
    }

    if (preferredProjectId && projects.some((project) => project.id === preferredProjectId)) {
      setSelectedProjectId(preferredProjectId);
      return;
    }

    setSelectedProjectId(projects[0].id);
  }, [hasLoadedPreferredProject, preferredProjectId, projects, selectedProjectId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (selectedProjectId) {
      window.localStorage.setItem(LAST_PROJECT_STORAGE_KEY, selectedProjectId);
    } else {
      window.localStorage.removeItem(LAST_PROJECT_STORAGE_KEY);
    }
  }, [selectedProjectId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(PROJECT_MULTILABEL_STORAGE_KEY);
      if (!raw) return;
      const parsed: unknown = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object") return;

      const normalized: Record<string, boolean> = {};
      for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
        normalized[key] = Boolean(value);
      }
      setProjectMultiLabelSettings(normalized);
    } catch {
      setProjectMultiLabelSettings({});
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(PROJECT_MULTILABEL_STORAGE_KEY, JSON.stringify(projectMultiLabelSettings));
  }, [projectMultiLabelSettings]);

  const datasets = projects.map((project) => ({ id: project.id, name: project.name }));
  const filteredDatasets = datasets.filter((dataset) => dataset.name.toLowerCase().includes(query.trim().toLowerCase()));
  const activeDatasetId = selectedProjectId;
  const multiLabelEnabled = selectedProjectId ? Boolean(projectMultiLabelSettings[selectedProjectId]) : false;

  const { data: assets, annotations, setAnnotations, refetch: refetchAssets, isLoading: isAssetsLoading } = useAssets(selectedProjectId);
  const { data: labels, refetch: refetchLabels } = useLabels(selectedProjectId);

  useEffect(() => {
    if (!importDialog.open) return;
    if (importMode !== "existing") return;
    if (!importExistingProjectId) return;
    if (importFolderOptionsByProject[importExistingProjectId]) return;

    let isActive = true;

    async function loadFolderOptions() {
      try {
        if (importExistingProjectId === selectedProjectId) {
          if (isAssetsLoading) return;
          if (!isActive) return;
          setImportFolderOptionsByProject((previous) => ({
            ...previous,
            [importExistingProjectId]: collectFolderPaths(assets),
          }));
          return;
        }

        const projectAssets = await listAssets(importExistingProjectId);
        if (!isActive) return;
        setImportFolderOptionsByProject((previous) => ({
          ...previous,
          [importExistingProjectId]: collectFolderPaths(projectAssets),
        }));
      } catch {
        if (!isActive) return;
        setImportFolderOptionsByProject((previous) => ({
          ...previous,
          [importExistingProjectId]: [],
        }));
      }
    }

    void loadFolderOptions();
    return () => {
      isActive = false;
    };
  }, [assets, importDialog.open, importExistingProjectId, importFolderOptionsByProject, importMode, isAssetsLoading, selectedProjectId]);
  const assetById = useMemo(() => {
    const map = new Map<string, (typeof assets)[number]>();
    for (const asset of assets) map.set(asset.id, asset);
    return map;
  }, [assets]);

  const treeBuild = useMemo(() => buildTreeEntries(assets), [assets]);
  const treeEntries = treeBuild.entries;
  const treeFolderPaths = useMemo(() => Object.keys(treeBuild.folderAssetIds), [treeBuild.folderAssetIds]);
  const visibleTreeEntries = useMemo(() => {
    function isHiddenByCollapsedAncestor(entry: TreeEntry): boolean {
      const parentPath = entry.kind === "folder" ? entry.path.split("/").slice(0, -1).join("/") : entry.folderPath ?? "";
      if (!parentPath) return false;
      for (const ancestor of folderChain(parentPath)) {
        if (collapsedFolders[ancestor]) return true;
      }
      return false;
    }

    return treeEntries.filter((entry) => !isHiddenByCollapsedAncestor(entry));
  }, [collapsedFolders, treeEntries]);
  const orderedAssetRows = useMemo(
    () =>
      treeBuild.orderedAssetIds
        .map((assetId) => assetById.get(assetId))
        .filter((asset): asset is (typeof assets)[number] => asset !== undefined),
    [assetById, treeBuild.orderedAssetIds],
  );

  const filteredAssetRows = useMemo(() => {
    if (!selectedTreeFolderPath) return orderedAssetRows;
    const prefix = `${selectedTreeFolderPath}/`;
    return orderedAssetRows.filter((asset) => asRelativePath(asset).replaceAll("\\", "/").startsWith(prefix));
  }, [orderedAssetRows, selectedTreeFolderPath]);

  const assetRows = filteredAssetRows;
  const allLabelRows = labels.map((label) => ({
    id: label.id,
    name: label.name,
    isActive: label.is_active,
    displayOrder: label.display_order,
  }));
  const activeLabelRows = allLabelRows.filter((label) => label.isActive).sort((a, b) => a.displayOrder - b.displayOrder);

  const safeAssetIndex = Math.min(assetIndex, Math.max(assetRows.length - 1, 0));
  const currentAsset = assetRows[safeAssetIndex] ?? null;
  const viewerAsset = currentAsset ? { id: currentAsset.id, uri: resolveAssetUri(currentAsset.uri) } : null;

  const annotationByAssetId = useMemo(() => {
    const map = new Map<string, Annotation>();
    for (const annotation of annotations) {
      map.set(annotation.asset_id, annotation);
    }
    return map;
  }, [annotations]);

  const assetReviewStatusById = useMemo(() => {
    const map = new Map<string, "labeled" | "unlabeled">();
    for (const asset of orderedAssetRows) {
      const pending = pendingAnnotations[asset.id];
      if (pending) {
        const isLabeled = pending.status !== "unlabeled" && pending.labelIds.length > 0;
        map.set(asset.id, isLabeled ? "labeled" : "unlabeled");
        continue;
      }
      const annotation = annotationByAssetId.get(asset.id);
      const isLabeled = Boolean(annotation && annotation.status !== "unlabeled");
      map.set(asset.id, isLabeled ? "labeled" : "unlabeled");
    }
    return map;
  }, [annotationByAssetId, orderedAssetRows, pendingAnnotations]);

  const pageStatuses = useMemo(
    () => assetRows.map((asset) => assetReviewStatusById.get(asset.id) ?? "unlabeled"),
    [assetReviewStatusById, assetRows],
  );
  const selectedDeleteAssetIds = useMemo(
    () => Object.keys(selectedDeleteAssets).filter((assetId) => selectedDeleteAssets[assetId]),
    [selectedDeleteAssets],
  );
  const selectedFolderAssetCount = useMemo(() => {
    if (!selectedTreeFolderPath) return 0;
    return treeBuild.folderAssetIds[selectedTreeFolderPath]?.length ?? 0;
  }, [selectedTreeFolderPath, treeBuild.folderAssetIds]);
  const messageTone = useMemo(() => {
    if (!message) return "info";
    const lower = message.toLowerCase();
    if (lower.includes("failed") || lower.includes("error")) return "error";
    return "success";
  }, [message]);

  const folderReviewStatusByPath = useMemo(() => {
    const status: Record<string, FolderReviewStatus> = {};
    for (const [folderPath, assetIds] of Object.entries(treeBuild.folderAssetIds)) {
      if (assetIds.length === 0) {
        status[folderPath] = "empty";
        continue;
      }
      const hasUnlabeled = assetIds.some((assetId) => (assetReviewStatusById.get(assetId) ?? "unlabeled") === "unlabeled");
      status[folderPath] = hasUnlabeled ? "has_unlabeled" : "all_labeled";
    }
    return status;
  }, [assetReviewStatusById, treeBuild.folderAssetIds]);

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
    const categoryIdValue = annotation.payload_json.category_id;
    const categoryIdsValue = annotation.payload_json.category_ids;
    if (Array.isArray(categoryIdsValue)) {
      const ids = categoryIdsValue.filter((item): item is number => typeof item === "number");
      setSelectedLabelIds(ids);
      return;
    }
    if (typeof categoryIdValue === "number") {
      setSelectedLabelIds([categoryIdValue]);
      return;
    }
    setSelectedLabelIds([]);
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

  useEffect(() => {
    setAssetIndex((previous) => Math.min(previous, Math.max(assetRows.length - 1, 0)));
  }, [assetRows.length]);

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

  useEffect(() => {
    if (!message) return;
    const timeout = window.setTimeout(() => setMessage(null), 6000);
    return () => window.clearTimeout(timeout);
  }, [message]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || target?.isContentEditable) return;
      if (event.altKey || event.ctrlKey || event.metaKey) return;

      if (event.key === "ArrowLeft") {
        event.preventDefault();
        setAssetIndex((previous) => (previous <= 0 ? 0 : previous - 1));
        return;
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        setAssetIndex((previous) => (previous >= assetRows.length - 1 ? previous : previous + 1));
        return;
      }

      const digit = parseLabelShortcutDigit(event);
      if (digit === null || digit > activeLabelRows.length) return;
      const label = activeLabelRows[digit - 1];
      if (!label) return;

      event.preventDefault();
      handleToggleLabel(label.id);
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeLabelRows, assetRows.length, currentAsset, currentStatus, multiLabelEnabled, selectedLabelIds]);

  function handleSelectDataset(id: string) {
    setSelectedProjectId(id);
    setSelectedTreeFolderPath(null);
    setCollapsedFolders({});
    setBulkDeleteMode(false);
    setSelectedDeleteAssets({});
    setAssetIndex(0);
    setPendingAnnotations({});
    setEditMode(false);
    setMessage(null);
  }

  function handlePrevAsset() {
    setAssetIndex((previous) => (previous <= 0 ? 0 : previous - 1));
  }

  function handleNextAsset() {
    setAssetIndex((previous) => (previous >= assetRows.length - 1 ? previous : previous + 1));
  }

  function stageLabelSelection(nextLabelIds: number[]) {
    if (!currentAsset) return;
    const normalizedLabelIds = Array.from(new Set(nextLabelIds));
    const nextStatus: AnnotationStatus = normalizedLabelIds.length === 0 ? "unlabeled" : currentStatus === "unlabeled" ? "labeled" : currentStatus;

    setSelectedLabelIds(normalizedLabelIds);
    setCurrentStatus(nextStatus);
    setPendingAnnotations((previous) => ({
      ...previous,
      [currentAsset.id]: { labelIds: normalizedLabelIds, status: nextStatus },
    }));
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

  function handleToggleProjectMultiLabel() {
    if (!selectedProjectId) return;
    setProjectMultiLabelSettings((previous) => ({
      ...previous,
      [selectedProjectId]: !Boolean(previous[selectedProjectId]),
    }));
  }

  async function submitSingleAnnotation() {
    if (!selectedProjectId || !currentAsset) {
      setMessage("Select a dataset and asset before submitting.");
      return;
    }

    const resolvedLabelIds = selectedLabelIds.filter((id) => activeLabelRows.some((label) => label.id === id));
    const selectedLabel = activeLabelRows.find((label) => label.id === resolvedLabelIds[0]);
    const isUnlabeledSelection = resolvedLabelIds.length === 0;
    if (!isUnlabeledSelection && !selectedLabel) {
      setMessage("Selected label could not be resolved.");
      return;
    }

    const annotation = await upsertAnnotation(selectedProjectId, {
      asset_id: currentAsset.id,
      status: isUnlabeledSelection ? "unlabeled" : currentStatus === "unlabeled" ? "labeled" : currentStatus,
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
    for (const [assetId, pending] of entries) {
      const selectedIds = pending.labelIds.filter((id) => activeLabelRows.some((item) => item.id === id));
      const label = activeLabelRows.find((item) => item.id === selectedIds[0]);
      const isUnlabeledSelection = selectedIds.length === 0;
      if (!isUnlabeledSelection && !label) continue;

      const annotation = await upsertAnnotation(selectedProjectId, {
        asset_id: assetId,
        status: isUnlabeledSelection ? "unlabeled" : pending.status,
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
      if (Object.keys(pendingAnnotations).length > 0) {
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
    const folderAssetIds = treeBuild.folderAssetIds[selectedTreeFolderPath] ?? [];
    if (folderAssetIds.length === 0) {
      setMessage(`Folder "${selectedTreeFolderPath}" has no images to delete.`);
      return;
    }

    const confirmed = window.confirm(
      `Delete folder "${selectedTreeFolderPath}" and ${folderAssetIds.length} image(s) in this subtree?`,
    );
    if (!confirmed) return;

    const result = await deleteAssetsWithSummary(folderAssetIds, `folder "${selectedTreeFolderPath}"`);
    if (result.removed > 0) {
      setSelectedTreeFolderPath(null);
      setAssetIndex(0);
    }
  }

  async function handleDeleteFolderPath(folderPath: string) {
    const folderAssetIds = treeBuild.folderAssetIds[folderPath] ?? [];
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

      setPendingAnnotations({});
      setSelectedLabelIds([]);
      setCurrentStatus("unlabeled");
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

  async function handleCreateLabel(name: string) {
    if (!selectedProjectId) {
      setMessage("Select a dataset before creating labels.");
      return;
    }

    try {
      setIsCreatingLabel(true);
      setMessage(null);
      const created = await createCategory(selectedProjectId, { name, display_order: allLabelRows.length });
      await refetchLabels();
      setSelectedLabelIds([created.id]);
      setMessage(`Created label "${created.name}".`);
    } catch (error) {
      setMessage(error instanceof Error ? `Failed to create label: ${error.message}` : "Failed to create label.");
    } finally {
      setIsCreatingLabel(false);
    }
  }

  async function handleSaveLabelChanges(
    changes: Array<{ id: number; name: string; isActive: boolean; displayOrder: number }>,
  ) {
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

  function openImportDialog(files: File[], sourceFolderName: string) {
    const defaultProject = projects.find((project) => project.id === selectedProjectId) ?? projects[0];
    const defaultMode: "existing" | "new" = projects.length > 0 ? "existing" : "new";
    setImportMode(defaultMode);
    setImportExistingProjectId(defaultProject?.id ?? "");
    setSelectedImportExistingFolder("");
    setImportNewProjectName(sourceFolderName);
    setImportFolderName(sourceFolderName);
    setImportProgress(null);
    setImportDialog({ open: true, sourceFolderName, files });
  }

  async function confirmImportFromDialog() {
    const files = importDialog.files;
    const sourceFolderName = importDialog.sourceFolderName;
    const folderName = importFolderName.trim();
    if (files.length === 0) {
      setMessage("Import cancelled: no files selected.");
      setSelectedImportExistingFolder("");
      setImportProgress(null);
      setImportDialog({ open: false, sourceFolderName: "", files: [] });
      return;
    }
    if (!folderName) {
      setMessage("Folder name is required.");
      return;
    }

    try {
      setIsImporting(true);
      setMessage("Importing images...");
      setImportFailures([]);
      const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
      setImportProgress({
        totalFiles: files.length,
        completedFiles: 0,
        uploadedFiles: 0,
        failedFiles: 0,
        totalBytes,
        processedBytes: 0,
        startedAtMs: Date.now(),
        activeFileName: null,
      });

      let targetProjectId = "";
      let targetProjectName = "";

      if (importMode === "new") {
        const projectName = importNewProjectName.trim();
        if (!projectName) {
          setMessage("Project name is required for new project imports.");
          return;
        }
        const project = await createProject({ name: projectName, task_type: "classification_single" });
        targetProjectId = project.id;
        targetProjectName = project.name;
      } else {
        const project = projects.find((item) => item.id === importExistingProjectId);
        if (!project) {
          setMessage("Please select an existing project.");
          return;
        }
        targetProjectId = project.id;
        targetProjectName = project.name;
      }

      let uploadedCount = 0;
      const failures: string[] = [];

      for (const file of files) {
        setImportProgress((previous) => (previous ? { ...previous, activeFileName: file.name } : previous));
        try {
          const targetRelativePath = buildTargetRelativePath(file, folderName);
          await uploadAsset(targetProjectId, file, targetRelativePath);
          uploadedCount += 1;
          setImportProgress((previous) =>
            previous
              ? {
                  ...previous,
                  completedFiles: previous.completedFiles + 1,
                  uploadedFiles: previous.uploadedFiles + 1,
                  processedBytes: previous.processedBytes + file.size,
                  activeFileName: null,
                }
              : previous,
          );
        } catch (error) {
          if (error instanceof ApiError) {
            const reason = error.responseBody ? ` (${error.responseBody})` : "";
            failures.push(`${file.name}: ${error.message}${reason}`);
          } else {
            failures.push(`${file.name}: ${error instanceof Error ? error.message : "unknown upload error"}`);
          }
          setImportProgress((previous) =>
            previous
              ? {
                  ...previous,
                  completedFiles: previous.completedFiles + 1,
                  failedFiles: previous.failedFiles + 1,
                  processedBytes: previous.processedBytes + file.size,
                  activeFileName: null,
                }
              : previous,
          );
        }
      }

      await refetchProjects();
      await refetchAssets(targetProjectId);
      setSelectedProjectId(targetProjectId);
      setSelectedTreeFolderPath(null);
      setCollapsedFolders({});
      setAssetIndex(0);
      setSelectedLabelIds([]);
      setCurrentStatus("unlabeled");
      setPendingAnnotations({});
      setEditMode(false);
      setSelectedImportExistingFolder("");
      setImportFolderOptionsByProject((previous) => {
        const importedRelativePaths = files.map((file) => buildTargetRelativePath(file, folderName));
        const importedFolders = collectFolderPathsFromRelativePaths(importedRelativePaths);
        const merged = new Set([...(previous[targetProjectId] ?? []), ...importedFolders]);
        return {
          ...previous,
          [targetProjectId]: Array.from(merged).sort((a, b) => a.localeCompare(b)),
        };
      });
      setImportFailures(failures);
      setSelectedImportExistingFolder("");
      setImportProgress(null);
      setImportDialog({ open: false, sourceFolderName: "", files: [] });

      if (uploadedCount === 0) setMessage(`Import failed: no files uploaded to "${folderName}".`);
      else if (failures.length > 0)
        setMessage(`Imported ${uploadedCount}/${files.length} images into "${targetProjectName}/${folderName}".`);
      else setMessage(`Imported ${uploadedCount} images into "${targetProjectName}/${folderName}".`);
    } catch (error) {
      setImportFailures([]);
      setImportProgress(null);
      setMessage(error instanceof Error ? `Import failed: ${error.message}` : "Import failed.");
    } finally {
      setIsImporting(false);
    }
  }

  async function handleImport() {
    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = "image/*";
    picker.multiple = true;
    (picker as HTMLInputElement & { webkitdirectory?: boolean }).webkitdirectory = true;

    picker.onchange = async () => {
      const files = Array.from(picker.files ?? []).filter(isImageCandidate);
      if (files.length === 0) {
        setImportFailures([]);
        setMessage("No image files were selected (supported by MIME or extension).");
        return;
      }
      const rootName = files[0].webkitRelativePath.split("/")[0] || `Dataset ${new Date().toLocaleString()}`;
      openImportDialog(files, rootName);
    };

    picker.click();
  }

  async function handleExport() {
    if (!selectedProjectId) {
      setMessage("Select a dataset before exporting.");
      return;
    }

    try {
      setIsExporting(true);
      setMessage("Building export...");
      const created = await createExport(selectedProjectId, {
        selection_criteria_json: { statuses: ["labeled", "approved", "needs_review", "skipped"] },
      });

      const url = resolveAssetUri(created.export_uri);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${selectedDatasetName.replace(/[^a-zA-Z0-9-_]+/g, "_") || "dataset"}-${created.hash.slice(0, 8)}.zip`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();

      const counts = created.manifest_json.counts as Record<string, number> | undefined;
      if (counts && typeof counts.assets === "number" && typeof counts.annotations === "number") {
        setMessage(`Export ready. ${counts.assets} assets, ${counts.annotations} annotations.`);
      } else {
        setMessage("Export ready.");
      }
    } catch (error) {
      setMessage(error instanceof Error ? `Export failed: ${error.message}` : "Export failed.");
    } finally {
      setIsExporting(false);
    }
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
    let scopedRows = assetRows;

    if (folderPath && folderPath !== selectedTreeFolderPath) {
      const prefix = `${folderPath}/`;
      scopedRows = orderedAssetRows.filter((asset) => asRelativePath(asset).replaceAll("\\", "/").startsWith(prefix));
      setSelectedTreeFolderPath(folderPath);
      setCollapsedFolders((previous) => {
        const next = { ...previous };
        for (const path of folderChain(folderPath)) next[path] = false;
        return next;
      });
    }

    let index = scopedRows.findIndex((item) => item.id === assetId);
    if (index < 0) {
      setSelectedTreeFolderPath(null);
      scopedRows = orderedAssetRows;
      index = scopedRows.findIndex((item) => item.id === assetId);
    }

    if (index >= 0) setAssetIndex(index);
  }

  const selectedDatasetName = datasets.find((dataset) => dataset.id === activeDatasetId)?.name ?? "No dataset selected";
  const headerTitle = selectedTreeFolderPath ? `${selectedDatasetName} / ${selectedTreeFolderPath}` : selectedDatasetName;
  const importExistingFolderOptions = importFolderOptionsByProject[importExistingProjectId] ?? [];
  const importProgressView = useMemo(() => {
    if (!importProgress) return null;

    const elapsedSeconds = Math.max((Date.now() - importProgress.startedAtMs) / 1000, 0.001);
    const fileRate = importProgress.completedFiles / elapsedSeconds;
    const byteRate = importProgress.processedBytes / elapsedSeconds;
    const remainingBytes = Math.max(importProgress.totalBytes - importProgress.processedBytes, 0);
    const etaSeconds = byteRate > 0 ? remainingBytes / byteRate : Number.POSITIVE_INFINITY;

    return {
      percent: importProgress.totalFiles > 0 ? Math.round((importProgress.completedFiles / importProgress.totalFiles) * 100) : 0,
      elapsedText: formatDuration(elapsedSeconds),
      etaText: Number.isFinite(etaSeconds) ? formatDuration(etaSeconds) : "--",
      fileRateText: `${fileRate.toFixed(fileRate >= 10 ? 0 : 1)} files/s`,
      speedText: `${formatBytes(byteRate)}/s`,
      progressText: `${importProgress.completedFiles}/${importProgress.totalFiles}`,
      bytesText: `${formatBytes(importProgress.processedBytes)} / ${formatBytes(importProgress.totalBytes)}`,
      remainingFilesText: `${Math.max(importProgress.totalFiles - importProgress.completedFiles, 0)} remaining`,
      uploadedFilesText: `${importProgress.uploadedFiles} uploaded`,
      failedFilesText: `${importProgress.failedFiles} failed`,
      activeFileName: importProgress.activeFileName,
    };
  }, [importProgress]);
  const canSubmit = Object.keys(pendingAnnotations).length > 0 || (!editMode && selectedLabelIds.length > 0 && currentAsset !== null);

  return (
    <main className="workspace-shell">
      <section className="workspace-frame">
        <header className="workspace-header">
          <div className="workspace-header-cell">Datasets</div>
          <div className="workspace-header-cell workspace-header-title">{headerTitle}</div>
          <div className="workspace-header-cell workspace-header-actions" aria-label="Toolbar">
            <span />
            <span />
            <span />
            <span />
          </div>
        </header>

        <div className="workspace-body">
          <aside className="workspace-sidebar">
            <Filters query={query} onQueryChange={setQuery} />
            <AssetGrid datasets={filteredDatasets} selectedDatasetId={activeDatasetId} onSelectDataset={handleSelectDataset} />
            <section className="project-tree">
              <div className="project-tree-head">
                <h3>Files</h3>
                <div className="tree-head-actions">
                  <button type="button" className="tree-scope-button" onClick={handleCollapseAllFolders}>
                    Collapse all
                  </button>
                  <button type="button" className="tree-scope-button" onClick={handleExpandAllFolders}>
                    Expand all
                  </button>
                  <button
                    type="button"
                    className={selectedTreeFolderPath === null ? "tree-scope-button active" : "tree-scope-button"}
                    onClick={() => handleSelectFolderScope(null)}
                  >
                    All files
                  </button>
                </div>
              </div>
              <div className="tree-delete-toolbar">
                <button
                  type="button"
                  className={bulkDeleteMode ? "tree-scope-button danger active" : "tree-scope-button danger"}
                  onClick={handleToggleBulkDeleteMode}
                  disabled={!selectedProjectId || isDeletingAssets}
                >
                  {bulkDeleteMode ? "Exit multi-delete" : "Multi-delete"}
                </button>
                {bulkDeleteMode ? (
                  <>
                    <button type="button" className="tree-scope-button" onClick={handleSelectAllDeleteScope} disabled={isDeletingAssets}>
                      Select scope
                    </button>
                    <button type="button" className="tree-scope-button" onClick={handleClearDeleteSelection} disabled={isDeletingAssets}>
                      Clear
                    </button>
                    <button
                      type="button"
                      className="tree-scope-button danger"
                      onClick={handleDeleteSelectedAssets}
                      disabled={isDeletingAssets || selectedDeleteAssetIds.length === 0}
                    >
                      Delete selected ({selectedDeleteAssetIds.length})
                    </button>
                  </>
                ) : null}
                {selectedTreeFolderPath ? (
                  <button
                    type="button"
                    className="tree-scope-button danger"
                    onClick={handleDeleteSelectedFolder}
                    disabled={isDeletingAssets || selectedFolderAssetCount === 0}
                  >
                    Delete folder ({selectedFolderAssetCount})
                  </button>
                ) : null}
              </div>
              {selectedTreeFolderPath ? <p className="tree-scope-caption">Scope: {selectedTreeFolderPath}</p> : null}
              <ul>
                {visibleTreeEntries.map((entry) => (
                  <li key={entry.key}>
                    {entry.kind === "folder" ? (
                      <div className="tree-folder-row" style={{ paddingLeft: `${entry.depth * 14 + 8}px` }}>
                        <button
                          type="button"
                          className="tree-folder-toggle"
                          aria-label={collapsedFolders[entry.path] ? "Expand folder" : "Collapse folder"}
                          onClick={() => handleToggleFolderCollapsed(entry.path)}
                        >
                          {collapsedFolders[entry.path] ? ">" : "v"}
                        </button>
                        <button
                          type="button"
                          className={`tree-folder-button${selectedTreeFolderPath === entry.path ? " active" : ""} ${
                            folderReviewStatusByPath[entry.path] === "all_labeled"
                              ? "is-labeled"
                              : folderReviewStatusByPath[entry.path] === "has_unlabeled"
                                ? "has-unlabeled"
                                : "is-empty"
                          }`}
                          onClick={() => handleSelectFolderScope(entry.path)}
                        >
                          {entry.name}
                        </button>
                        <button
                          type="button"
                          className="tree-row-delete"
                          onClick={() => void handleDeleteFolderPath(entry.path)}
                          disabled={isDeletingAssets}
                          title={`Delete "${entry.path}"`}
                        >
                          x
                        </button>
                      </div>
                    ) : (
                      <div className="tree-file-row" style={{ paddingLeft: `${entry.depth * 14 + 8}px` }}>
                        {bulkDeleteMode && entry.assetId ? (
                          <input
                            className="tree-file-checkbox"
                            type="checkbox"
                            checked={Boolean(selectedDeleteAssets[entry.assetId])}
                            onChange={() => {
                              if (entry.assetId) handleToggleDeleteSelection(entry.assetId);
                            }}
                            disabled={isDeletingAssets}
                            aria-label={`Select ${entry.name} for delete`}
                          />
                        ) : null}
                        <button
                          type="button"
                          className={`tree-file${entry.assetId === currentAsset?.id ? " active" : ""} ${
                            entry.assetId && assetReviewStatusById.get(entry.assetId) === "labeled" ? "is-labeled" : "is-unlabeled"
                          }${entry.assetId && selectedDeleteAssets[entry.assetId] ? " delete-selected" : ""}`}
                          onClick={() =>
                            entry.assetId &&
                            (bulkDeleteMode ? handleToggleDeleteSelection(entry.assetId) : handleSelectTreeAsset(entry.assetId, entry.folderPath))
                          }
                        >
                          {entry.name}
                        </button>
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          </aside>

          <Viewer
            currentAsset={viewerAsset}
            totalAssets={assetRows.length}
            currentIndex={safeAssetIndex}
            pageStatuses={pageStatuses}
            onSelectIndex={setAssetIndex}
            onPrev={handlePrevAsset}
            onNext={handleNextAsset}
          />
          <LabelPanel
            labels={activeLabelRows}
            allLabels={allLabelRows}
            selectedLabelIds={selectedLabelIds}
            onToggleLabel={handleToggleLabel}
            onSubmit={handleSubmit}
            isSaving={isSaving}
            onCreateLabel={handleCreateLabel}
            isCreatingLabel={isCreatingLabel}
            editMode={editMode}
            onToggleEditMode={() => setEditMode((value) => !value)}
            pendingCount={Object.keys(pendingAnnotations).length}
            onSaveLabelChanges={handleSaveLabelChanges}
            isSavingLabelChanges={isSavingLabelChanges}
            canSubmit={canSubmit}
            multiLabelEnabled={multiLabelEnabled}
            onToggleMultiLabel={handleToggleProjectMultiLabel}
          />
        </div>

        <footer className="workspace-footer">
          <div className="footer-left">
            <button type="button" className="ghost-button" onClick={handleImport} disabled={isImporting}>
              {isImporting ? "Importing..." : "Import"}
            </button>
            <button type="button" className="ghost-button" onClick={handleExport} disabled={isExporting || !selectedProjectId}>
              {isExporting ? "Exporting..." : "Export Dataset"}
            </button>
            <button
              type="button"
              className="ghost-button danger-button"
              onClick={handleDeleteCurrentAsset}
              disabled={isDeletingAssets || !selectedProjectId || !currentAsset}
            >
              {isDeletingAssets ? "Removing..." : "Remove Image"}
            </button>
            <button
              type="button"
              className={bulkDeleteMode ? "ghost-button active-toggle" : "ghost-button"}
              onClick={handleToggleBulkDeleteMode}
              disabled={!selectedProjectId || isDeletingAssets}
            >
              {bulkDeleteMode ? "Exit Multi-delete" : "Multi-delete"}
            </button>
            <button
              type="button"
              className="ghost-button danger-button"
              onClick={handleDeleteSelectedAssets}
              disabled={isDeletingAssets || selectedDeleteAssetIds.length === 0}
            >
              {isDeletingAssets ? "Removing..." : `Delete Selected (${selectedDeleteAssetIds.length})`}
            </button>
            <button
              type="button"
              className="ghost-button danger-button"
              onClick={handleDeleteSelectedFolder}
              disabled={isDeletingAssets || !selectedTreeFolderPath || selectedFolderAssetCount === 0}
            >
              Delete Folder
            </button>
            <button
              type="button"
              className="ghost-button danger-button"
              onClick={handleDeleteCurrentProject}
              disabled={isDeletingProject || !selectedProjectId}
            >
              {isDeletingProject ? "Deleting..." : "Delete Project"}
            </button>
          </div>
        </footer>
      </section>
      {message ? (
        <div className={`status-toast ${messageTone === "error" ? "is-error" : "is-success"}`} role="status" aria-live="polite">
          <span>{message}</span>
          <button type="button" aria-label="Dismiss message" onClick={() => setMessage(null)}>
            x
          </button>
        </div>
      ) : null}
      {importFailures.length > 0 ? (
        <ul className="status-errors">
          {importFailures.map((failure) => (
            <li key={failure}>{failure}</li>
          ))}
        </ul>
      ) : null}
      {importDialog.open ? (
        <div className="import-modal-backdrop">
          <div className="import-modal">
            <h3>Import Images</h3>
            <div className="import-mode-row">
              <label>
                <input
                  type="radio"
                  checked={importMode === "existing"}
                  onChange={() => {
                    setImportMode("existing");
                    setSelectedImportExistingFolder("");
                    setImportFolderName(importDialog.sourceFolderName);
                  }}
                  disabled={projects.length === 0}
                />
                Existing Project
              </label>
              <label>
                <input
                  type="radio"
                  checked={importMode === "new"}
                  onChange={() => {
                    setImportMode("new");
                    setSelectedImportExistingFolder("");
                    setImportFolderName(importDialog.sourceFolderName);
                  }}
                />
                New Project
              </label>
            </div>
            <label className="import-field">
              <span>Project</span>
              {importMode === "new" ? (
                <input value={importNewProjectName} onChange={(event) => setImportNewProjectName(event.target.value)} placeholder="Project name" />
              ) : (
                <select
                  value={importExistingProjectId}
                  onChange={(event) => {
                    setImportExistingProjectId(event.target.value);
                    setSelectedImportExistingFolder("");
                    setImportFolderName(importDialog.sourceFolderName);
                  }}
                >
                  <option value="">Select project</option>
                  {projects.map((project) => (
                    <option key={project.id} value={project.id}>
                      {project.name}
                    </option>
                  ))}
                </select>
              )}
            </label>
            {importMode === "existing" ? (
              <label className="import-field">
                <span>Existing Folder/Subfolder (optional)</span>
                <select
                  value={selectedImportExistingFolder}
                  onChange={(event) => {
                    const value = event.target.value;
                    setSelectedImportExistingFolder(value);
                    if (value) setImportFolderName(value);
                  }}
                >
                  <option value="">Create new / custom</option>
                  {importExistingFolderOptions.map((folderPath) => (
                    <option key={folderPath} value={folderPath}>
                      {folderPath}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            <label className="import-field">
              <span>Folder Name</span>
              <input value={importFolderName} onChange={(event) => setImportFolderName(event.target.value)} placeholder={importDialog.sourceFolderName} />
            </label>
            {isImporting && importProgressView ? (
              <section className="import-progress" aria-live="polite">
                <div className="import-progress-head">
                  <strong>Importing {importProgressView.progressText}</strong>
                  <span>{importProgressView.percent}%</span>
                </div>
                <div className="import-progress-bar">
                  <span style={{ width: `${importProgressView.percent}%` }} />
                </div>
                <div className="import-progress-metrics">
                  <span>{importProgressView.bytesText}</span>
                  <span>{importProgressView.speedText}</span>
                  <span>{importProgressView.fileRateText}</span>
                </div>
                <div className="import-progress-metrics">
                  <span>{importProgressView.uploadedFilesText}</span>
                  <span>{importProgressView.failedFilesText}</span>
                  <span>{importProgressView.remainingFilesText}</span>
                </div>
                <div className="import-progress-metrics">
                  <span>Elapsed: {importProgressView.elapsedText}</span>
                  <span>ETA: {importProgressView.etaText}</span>
                </div>
                {importProgressView.activeFileName ? <p className="import-progress-file">Uploading: {importProgressView.activeFileName}</p> : null}
              </section>
            ) : null}
            <div className="import-modal-actions">
                <button
                  type="button"
                  className="ghost-button"
                  onClick={() => {
                    setSelectedImportExistingFolder("");
                    setImportProgress(null);
                    setImportDialog({ open: false, sourceFolderName: "", files: [] });
                  }}
                  disabled={isImporting}
                >
                Cancel
              </button>
              <button type="button" className="primary-button" onClick={confirmImportFromDialog} disabled={isImporting}>
                {isImporting ? "Importing..." : "Import"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}
