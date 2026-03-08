const ALL_STATUSES = ["unlabeled", "labeled", "skipped", "needs_review", "approved"];

function asRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function normalizeStatusList(values) {
  if (!Array.isArray(values)) return [];
  const seen = new Set();
  const out = [];
  for (const value of values) {
    if (typeof value !== "string") continue;
    const normalized = value.trim();
    if (!ALL_STATUSES.includes(normalized)) continue;
    if (seen.has(normalized)) continue;
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

function toggleStatusSelection(params) {
  const { selected, otherSelected, status, checked } = params;
  const selectedSet = new Set(normalizeStatusList(selected));
  const otherSet = new Set(normalizeStatusList(otherSelected));
  if (checked) {
    selectedSet.add(status);
    otherSet.delete(status);
  } else {
    selectedSet.delete(status);
  }
  return {
    selected: Array.from(selectedSet),
    otherSelected: Array.from(otherSet),
  };
}

function buildFolderTree(folderPaths) {
  const sorted = Array.from(new Set((folderPaths || []).filter((value) => typeof value === "string" && value.trim()))).sort((a, b) =>
    a.localeCompare(b),
  );
  const root = { name: "", path: "", childrenMap: new Map() };
  for (const folderPath of sorted) {
    const parts = folderPath.split("/").filter(Boolean);
    let cursor = root;
    let prefix = "";
    for (const part of parts) {
      prefix = prefix ? `${prefix}/${part}` : part;
      if (!cursor.childrenMap.has(part)) {
        cursor.childrenMap.set(part, { name: part, path: prefix, childrenMap: new Map() });
      }
      cursor = cursor.childrenMap.get(part);
    }
  }

  function materialize(node) {
    const children = Array.from(node.childrenMap.values())
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((child) => materialize(child));
    return { name: node.name, path: node.path, children };
  }

  return Array.from(root.childrenMap.values())
    .sort((a, b) => a.name.localeCompare(b.name))
    .map((child) => materialize(child));
}

function buildDescendantsByPath(folderPaths) {
  const sorted = Array.from(new Set((folderPaths || []).filter((value) => typeof value === "string" && value.trim())));
  const output = {};
  for (const path of sorted) {
    output[path] = sorted.filter((candidate) => candidate === path || candidate.startsWith(`${path}/`));
  }
  return output;
}

function folderCheckState(path, selectedPaths, descendantsByPath) {
  const selectedSet = new Set(selectedPaths || []);
  const affected = descendantsByPath[path] || [path];
  const count = affected.filter((item) => selectedSet.has(item)).length;
  if (count === 0) return "unchecked";
  if (count === affected.length) return "checked";
  return "indeterminate";
}

function toggleFolderPathSelection(params) {
  const { selectedPaths, opposingSelectedPaths, path, checked, descendantsByPath } = params;
  const selected = new Set((selectedPaths || []).filter((value) => typeof value === "string"));
  const opposing = new Set((opposingSelectedPaths || []).filter((value) => typeof value === "string"));
  const affected = descendantsByPath[path] || [path];

  if (checked) {
    for (const item of affected) selected.add(item);
    // Keep include/exclude rules explicit: exact same path cannot remain in both sets.
    for (const item of affected) opposing.delete(item);
  } else {
    for (const item of affected) selected.delete(item);
  }

  return {
    selectedPaths: Array.from(selected).sort((a, b) => a.localeCompare(b)),
    opposingSelectedPaths: Array.from(opposing).sort((a, b) => a.localeCompare(b)),
  };
}

function contentUrlForAsset(assetId) {
  return `/api/v1/assets/${assetId}/content`;
}

function datasetVersionIdOf(item) {
  const version = item && typeof item === "object" ? item.version : null;
  const value = version && typeof version === "object" ? version.dataset_version_id : null;
  return typeof value === "string" ? value : "";
}

function summaryFromVersion(versionEnvelope) {
  if (!versionEnvelope) return null;
  const stats = asRecord(asRecord(versionEnvelope.version).stats);
  const splitCounts = asRecord(stats.split_counts);
  const warnings = Array.isArray(stats.warnings) ? stats.warnings.filter((item) => typeof item === "string") : [];
  const classCountsRaw = asRecord(stats.class_counts);
  const classCounts = {};
  for (const [key, value] of Object.entries(classCountsRaw)) {
    if (typeof value === "number" && Number.isFinite(value)) classCounts[key] = value;
  }

  const total =
    typeof stats.asset_count === "number"
      ? stats.asset_count
      : Object.values(splitCounts).reduce((sum, value) => sum + (typeof value === "number" ? value : 0), 0);
  return {
    total,
    class_counts: classCounts,
    split_counts: {
      train: typeof splitCounts.train === "number" ? splitCounts.train : 0,
      val: typeof splitCounts.val === "number" ? splitCounts.val : 0,
      test: typeof splitCounts.test === "number" ? splitCounts.test : 0,
    },
    warnings,
  };
}

function selectedVersionName(item) {
  if (!item) return "(none)";
  const version = asRecord(item.version);
  const name = version.name;
  const id = version.dataset_version_id;
  if (typeof name === "string" && name.trim()) return name;
  if (typeof id === "string" && id.trim()) return id;
  return "(unnamed version)";
}

function classNamesFromVersion(versionEnvelope) {
  if (!versionEnvelope) return {};
  const version = asRecord(versionEnvelope.version);
  const labelSchema = asRecord(asRecord(asRecord(version.labels).label_schema));
  const classes = Array.isArray(labelSchema.classes) ? labelSchema.classes : [];
  const mapping = {};
  for (const value of classes) {
    const row = asRecord(value);
    const categoryId = row.category_id;
    const name = row.name;
    if (typeof categoryId === "string" && categoryId.trim() && typeof name === "string" && name.trim()) {
      mapping[categoryId] = name;
    }
  }
  return mapping;
}

function fallbackClassName(classId) {
  if (/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(classId)) {
    return `Deleted category (${classId.slice(0, 8)})`;
  }
  return classId;
}

function classDisplayName(classId, sources) {
  if (classId === "__missing__") return "Unlabeled / missing primary";
  return sources.summaryClassNames[classId] || sources.versionClassNames[classId] || sources.categoryNameById[classId] || fallbackClassName(classId);
}

function previewAssetPrimaryCategoryId(item) {
  if (!item || typeof item !== "object") return null;
  const labelSummary = item.label_summary;
  if (!labelSummary || typeof labelSummary !== "object" || Array.isArray(labelSummary)) return null;
  const primaryCategoryId = labelSummary.primary_category_id;
  return typeof primaryCategoryId === "string" && primaryCategoryId.trim() ? primaryCategoryId : null;
}

function filterPreviewAssets(items, filters) {
  const rows = Array.isArray(items) ? items : [];
  const splitFilter = filters && typeof filters.splitFilter === "string" ? filters.splitFilter : "all";
  const statusFilter = filters && typeof filters.statusFilter === "string" ? filters.statusFilter : "all";
  const classFilter = filters && typeof filters.classFilter === "string" ? filters.classFilter : "all";
  const searchText = filters && typeof filters.searchText === "string" ? filters.searchText.trim().toLowerCase() : "";

  return rows.filter((item) => {
    if (!item || typeof item !== "object") return false;
    if (splitFilter !== "all" && item.split !== splitFilter) return false;
    if (statusFilter !== "all" && item.status !== statusFilter) return false;
    if (classFilter !== "all" && previewAssetPrimaryCategoryId(item) !== classFilter) return false;
    if (!searchText) return true;

    const filename = typeof item.filename === "string" ? item.filename.toLowerCase() : "";
    const relativePath = typeof item.relative_path === "string" ? item.relative_path.toLowerCase() : "";
    const assetId = typeof item.asset_id === "string" ? item.asset_id.toLowerCase() : "";
    return filename.includes(searchText) || relativePath.includes(searchText) || assetId.includes(searchText);
  });
}

/**
 * Resolves which asset section to render in the dataset panel.
 * - "browse"  → show paginated assetsPayload (saved version selected)
 * - "prompt"  → show a "run preview first" message
 * - "empty"   → show a "no assets matched" message
 * - "samples" → show thumbnails for sample_asset_ids from the preview response
 */
function resolvePreviewAssetsSection(mode, previewSummary) {
  if (mode !== "draft") return { kind: "browse" };
  if (!previewSummary) return { kind: "prompt" };
  if (!previewSummary.sample_asset_ids || previewSummary.sample_asset_ids.length === 0) return { kind: "empty" };
  return { kind: "samples", assetIds: previewSummary.sample_asset_ids };
}

module.exports = {
  ALL_STATUSES,
  asRecord,
  normalizeStatusList,
  toggleStatusSelection,
  buildFolderTree,
  buildDescendantsByPath,
  folderCheckState,
  toggleFolderPathSelection,
  contentUrlForAsset,
  datasetVersionIdOf,
  summaryFromVersion,
  selectedVersionName,
  classNamesFromVersion,
  fallbackClassName,
  classDisplayName,
  previewAssetPrimaryCategoryId,
  filterPreviewAssets,
  resolvePreviewAssetsSection,
};
