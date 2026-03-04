const ALL_STATUSES = ["unlabeled", "labeled", "skipped", "needs_review", "approved"];

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

module.exports = {
  ALL_STATUSES,
  normalizeStatusList,
  toggleStatusSelection,
  buildFolderTree,
  buildDescendantsByPath,
  folderCheckState,
  toggleFolderPathSelection,
  contentUrlForAsset,
};
