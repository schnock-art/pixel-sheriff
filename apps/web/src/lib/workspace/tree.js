function asRelativePath(asset) {
  if (typeof asset.relative_path === "string" && asset.relative_path.trim() !== "") return asset.relative_path;
  if (typeof asset.file_name === "string" && asset.file_name.trim() !== "") {
    const folderPath = typeof asset.folder_path === "string" ? asset.folder_path.replaceAll("\\", "/").trim() : "";
    return folderPath ? `${folderPath.replace(/^\/+|\/+$/g, "")}/${asset.file_name}` : asset.file_name;
  }
  const fromMetadata = asset.metadata_json.relative_path;
  if (typeof fromMetadata === "string" && fromMetadata.trim() !== "") return fromMetadata;
  const original = asset.metadata_json.original_filename;
  if (typeof original === "string" && original.trim() !== "") return original;
  return asset.uri;
}

function collectFolderPaths(assets, folders = []) {
  const explicit = folders
    .map((folder) => (typeof folder.path === "string" ? folder.path.replaceAll("\\", "/").trim().replace(/^\/+|\/+$/g, "") : ""))
    .filter(Boolean);
  return collectFolderPathsFromRelativePaths([...assets.map((asset) => asRelativePath(asset)), ...explicit]);
}

function collectFolderPathsFromRelativePaths(relativePaths) {
  const paths = new Set();
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

function folderChain(path) {
  const parts = path.split("/").filter(Boolean);
  const chain = [];
  let prefix = "";
  for (const part of parts) {
    prefix = prefix ? `${prefix}/${part}` : part;
    chain.push(prefix);
  }
  return chain;
}

function buildTreeEntries(assets, folders = []) {
  const root = { name: "", path: "", folders: new Map(), files: [] };

  function ensureFolderNode(path, folder = null) {
    const normalized = String(path || "").replaceAll("\\", "/").replace(/^\/+|\/+$/g, "");
    if (!normalized) return root;
    const segments = normalized.split("/").filter(Boolean);
    let cursor = root;
    let prefix = "";
    for (const part of segments) {
      prefix = prefix ? `${prefix}/${part}` : part;
      const existing = cursor.folders.get(part);
      if (existing) {
        cursor = existing;
      } else {
        const next = {
          name: part,
          path: prefix,
          folderId: null,
          sequenceId: null,
          sequenceStatus: null,
          sequenceSourceType: null,
          sequenceName: null,
          sequenceFrameCount: null,
          folders: new Map(),
          files: [],
        };
        cursor.folders.set(part, next);
        cursor = next;
      }
    }
    if (folder) {
      cursor.folderId = folder.id ?? cursor.folderId ?? null;
      cursor.sequenceId = folder.sequence_id ?? cursor.sequenceId ?? null;
      cursor.sequenceStatus = folder.sequence_status ?? cursor.sequenceStatus ?? null;
      cursor.sequenceSourceType = folder.sequence_source_type ?? cursor.sequenceSourceType ?? null;
      cursor.sequenceName = folder.sequence_name ?? cursor.sequenceName ?? null;
      cursor.sequenceFrameCount = folder.sequence_frame_count ?? cursor.sequenceFrameCount ?? null;
    }
    return cursor;
  }

  const sortedFolders = folders.slice().sort((a, b) => {
    const left = String(a.path || "");
    const right = String(b.path || "");
    return left.localeCompare(right);
  });
  for (const folder of sortedFolders) ensureFolderNode(folder.path, folder);

  for (const asset of assets) {
    const rel = asRelativePath(asset).replaceAll("\\", "/");
    const segments = rel.split("/").filter(Boolean);
    const filename = segments[segments.length - 1] ?? rel;
    const folderParts = segments.slice(0, -1);

    const cursor = ensureFolderNode(folderParts.join("/"));

    cursor.files.push({ id: asset.id, name: filename, path: rel });
  }

  const result = [];
  const orderedAssetIds = [];
  const folderAssetIds = new Map();

  function visit(node, depth) {
    const subtreeAssetIds = [];
    const folders = Array.from(node.folders.values()).sort((a, b) => a.name.localeCompare(b.name));
    for (const folder of folders) {
      result.push({
        key: `folder:${folder.folderId ?? folder.path}`,
        name: folder.name,
        depth,
        kind: "folder",
        path: folder.path,
        folderId: folder.folderId ?? undefined,
        sequenceId: folder.sequenceId ?? undefined,
        sequenceStatus: folder.sequenceStatus ?? undefined,
        sequenceSourceType: folder.sequenceSourceType ?? undefined,
        sequenceName: folder.sequenceName ?? undefined,
        sequenceFrameCount: folder.sequenceFrameCount ?? undefined,
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
        path: file.path,
        assetId: file.id,
        folderPath: node.path,
      });
    }
    return subtreeAssetIds;
  }

  visit(root, 0);
  const folderAssetIdsRecord = {};
  folderAssetIds.forEach((ids, folderPath) => {
    folderAssetIdsRecord[folderPath] = ids;
  });

  return { entries: result, orderedAssetIds, folderAssetIds: folderAssetIdsRecord };
}

module.exports = {
  asRelativePath,
  collectFolderPaths,
  collectFolderPathsFromRelativePaths,
  folderChain,
  buildTreeEntries,
};
