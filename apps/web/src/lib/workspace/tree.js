function asRelativePath(asset) {
  const fromMetadata = asset.metadata_json.relative_path;
  if (typeof fromMetadata === "string" && fromMetadata.trim() !== "") return fromMetadata;
  const original = asset.metadata_json.original_filename;
  if (typeof original === "string" && original.trim() !== "") return original;
  return asset.uri;
}

function collectFolderPaths(assets) {
  return collectFolderPathsFromRelativePaths(assets.map((asset) => asRelativePath(asset)));
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

function buildTreeEntries(assets) {
  const root = { name: "", path: "", folders: new Map(), files: [] };

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
        const next = { name: part, path: prefix, folders: new Map(), files: [] };
        cursor.folders.set(part, next);
        cursor = next;
      }
    }

    cursor.files.push({ id: asset.id, name: filename });
  }

  const result = [];
  const orderedAssetIds = [];
  const folderAssetIds = new Map();

  function visit(node, depth) {
    const subtreeAssetIds = [];
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
