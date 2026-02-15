export interface AssetLike {
  id: string;
  uri: string;
  metadata_json: Record<string, unknown>;
}

export interface TreeEntry {
  key: string;
  name: string;
  depth: number;
  kind: "folder" | "file";
  path: string;
  assetId?: string;
  folderPath?: string;
}

export interface TreeBuildResult {
  entries: TreeEntry[];
  orderedAssetIds: string[];
  folderAssetIds: Record<string, string[]>;
}

export function asRelativePath(asset: Pick<AssetLike, "uri" | "metadata_json">): string;
export function collectFolderPaths(assets: Array<Pick<AssetLike, "uri" | "metadata_json">>): string[];
export function collectFolderPathsFromRelativePaths(relativePaths: string[]): string[];
export function folderChain(path: string): string[];
export function buildTreeEntries(assets: AssetLike[]): TreeBuildResult;
