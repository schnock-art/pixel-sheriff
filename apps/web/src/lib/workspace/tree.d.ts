export interface AssetLike {
  id: string;
  uri: string;
  relative_path?: string | null;
  folder_path?: string | null;
  file_name?: string | null;
  metadata_json: Record<string, unknown>;
}

export interface FolderLike {
  id?: string;
  path: string;
  sequence_id?: string | null;
  sequence_status?: string | null;
  sequence_source_type?: string | null;
  sequence_name?: string | null;
  sequence_frame_count?: number | null;
}

export interface TreeEntry {
  key: string;
  name: string;
  depth: number;
  kind: "folder" | "file";
  path: string;
  folderId?: string;
  sequenceId?: string;
  sequenceStatus?: string;
  sequenceSourceType?: string;
  sequenceName?: string;
  sequenceFrameCount?: number;
  assetId?: string;
  folderPath?: string;
}

export interface TreeBuildResult {
  entries: TreeEntry[];
  orderedAssetIds: string[];
  folderAssetIds: Record<string, string[]>;
}

export function asRelativePath(asset: Pick<AssetLike, "uri" | "metadata_json">): string;
export function collectFolderPaths(assets: Array<Pick<AssetLike, "uri" | "metadata_json" | "relative_path" | "folder_path" | "file_name">>, folders?: FolderLike[]): string[];
export function collectFolderPathsFromRelativePaths(relativePaths: string[]): string[];
export function folderChain(path: string): string[];
export function buildTreeEntries(assets: AssetLike[], folders?: FolderLike[]): TreeBuildResult;
