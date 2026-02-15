export interface ImportFileLike {
  name: string;
  type?: string;
  webkitRelativePath?: string;
}

export function isImageCandidate(file: ImportFileLike): boolean;
export function buildTargetRelativePath(file: ImportFileLike, targetFolder: string): string;
