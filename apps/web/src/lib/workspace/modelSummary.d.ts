export interface ModelSummaryView {
  task: string;
  numClasses: number;
  classNames: string[];
  classNamesText: string;
  inputSizeText: string;
  resizePolicy: string;
  normalizationType: string;
  architectureFamily: string;
  backboneName: string;
  neckType: string;
  headType: string;
  primaryOutputFormat: string;
  onnxEnabled: boolean;
  onnxOpset: number;
  dynamicBatch: boolean;
  dynamicHeightWidth: boolean;
}

export function readModelSummary(config: Record<string, unknown> | null | undefined): ModelSummaryView;

