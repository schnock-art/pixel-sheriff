import type { AnnotationStatus, TaskKind } from "../../../lib/api";
import { StatusMultiSelectDropdown, FolderMultiSelectDropdown } from "./DatasetFilterDropdowns";

export function DatasetDraftPanel({
  mode,
  hasSelectedVersion,
  selectedVersionName,
  draftName,
  seed,
  trainRatio,
  valRatio,
  testRatio,
  includeStatuses,
  excludeStatuses,
  includeFolderPaths,
  excludeFolderPaths,
  includeLabeledOnly,
  includeNegativeImages,
  stratify,
  folderPaths,
  selectedTaskKind,
  isPreviewing,
  isCreating,
  onDraftNameChange,
  onSeedChange,
  onTrainRatioChange,
  onValRatioChange,
  onTestRatioChange,
  onIncludeStatusesChange,
  onExcludeStatusesChange,
  onIncludeFolderPathsChange,
  onExcludeFolderPathsChange,
  onIncludeLabeledOnlyChange,
  onIncludeNegativeImagesChange,
  onStratifyChange,
  onPreview,
  onCreate,
  onDiscardDraft,
  onDuplicateAndEdit,
}: {
  mode: "browse" | "draft";
  hasSelectedVersion: boolean;
  selectedVersionName: string;
  draftName: string;
  seed: number;
  trainRatio: number;
  valRatio: number;
  testRatio: number;
  includeStatuses: AnnotationStatus[];
  excludeStatuses: AnnotationStatus[];
  includeFolderPaths: string[];
  excludeFolderPaths: string[];
  includeLabeledOnly: boolean;
  includeNegativeImages: boolean;
  stratify: boolean;
  folderPaths: string[];
  selectedTaskKind: TaskKind | null;
  isPreviewing: boolean;
  isCreating: boolean;
  onDraftNameChange: (value: string) => void;
  onSeedChange: (value: number) => void;
  onTrainRatioChange: (value: number) => void;
  onValRatioChange: (value: number) => void;
  onTestRatioChange: (value: number) => void;
  onIncludeStatusesChange: (value: AnnotationStatus[]) => void;
  onExcludeStatusesChange: (value: AnnotationStatus[]) => void;
  onIncludeFolderPathsChange: (value: string[]) => void;
  onExcludeFolderPathsChange: (value: string[]) => void;
  onIncludeLabeledOnlyChange: (value: boolean) => void;
  onIncludeNegativeImagesChange: (value: boolean) => void;
  onStratifyChange: (value: boolean) => void;
  onPreview: () => void;
  onCreate: () => void;
  onDiscardDraft: () => void;
  onDuplicateAndEdit: () => void;
}) {
  return (
    <>
      {mode === "browse" ? (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 8 }}>
          <strong>Browsing saved dataset version: {selectedVersionName}</strong>
          <button type="button" className="ghost-button" disabled={!hasSelectedVersion} onClick={onDuplicateAndEdit}>
            Duplicate &amp; Edit
          </button>
        </div>
      ) : (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 8 }}>
          <strong>Draft preview - not saved</strong>
          <div style={{ display: "flex", gap: 8 }}>
            <button type="button" className="primary-button" disabled={isCreating} onClick={onCreate}>
              {isCreating ? "Creating..." : "Create Version"}
            </button>
            <button type="button" className="ghost-button" onClick={onDiscardDraft}>
              Discard Draft
            </button>
          </div>
        </div>
      )}

      <h3>Create Version</h3>
      <div style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(2, minmax(0, 1fr))" }}>
        <label className="project-field">
          <span>Name</span>
          <input value={draftName} onChange={(event) => onDraftNameChange(event.target.value)} />
        </label>
        <label className="project-field">
          <span>Seed</span>
          <input type="number" value={seed} onChange={(event) => onSeedChange(Number(event.target.value) || 1337)} />
        </label>
        <label className="project-field">
          <span>Train Ratio</span>
          <input type="number" step="0.01" value={trainRatio} onChange={(event) => onTrainRatioChange(Number(event.target.value) || 0)} />
        </label>
        <label className="project-field">
          <span>Val Ratio</span>
          <input type="number" step="0.01" value={valRatio} onChange={(event) => onValRatioChange(Number(event.target.value) || 0)} />
        </label>
        <label className="project-field">
          <span>Test Ratio</span>
          <input type="number" step="0.01" value={testRatio} onChange={(event) => onTestRatioChange(Number(event.target.value) || 0)} />
        </label>
      </div>

      <div style={{ display: "grid", gap: 8, marginTop: 8, gridTemplateColumns: "repeat(2, minmax(0, 1fr))" }}>
        <StatusMultiSelectDropdown
          label="Include statuses"
          selected={includeStatuses}
          otherSelected={excludeStatuses}
          onSelectedChange={onIncludeStatusesChange}
          onOtherSelectedChange={onExcludeStatusesChange}
        />
        <StatusMultiSelectDropdown
          label="Exclude statuses"
          selected={excludeStatuses}
          otherSelected={includeStatuses}
          onSelectedChange={onExcludeStatusesChange}
          onOtherSelectedChange={onIncludeStatusesChange}
        />
        <FolderMultiSelectDropdown
          label="Include folders"
          folderPaths={folderPaths}
          selectedPaths={includeFolderPaths}
          opposingSelectedPaths={excludeFolderPaths}
          onSelectedChange={onIncludeFolderPathsChange}
          onOpposingChange={onExcludeFolderPathsChange}
        />
        <FolderMultiSelectDropdown
          label="Exclude folders"
          folderPaths={folderPaths}
          selectedPaths={excludeFolderPaths}
          opposingSelectedPaths={includeFolderPaths}
          onSelectedChange={onExcludeFolderPathsChange}
          onOpposingChange={onIncludeFolderPathsChange}
        />
      </div>

      <p style={{ marginTop: 6, marginBottom: 0, fontSize: 12, color: "var(--muted, #6f7b8a)" }}>
        Exclude statuses: {excludeStatuses.length === 0 ? "none" : excludeStatuses.join(", ")}
      </p>

      <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
        <label className="model-builder-checkbox">
          <input type="checkbox" checked={includeLabeledOnly} onChange={(event) => onIncludeLabeledOnlyChange(event.target.checked)} />
          <span>Labeled only</span>
        </label>
        {selectedTaskKind === "bbox" || selectedTaskKind === "segmentation" ? (
          <label className="model-builder-checkbox">
            <input type="checkbox" checked={includeNegativeImages} onChange={(event) => onIncludeNegativeImagesChange(event.target.checked)} />
            <span>Include images with no objects</span>
          </label>
        ) : null}
        <label className="model-builder-checkbox">
          <input type="checkbox" checked={stratify} onChange={(event) => onStratifyChange(event.target.checked)} />
          <span>Stratify by primary label</span>
        </label>
      </div>
      <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
        <button type="button" className="ghost-button" disabled={isPreviewing} onClick={onPreview}>
          {isPreviewing ? "Previewing..." : "Preview"}
        </button>
      </div>
    </>
  );
}
