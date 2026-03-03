interface ImportValidationView {
  canSubmit: boolean;
  filesError: string | null;
  projectError: string | null;
  folderError: string | null;
}

interface ImportProgressView {
  percent: number;
  elapsedText: string;
  etaText: string;
  fileRateText: string;
  speedText: string;
  progressText: string;
  bytesText: string;
  remainingFilesText: string;
  uploadedFilesText: string;
  failedFilesText: string;
  activeFileName: string | null;
}

interface ProjectOption {
  id: string;
  name: string;
}

type ImportMode = "existing" | "new";
type NewProjectTaskType = "classification_single" | "bbox" | "segmentation";

interface ProjectAssetsImportModalProps {
  open: boolean;
  filesCount: number;
  isImporting: boolean;
  projects: ProjectOption[];
  importMode: ImportMode;
  importNewProjectName: string;
  importExistingProjectId: string;
  importExistingFolderOptions: string[];
  selectedImportExistingFolder: string;
  importFolderName: string;
  importSourceFolderName: string;
  importNewProjectTaskType: NewProjectTaskType;
  importValidation: ImportValidationView;
  importProgressView: ImportProgressView | null;
  onSetImportModeWithDefaults: (mode: ImportMode) => void;
  onSetImportNewProjectName: (value: string) => void;
  onSetImportExistingProjectWithDefaults: (projectId: string) => void;
  onSetImportNewProjectTaskType: (taskType: NewProjectTaskType) => void;
  onSetImportExistingFolderWithDefaults: (folderPath: string) => void;
  onSetImportFolderName: (value: string) => void;
  onClose: () => void;
  onConfirm: () => void;
}

export function ProjectAssetsImportModal({
  open,
  filesCount,
  isImporting,
  projects,
  importMode,
  importNewProjectName,
  importExistingProjectId,
  importExistingFolderOptions,
  selectedImportExistingFolder,
  importFolderName,
  importSourceFolderName,
  importNewProjectTaskType,
  importValidation,
  importProgressView,
  onSetImportModeWithDefaults,
  onSetImportNewProjectName,
  onSetImportExistingProjectWithDefaults,
  onSetImportNewProjectTaskType,
  onSetImportExistingFolderWithDefaults,
  onSetImportFolderName,
  onClose,
  onConfirm,
}: ProjectAssetsImportModalProps) {
  if (!open) return null;

  return (
    <div className="import-modal-backdrop">
      <div className="import-modal">
        <h3>Import Images</h3>
        <p className="import-selection-summary">
          {filesCount} file{filesCount === 1 ? "" : "s"} selected
        </p>
        <div className="import-mode-row">
          <label>
            <input
              type="radio"
              checked={importMode === "existing"}
              onChange={() => onSetImportModeWithDefaults("existing")}
              disabled={projects.length === 0}
            />
            Existing Project
          </label>
          <label>
            <input
              type="radio"
              checked={importMode === "new"}
              onChange={() => onSetImportModeWithDefaults("new")}
            />
            New Project
          </label>
        </div>
        <label className="import-field">
          <span>Project</span>
          {importMode === "new" ? (
            <input
              value={importNewProjectName}
              onChange={(event) => onSetImportNewProjectName(event.target.value)}
              placeholder="Project name"
              aria-invalid={Boolean(importValidation.projectError)}
            />
          ) : (
            <select
              value={importExistingProjectId}
              onChange={(event) => onSetImportExistingProjectWithDefaults(event.target.value)}
              aria-invalid={Boolean(importValidation.projectError)}
            >
              <option value="">Select project</option>
              {projects.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </select>
          )}
          {importValidation.projectError ? (
            <span className="import-field-error">{importValidation.projectError}</span>
          ) : (
            <span className="import-field-hint">
              {importMode === "new" ? "Create a new project for this import." : "Choose the project to receive these files."}
            </span>
          )}
        </label>
        {importMode === "new" ? (
          <label className="import-field">
            <span>Project Task Mode</span>
            <select
              value={importNewProjectTaskType}
              onChange={(event) => onSetImportNewProjectTaskType(event.target.value as NewProjectTaskType)}
            >
              <option value="classification_single">Labels (single-label classification)</option>
              <option value="bbox">Bounding Boxes</option>
              <option value="segmentation">Segmentation</option>
            </select>
            <span className="import-field-hint">This mode is locked per project and controls available annotation tools.</span>
          </label>
        ) : null}
        {importMode === "existing" ? (
          <label className="import-field">
            <span>Existing Folder/Subfolder (optional)</span>
            <select
              value={selectedImportExistingFolder}
              onChange={(event) => onSetImportExistingFolderWithDefaults(event.target.value)}
            >
              <option value="">Create new / custom</option>
              {importExistingFolderOptions.map((folderPath) => (
                <option key={folderPath} value={folderPath}>
                  {folderPath}
                </option>
              ))}
            </select>
            <span className="import-field-hint">Defaults remember your last selected destination folder per project.</span>
          </label>
        ) : null}
        <label className="import-field">
          <span>Folder Name</span>
          <input
            value={importFolderName}
            onChange={(event) => onSetImportFolderName(event.target.value)}
            placeholder={importSourceFolderName}
            aria-invalid={Boolean(importValidation.folderError)}
          />
          {importValidation.folderError ? (
            <span className="import-field-error">{importValidation.folderError}</span>
          ) : (
            <span className="import-field-hint">You can use nested paths like train/cats.</span>
          )}
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
          <button type="button" className="ghost-button" onClick={onClose} disabled={isImporting}>
            Cancel
          </button>
          <button type="button" className="primary-button" onClick={onConfirm} disabled={isImporting || !importValidation.canSubmit}>
            {isImporting ? "Importing..." : "Import"}
          </button>
        </div>
      </div>
    </div>
  );
}
