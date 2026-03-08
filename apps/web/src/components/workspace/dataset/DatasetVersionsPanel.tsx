import type { DatasetVersionSummaryEnvelope, Task } from "../../../lib/api";
import { datasetVersionIdOf } from "../../../lib/workspace/datasetPage";

export function DatasetVersionsPanel({
  versions,
  tasks,
  isLoadingVersions,
  activeDatasetVersionId,
  selectedDatasetVersionId,
  isExporting,
  onSelectVersion,
  onSetActive,
  onExport,
}: {
  versions: DatasetVersionSummaryEnvelope[];
  tasks: Task[];
  isLoadingVersions: boolean;
  activeDatasetVersionId: string | null;
  selectedDatasetVersionId: string | null;
  isExporting: boolean;
  onSelectVersion: (datasetVersionId: string) => void;
  onSetActive: (datasetVersionId: string) => void;
  onExport: () => void;
}) {
  return (
    <section className="placeholder-card" data-testid="dataset-versions-panel">
      <h3>Versions</h3>
      <div style={{ display: "grid", gap: 8 }}>
        {isLoadingVersions ? <p>Loading versions...</p> : null}
        {!isLoadingVersions && versions.length === 0 ? <p>No dataset versions yet.</p> : null}
        {versions.map((item) => {
          const id = datasetVersionIdOf(item);
          const isActive = activeDatasetVersionId === id;
          const isSelected = selectedDatasetVersionId === id;
          const name = typeof item.version?.name === "string" ? item.version.name : id;
          const versionTaskId = typeof item.version?.task_id === "string" ? item.version.task_id : "";
          const versionTaskName = tasks.find((task) => task.id === versionTaskId)?.name;
          const versionTaskKind = typeof item.version?.task === "string" ? item.version.task : "";
          return (
            <button
              key={id}
              type="button"
              className={isSelected ? "ghost-button active-toggle" : "ghost-button"}
              onClick={() => onSelectVersion(id)}
              style={{ justifyContent: "space-between", display: "flex", alignItems: "center" }}
              data-testid="dataset-version-item"
              data-dataset-version-id={id}
            >
              <span>
                {name}
                {versionTaskKind ? ` [${versionTaskName ?? versionTaskKind}]` : ""}
              </span>
              <span>{isActive ? "active" : item.is_archived ? "archived" : ""}</span>
            </button>
          );
        })}
      </div>
      {selectedDatasetVersionId ? (
        <div style={{ marginTop: 12, display: "grid", gap: 8 }}>
          <button
            type="button"
            className="ghost-button"
            onClick={() => onSetActive(selectedDatasetVersionId)}
            data-testid="dataset-set-active-button"
          >
            Set Active
          </button>
          <button
            type="button"
            className="primary-button"
            disabled={isExporting}
            onClick={onExport}
            data-testid="dataset-export-button"
          >
            {isExporting ? "Exporting..." : "Export Dataset Zip"}
          </button>
        </div>
      ) : null}
    </section>
  );
}
