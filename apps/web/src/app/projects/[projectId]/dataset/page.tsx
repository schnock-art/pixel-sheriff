"use client";

import { useMemo } from "react";
import { useSearchParams } from "next/navigation";

import { resolveAssetUri } from "../../../../lib/api";
import { DatasetAssetsPanel } from "../../../../components/workspace/dataset/DatasetAssetsPanel";
import { DatasetDraftPanel } from "../../../../components/workspace/dataset/DatasetDraftPanel";
import { DatasetSummaryPanel } from "../../../../components/workspace/dataset/DatasetSummaryPanel";
import { DatasetVersionsPanel } from "../../../../components/workspace/dataset/DatasetVersionsPanel";
import { useDatasetPageState } from "../../../../lib/hooks/useDatasetPageState";
import { classDisplayName, selectedVersionName } from "../../../../lib/workspace/datasetPage";

interface DatasetPageProps {
  params: {
    projectId: string;
  };
}

export default function DatasetPage({ params }: DatasetPageProps) {
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);
  const searchParams = useSearchParams();
  const requestedTaskId = searchParams.get("taskId");
  const state = useDatasetPageState({ projectId, requestedTaskId });
  const resolveDatasetAssetUri = (assetId: string) => resolveAssetUri(`/api/v1/assets/${assetId}/content`);

  return (
    <main className="workspace-shell project-page-shell">
      <section className="workspace-frame project-content-frame placeholder-page">
        <header className="project-section-header">
          <h2>Dataset</h2>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <label htmlFor="dataset-task-select">Task</label>
            <select
              id="dataset-task-select"
              value={state.selectedTaskId ?? ""}
              onChange={(event) => state.setSelectedTaskId(event.target.value || null)}
              disabled={state.tasks.length === 0}
            >
              {state.tasks.length === 0 ? <option value="">No tasks</option> : null}
              {state.tasks.map((task) => (
                <option key={task.id} value={task.id}>
                  {task.name} [{task.kind}]
                </option>
              ))}
            </select>
            {state.selectedTask ? <span style={{ fontSize: 12, opacity: 0.8 }}>{state.selectedTask.kind}</span> : null}
          </div>
        </header>

        {state.errorMessage ? <p className="project-field-error">{state.errorMessage}</p> : null}

        <div style={{ display: "grid", gap: 12, gridTemplateColumns: "280px 1fr 320px" }}>
          <DatasetVersionsPanel
            versions={state.versions}
            tasks={state.tasks}
            isLoadingVersions={state.isLoadingVersions}
            activeDatasetVersionId={state.activeDatasetVersionId}
            selectedDatasetVersionId={state.browser.selectedDatasetVersionId}
            isExporting={state.isExporting}
            onSelectVersion={(datasetVersionId) => {
              state.browser.setSelectedDatasetVersionId(datasetVersionId);
              state.setMode("browse");
            }}
            onSetActive={(datasetVersionId) => void state.handleSetActive(datasetVersionId)}
            onExport={async () => {
              const exported = await state.handleExport();
              if (!exported || !state.browser.selectedDatasetVersionId) return;
              const url = resolveAssetUri(exported.export_uri);
              const anchor = document.createElement("a");
              anchor.href = url;
              anchor.download = `${state.browser.selectedDatasetVersionId}-${exported.hash.slice(0, 8)}.zip`;
              document.body.appendChild(anchor);
              anchor.click();
              anchor.remove();
            }}
          />

          <section className="placeholder-card">
            <DatasetDraftPanel
              mode={state.mode}
              hasSelectedVersion={Boolean(state.selectedVersion)}
              selectedVersionName={selectedVersionName(state.selectedVersion)}
              draftName={state.draft.draftName}
              seed={state.draft.seed}
              trainRatio={state.draft.trainRatio}
              valRatio={state.draft.valRatio}
              testRatio={state.draft.testRatio}
              includeStatuses={state.draft.includeStatuses}
              excludeStatuses={state.draft.excludeStatuses}
              includeFolderPaths={state.draft.includeFolderPaths}
              excludeFolderPaths={state.draft.excludeFolderPaths}
              includeLabeledOnly={state.draft.includeLabeledOnly}
              includeNegativeImages={state.draft.includeNegativeImages}
              stratify={state.draft.stratify}
              folderPaths={state.folderPaths}
              selectedTaskKind={state.selectedTask?.kind ?? null}
              isPreviewing={state.isPreviewing}
              isCreating={state.isCreating}
              onDraftNameChange={state.draft.setDraftName}
              onSeedChange={state.draft.setSeed}
              onTrainRatioChange={state.draft.setTrainRatio}
              onValRatioChange={state.draft.setValRatio}
              onTestRatioChange={state.draft.setTestRatio}
              onIncludeStatusesChange={state.draft.setIncludeStatuses}
              onExcludeStatusesChange={state.draft.setExcludeStatuses}
              onIncludeFolderPathsChange={state.draft.setIncludeFolderPaths}
              onExcludeFolderPathsChange={state.draft.setExcludeFolderPaths}
              onIncludeLabeledOnlyChange={state.draft.setIncludeLabeledOnly}
              onIncludeNegativeImagesChange={state.draft.setIncludeNegativeImages}
              onStratifyChange={state.draft.setStratify}
              onPreview={() => void state.handlePreview()}
              onCreate={() => void state.handleCreate()}
              onDiscardDraft={state.handleDiscardDraft}
              onDuplicateAndEdit={state.handleDuplicateAndEdit}
            />

            <DatasetAssetsPanel
              mode={state.mode}
              splitFilter={state.browser.splitFilter}
              statusFilter={state.browser.statusFilter}
              classFilter={state.classFilter}
              searchText={state.browser.searchText}
              viewMode={state.browser.viewMode}
              classFilterOptions={state.classFilterOptions}
              isLoadingAssets={state.isLoadingAssets}
              assetsPayload={state.assetsPayload}
              previewSummary={state.draft.previewSummary}
              filteredPreviewAssets={state.filteredPreviewAssets}
              resolveImageUrl={resolveDatasetAssetUri}
              onSplitFilterChange={state.browser.setSplitFilter}
              onStatusFilterChange={state.browser.setStatusFilter}
              onClassFilterChange={(value) => {
                state.setClassFilter(value);
                state.browser.setPage(1);
              }}
              onSearchTextChange={state.browser.setSearchText}
              onViewModeChange={state.browser.setViewMode}
              onPreviousPage={() => state.browser.setPage((value) => Math.max(1, value - 1))}
              onNextPage={() => state.browser.setPage((value) => value + 1)}
            />
          </section>

          <DatasetSummaryPanel
            summarySource={state.summarySource}
            summaryData={state.summaryData}
            classDisplayName={(classId) =>
              classDisplayName(classId, {
                summaryClassNames: state.summaryClassNames,
                versionClassNames: state.versionClassNames,
                categoryNameById: state.categoryNameById,
              })
            }
          />
        </div>
      </section>
    </main>
  );
}
