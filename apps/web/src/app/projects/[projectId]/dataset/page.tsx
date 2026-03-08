"use client";

import Link from "next/link";
import { useMemo } from "react";

import { resolveAssetUri } from "../../../../lib/api";
import { DatasetAssetsPanel } from "../../../../components/workspace/dataset/DatasetAssetsPanel";
import { DatasetDraftPanel } from "../../../../components/workspace/dataset/DatasetDraftPanel";
import { DatasetSummaryPanel } from "../../../../components/workspace/dataset/DatasetSummaryPanel";
import { DatasetVersionsPanel } from "../../../../components/workspace/dataset/DatasetVersionsPanel";
import { ProjectSectionLayout } from "../../../../components/workspace/project-shell/ProjectSectionLayout";
import { useProjectShell } from "../../../../components/workspace/project-shell/ProjectShellContext";
import { useDatasetPageState } from "../../../../lib/hooks/useDatasetPageState";
import { classDisplayName, selectedVersionName } from "../../../../lib/workspace/datasetPage";
import { buildModelCreateHref } from "../../../../lib/workspace/projectRouting";

interface DatasetPageProps {
  params: {
    projectId: string;
  };
}

function summaryValue(value: number | null | undefined) {
  return typeof value === "number" ? value : "-";
}

export default function DatasetPage({ params }: DatasetPageProps) {
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);
  const { selectedTaskId, selectedTask, tasks } = useProjectShell();
  const state = useDatasetPageState({
    projectId,
    selectedTaskId,
    selectedTaskKind: selectedTask?.kind ?? null,
  });
  const resolveDatasetAssetUri = (assetId: string) => resolveAssetUri(`/api/v1/assets/${assetId}/content`);
  const trainModelHref = buildModelCreateHref(projectId, {
    taskId: selectedTaskId,
    datasetVersionId: state.browser.selectedDatasetVersionId,
  });
  const summary = state.summaryData;
  const selectedVersionLabel = selectedVersionName(state.selectedVersion);

  return (
    <ProjectSectionLayout
      title="Dataset"
      description="Turn labeled assets into versioned datasets, inspect splits, export packages, and move cleanly into model creation."
      actions={
        <>
          <button
            type="button"
            className="ghost-button"
            onClick={async () => {
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
            disabled={!state.browser.selectedDatasetVersionId || state.isExporting}
          >
            {state.isExporting ? "Exporting..." : "Export Dataset Zip"}
          </button>
          <Link
            className={`primary-button${state.browser.selectedDatasetVersionId ? "" : " disabled-link"}`}
            href={state.browser.selectedDatasetVersionId ? trainModelHref : "#"}
            aria-disabled={!state.browser.selectedDatasetVersionId}
            data-testid="dataset-train-model-button"
            onClick={(event) => {
              if (!state.browser.selectedDatasetVersionId) event.preventDefault();
            }}
          >
            Train Model
          </Link>
        </>
      }
    >
      {state.errorMessage ? <p className="project-field-error">{state.errorMessage}</p> : null}

      <section className="dataset-summary-header placeholder-card">
        <div>
          <p className="dataset-summary-kicker">Current dataset version</p>
          <h3>{selectedVersionLabel}</h3>
          <p className="dataset-summary-meta">
            {selectedTask ? `${selectedTask.name} [${selectedTask.kind}]` : "No task selected"}
            {state.activeDatasetVersionId && state.browser.selectedDatasetVersionId === state.activeDatasetVersionId ? (
              <span className="dataset-active-badge">Active</span>
            ) : null}
          </p>
        </div>
        <dl className="dataset-summary-stats">
          <div>
            <dt>Total images</dt>
            <dd>{summaryValue(summary?.total)}</dd>
          </div>
          <div>
            <dt>Train</dt>
            <dd>{summaryValue(summary?.split_counts?.train)}</dd>
          </div>
          <div>
            <dt>Val</dt>
            <dd>{summaryValue(summary?.split_counts?.val)}</dd>
          </div>
          <div>
            <dt>Test</dt>
            <dd>{summaryValue(summary?.split_counts?.test)}</dd>
          </div>
          <div>
            <dt>Classes</dt>
            <dd>{summaryValue(summary ? Object.keys(summary.class_counts ?? {}).length : null)}</dd>
          </div>
        </dl>
      </section>

      <div className="dataset-workspace-grid">
        <DatasetVersionsPanel
          versions={state.versions}
          tasks={tasks}
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

        <section className="placeholder-card dataset-main-panel">
          <DatasetDraftPanel
            mode={state.mode}
            hasSelectedVersion={Boolean(state.selectedVersion)}
            selectedVersionName={selectedVersionLabel}
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
            selectedTaskKind={selectedTask?.kind ?? null}
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
    </ProjectSectionLayout>
  );
}
