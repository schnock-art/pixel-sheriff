"use client";

import { useEffect, useMemo, useState } from "react";

import {
  ApiError,
  createDatasetVersion,
  exportDatasetVersion,
  listCategories,
  listDatasetVersionAssets,
  listDatasetVersions,
  previewDatasetVersion,
  resolveAssetUri,
  setActiveDatasetVersion,
  type AnnotationStatus,
  type DatasetVersionAssetsPayload,
  type DatasetVersionSummaryEnvelope,
} from "../../../../lib/api";

interface DatasetPageProps {
  params: {
    projectId: string;
  };
}

function parseApiErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof ApiError && error.responseBody) {
    try {
      const parsed = JSON.parse(error.responseBody) as {
        error?: { message?: string; details?: { issues?: Array<{ path?: string; message?: string }> } };
      };
      const issue = parsed.error?.details?.issues?.[0];
      if (issue?.path && issue?.message) {
        return `${parsed.error?.message ?? fallback} (${issue.path}: ${issue.message})`;
      }
      if (parsed.error?.message) return parsed.error.message;
      return error.responseBody;
    } catch {
      return error.responseBody;
    }
  }
  if (error instanceof Error) return error.message;
  return fallback;
}

function versionIdOf(item: DatasetVersionSummaryEnvelope): string {
  const version = item.version;
  const value = version?.dataset_version_id;
  return typeof value === "string" ? value : "";
}

export default function DatasetPage({ params }: DatasetPageProps) {
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);
  const [versions, setVersions] = useState<DatasetVersionSummaryEnvelope[]>([]);
  const [activeDatasetVersionId, setActiveDatasetVersionIdState] = useState<string | null>(null);
  const [selectedDatasetVersionId, setSelectedDatasetVersionId] = useState<string | null>(null);
  const [assetsPayload, setAssetsPayload] = useState<DatasetVersionAssetsPayload | null>(null);
  const [isLoadingVersions, setIsLoadingVersions] = useState(true);
  const [isLoadingAssets, setIsLoadingAssets] = useState(false);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [previewSummary, setPreviewSummary] = useState<{
    total: number;
    class_counts: Record<string, number>;
    split_counts: { train: number; val: number; test: number };
    warnings: string[];
    sample_asset_ids: string[];
  } | null>(null);
  const [categoryNameById, setCategoryNameById] = useState<Record<string, string>>({});

  const [draftName, setDraftName] = useState("Dataset v1");
  const [includeLabeledOnly, setIncludeLabeledOnly] = useState(true);
  const [statusSelection, setStatusSelection] = useState<AnnotationStatus[]>(["labeled", "approved", "needs_review"]);
  const [includeFolderPaths, setIncludeFolderPaths] = useState("");
  const [excludeFolderPaths, setExcludeFolderPaths] = useState("");
  const [seed, setSeed] = useState(1337);
  const [trainRatio, setTrainRatio] = useState(0.8);
  const [valRatio, setValRatio] = useState(0.1);
  const [testRatio, setTestRatio] = useState(0.1);
  const [stratify, setStratify] = useState(true);
  const [splitFilter, setSplitFilter] = useState<"all" | "train" | "val" | "test">("all");
  const [statusFilter, setStatusFilter] = useState<"all" | AnnotationStatus>("all");
  const [searchText, setSearchText] = useState("");
  const [page, setPage] = useState(1);

  async function loadVersions() {
    setIsLoadingVersions(true);
    try {
      const payload = await listDatasetVersions(projectId);
      setVersions(payload.items ?? []);
      setActiveDatasetVersionIdState(payload.active_dataset_version_id ?? null);
      const firstId = payload.active_dataset_version_id ?? versionIdOf(payload.items?.[0] ?? { version: {}, is_active: false, is_archived: false });
      setSelectedDatasetVersionId(firstId || null);
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(parseApiErrorMessage(error, "Failed to load dataset versions"));
      setVersions([]);
      setSelectedDatasetVersionId(null);
    } finally {
      setIsLoadingVersions(false);
    }
  }

  useEffect(() => {
    void loadVersions();
  }, [projectId]);

  useEffect(() => {
    let isMounted = true;
    async function loadCategoryNames() {
      try {
        const categories = await listCategories(projectId);
        if (!isMounted) return;
        const mapping: Record<string, string> = {};
        for (const category of categories) {
          if (typeof category.id === "string" && category.id.trim() && typeof category.name === "string" && category.name.trim()) {
            mapping[category.id] = category.name;
          }
        }
        setCategoryNameById(mapping);
      } catch {
        if (!isMounted) return;
        setCategoryNameById({});
      }
    }
    void loadCategoryNames();
    return () => {
      isMounted = false;
    };
  }, [projectId]);

  function classDisplayName(classId: string): string {
    if (classId === "__missing__") return "Unlabeled / missing primary";
    return categoryNameById[classId] ?? classId;
  }

  useEffect(() => {
    if (!selectedDatasetVersionId) {
      setAssetsPayload(null);
      return;
    }
    let isMounted = true;
    async function loadAssets() {
      setIsLoadingAssets(true);
      try {
        const payload = await listDatasetVersionAssets(projectId, selectedDatasetVersionId, {
          page,
          page_size: 50,
          split: splitFilter === "all" ? undefined : splitFilter,
          status: statusFilter === "all" ? undefined : statusFilter,
          search: searchText.trim() || undefined,
        });
        if (!isMounted) return;
        setAssetsPayload(payload);
      } catch (error) {
        if (!isMounted) return;
        setErrorMessage(parseApiErrorMessage(error, "Failed to load dataset assets"));
      } finally {
        if (isMounted) setIsLoadingAssets(false);
      }
    }
    void loadAssets();
    return () => {
      isMounted = false;
    };
  }, [page, projectId, searchText, selectedDatasetVersionId, splitFilter, statusFilter]);

  function parseFolderPaths(raw: string): string[] {
    return raw
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
  }

  function selectionPayload() {
    return {
      mode: "filter_snapshot" as const,
      filters: {
        include_labeled_only: includeLabeledOnly,
        include_statuses: statusSelection,
        include_folder_paths: parseFolderPaths(includeFolderPaths),
        exclude_folder_paths: parseFolderPaths(excludeFolderPaths),
      },
    };
  }

  function splitPayload() {
    return {
      seed,
      ratios: { train: trainRatio, val: valRatio, test: testRatio },
      stratify: { enabled: stratify, by: "label_primary" as const, strict_stratify: false },
    };
  }

  async function handlePreview() {
    setIsPreviewing(true);
    try {
      const payload = await previewDatasetVersion(projectId, {
        task: "classification",
        selection: selectionPayload(),
        split: splitPayload(),
      });
      setPreviewSummary({
        total: payload.counts.total,
        class_counts: payload.counts.class_counts,
        split_counts: payload.counts.split_counts,
        warnings: payload.warnings ?? [],
        sample_asset_ids: payload.sample_asset_ids ?? [],
      });
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(parseApiErrorMessage(error, "Failed to preview dataset"));
    } finally {
      setIsPreviewing(false);
    }
  }

  async function handleCreate() {
    setIsCreating(true);
    try {
      const created = await createDatasetVersion(projectId, {
        name: draftName.trim() || "Dataset version",
        task: "classification",
        selection: selectionPayload(),
        split: splitPayload(),
        set_active: true,
      });
      const createdId = versionIdOf(created);
      await loadVersions();
      if (createdId) setSelectedDatasetVersionId(createdId);
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(parseApiErrorMessage(error, "Failed to create dataset version"));
    } finally {
      setIsCreating(false);
    }
  }

  async function handleSetActive(datasetVersionId: string) {
    try {
      const response = await setActiveDatasetVersion(projectId, datasetVersionId);
      setActiveDatasetVersionIdState(response.active_dataset_version_id);
      await loadVersions();
    } catch (error) {
      setErrorMessage(parseApiErrorMessage(error, "Failed to set active dataset version"));
    }
  }

  async function handleExport() {
    if (!selectedDatasetVersionId) return;
    setIsExporting(true);
    try {
      const exported = await exportDatasetVersion(projectId, selectedDatasetVersionId);
      const url = resolveAssetUri(exported.export_uri);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${selectedDatasetVersionId}-${exported.hash.slice(0, 8)}.zip`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(parseApiErrorMessage(error, "Failed to export dataset version"));
    } finally {
      setIsExporting(false);
    }
  }

  return (
    <main className="workspace-shell project-page-shell">
      <section className="workspace-frame project-content-frame placeholder-page">
        <header className="project-section-header">
          <h2>Dataset</h2>
        </header>

        {errorMessage ? <p className="project-field-error">{errorMessage}</p> : null}

        <div style={{ display: "grid", gap: 12, gridTemplateColumns: "280px 1fr 320px" }}>
          <section className="placeholder-card">
            <h3>Versions</h3>
            <div style={{ display: "grid", gap: 8 }}>
              {isLoadingVersions ? <p>Loading versions...</p> : null}
              {!isLoadingVersions && versions.length === 0 ? <p>No dataset versions yet.</p> : null}
              {versions.map((item) => {
                const id = versionIdOf(item);
                const isActive = activeDatasetVersionId === id;
                const isSelected = selectedDatasetVersionId === id;
                const name = typeof item.version?.name === "string" ? item.version.name : id;
                return (
                  <button
                    key={id}
                    type="button"
                    className={isSelected ? "ghost-button active-toggle" : "ghost-button"}
                    onClick={() => {
                      setSelectedDatasetVersionId(id);
                      setPage(1);
                    }}
                    style={{ justifyContent: "space-between", display: "flex", alignItems: "center" }}
                  >
                    <span>{name}</span>
                    <span>{isActive ? "active" : item.is_archived ? "archived" : ""}</span>
                  </button>
                );
              })}
            </div>
            {selectedDatasetVersionId ? (
              <div style={{ marginTop: 12, display: "grid", gap: 8 }}>
                <button type="button" className="ghost-button" onClick={() => void handleSetActive(selectedDatasetVersionId)}>
                  Set Active
                </button>
                <button type="button" className="primary-button" disabled={isExporting} onClick={() => void handleExport()}>
                  {isExporting ? "Exporting..." : "Export Dataset Zip"}
                </button>
              </div>
            ) : null}
          </section>

          <section className="placeholder-card">
            <h3>Create Version</h3>
            <div style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(2, minmax(0, 1fr))" }}>
              <label className="project-field">
                <span>Name</span>
                <input value={draftName} onChange={(event) => setDraftName(event.target.value)} />
              </label>
              <label className="project-field">
                <span>Seed</span>
                <input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value) || 1337)} />
              </label>
              <label className="project-field">
                <span>Train Ratio</span>
                <input type="number" step="0.01" value={trainRatio} onChange={(event) => setTrainRatio(Number(event.target.value) || 0)} />
              </label>
              <label className="project-field">
                <span>Val Ratio</span>
                <input type="number" step="0.01" value={valRatio} onChange={(event) => setValRatio(Number(event.target.value) || 0)} />
              </label>
              <label className="project-field">
                <span>Test Ratio</span>
                <input type="number" step="0.01" value={testRatio} onChange={(event) => setTestRatio(Number(event.target.value) || 0)} />
              </label>
              <label className="project-field">
                <span>Statuses (comma)</span>
                <input
                  value={statusSelection.join(",")}
                  onChange={(event) =>
                    setStatusSelection(
                      event.target.value
                        .split(",")
                        .map((value) => value.trim())
                        .filter((value): value is AnnotationStatus =>
                          ["unlabeled", "labeled", "skipped", "needs_review", "approved"].includes(value),
                        ),
                    )
                  }
                />
              </label>
              <label className="project-field">
                <span>Include folders (comma)</span>
                <input value={includeFolderPaths} onChange={(event) => setIncludeFolderPaths(event.target.value)} />
              </label>
              <label className="project-field">
                <span>Exclude folders (comma)</span>
                <input value={excludeFolderPaths} onChange={(event) => setExcludeFolderPaths(event.target.value)} />
              </label>
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
              <label className="model-builder-checkbox">
                <input type="checkbox" checked={includeLabeledOnly} onChange={(event) => setIncludeLabeledOnly(event.target.checked)} />
                <span>Labeled only</span>
              </label>
              <label className="model-builder-checkbox">
                <input type="checkbox" checked={stratify} onChange={(event) => setStratify(event.target.checked)} />
                <span>Stratify by primary label</span>
              </label>
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
              <button type="button" className="ghost-button" disabled={isPreviewing} onClick={() => void handlePreview()}>
                {isPreviewing ? "Previewing..." : "Preview"}
              </button>
              <button type="button" className="primary-button" disabled={isCreating} onClick={() => void handleCreate()}>
                {isCreating ? "Creating..." : "Create Version"}
              </button>
            </div>

            <h3 style={{ marginTop: 16 }}>Preview Assets</h3>
            {!selectedDatasetVersionId && previewSummary ? (
              <p style={{ marginBottom: 8, fontSize: 13 }}>
                Preview updated: {previewSummary.total} assets selected
                {previewSummary.sample_asset_ids.length > 0 ? ` (sample ${previewSummary.sample_asset_ids.length})` : ""}.
                Create a dataset version to browse paginated assets.
              </p>
            ) : null}
            <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
              <select value={splitFilter} onChange={(event) => setSplitFilter(event.target.value as "all" | "train" | "val" | "test")}>
                <option value="all">All splits</option>
                <option value="train">Train</option>
                <option value="val">Val</option>
                <option value="test">Test</option>
              </select>
              <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as "all" | AnnotationStatus)}>
                <option value="all">All statuses</option>
                <option value="unlabeled">unlabeled</option>
                <option value="labeled">labeled</option>
                <option value="skipped">skipped</option>
                <option value="needs_review">needs_review</option>
                <option value="approved">approved</option>
              </select>
              <input
                placeholder="Search assets..."
                value={searchText}
                onChange={(event) => {
                  setSearchText(event.target.value);
                  setPage(1);
                }}
              />
            </div>
            {isLoadingAssets ? <p>Loading assets...</p> : null}
            {!isLoadingAssets ? (
              <div style={{ display: "grid", gap: 6 }}>
                {(assetsPayload?.items ?? []).map((item) => (
                  <div key={item.asset_id} style={{ display: "grid", gridTemplateColumns: "1fr auto auto", gap: 8 }}>
                    <span>{item.relative_path || item.filename}</span>
                    <span>{item.status}</span>
                    <span>{item.split ?? "-"}</span>
                  </div>
                ))}
              </div>
            ) : null}
            {assetsPayload ? (
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 10 }}>
                <button type="button" className="ghost-button" disabled={page <= 1} onClick={() => setPage((value) => Math.max(1, value - 1))}>
                  Prev
                </button>
                <span>
                  Page {assetsPayload.page} | {assetsPayload.total} assets
                </span>
                <button
                  type="button"
                  className="ghost-button"
                  disabled={assetsPayload.page * assetsPayload.page_size >= assetsPayload.total}
                  onClick={() => setPage((value) => value + 1)}
                >
                  Next
                </button>
              </div>
            ) : null}
          </section>

          <section className="placeholder-card">
            <h3>Summary</h3>
            {!previewSummary ? <p>Run preview to compute counts.</p> : null}
            {previewSummary ? (
              <div style={{ display: "grid", gap: 10 }}>
                <p>Total: {previewSummary.total}</p>
                <p>
                  Splits: train {previewSummary.split_counts.train} | val {previewSummary.split_counts.val} | test{" "}
                  {previewSummary.split_counts.test}
                </p>
                <div>
                  <h4>Class Distribution</h4>
                  <div style={{ display: "grid", gap: 6 }}>
                    {Object.entries(previewSummary.class_counts).map(([classId, count]) => (
                      <div key={classId} style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 8 }}>
                        <span>{classDisplayName(classId)}</span>
                        <span>{count}</span>
                      </div>
                    ))}
                  </div>
                </div>
                {previewSummary.warnings.length > 0 ? (
                  <div>
                    <h4>Warnings</h4>
                    <ul>
                      {previewSummary.warnings.map((warning) => (
                        <li key={warning}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            ) : null}
          </section>
        </div>
      </section>
    </main>
  );
}
