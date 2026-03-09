"use client";

import { type CSSProperties, type FormEvent, useEffect, useState } from "react";
import { getClassColor } from "../lib/workspace/classColors";

export type AnnotationMode = "labels" | "bbox" | "segmentation";

interface LabelItem {
  id: string;
  name: string;
  isActive: boolean;
  displayOrder: number;
}

interface ManageLabelItem {
  id: string;
  name: string;
  isActive: boolean;
  displayOrder: number;
}

interface GeometryObjectListItem {
  id: string;
  kind: "bbox" | "polygon";
  categoryId: string;
  categoryName: string;
}

interface LabelPanelProps {
  labels: LabelItem[];
  allLabels: LabelItem[];
  selectedLabelIds: string[];
  onToggleLabel: (id: string) => void;
  onClearLabels: () => void;
  onSubmit: () => void;
  isSaving: boolean;
  onCreateLabel: (name: string) => Promise<void>;
  isCreatingLabel: boolean;
  editMode: boolean;
  onToggleEditMode: () => void;
  pendingCount: number;
  onSaveLabelChanges: (changes: ManageLabelItem[]) => Promise<void>;
  isSavingLabelChanges: boolean;
  onDeleteLabel: (labelId: string, labelName: string) => Promise<void>;
  deletingLabelId: string | null;
  labelsLocked: boolean;
  canSubmit: boolean;
  multiLabelEnabled: boolean;
  onToggleMultiLabel: () => void;
  annotationMode: AnnotationMode;
  projectMode: AnnotationMode;
  onChangeAnnotationMode: (mode: AnnotationMode) => void;
  selectedObjectId: string | null;
  geometryObjectCount: number;
  geometryObjects: GeometryObjectListItem[];
  hoveredObjectId: string | null;
  onHoverObject: (objectId: string | null) => void;
  onSelectObject: (objectId: string | null) => void;
  onDeleteSelectedObject: () => void;
  availableDeployments?: Array<{ deployment_id: string; name: string; device_preference: string }>;
  selectedDeploymentId?: string | null;
  selectedDeploymentName?: string | null;
  selectedDeploymentDevicePreference?: string | null;
  lastInferenceDeviceSelected?: string | null;
  suggestionPredictions?: Array<{ class_id: string; class_name: string; score: number }>;
  suggestionBoxes?: Array<{ class_id: string; class_name: string; score: number; bbox: number[] }>;
  suggestionScoreThreshold?: number;
  onChangeSuggestionScoreThreshold?: (value: number) => void;
  isSuggesting?: boolean;
  hasCompatibleDeployment?: boolean;
  onChangeSelectedDeploymentId?: (deploymentId: string) => void;
  onSuggest?: () => void;
  onApplySuggestedLabel?: (categoryId: string) => void;
}

export function LabelPanel({
  labels,
  allLabels,
  selectedLabelIds,
  onToggleLabel,
  onClearLabels,
  onSubmit,
  isSaving,
  onCreateLabel,
  isCreatingLabel,
  editMode,
  onToggleEditMode,
  pendingCount,
  onSaveLabelChanges,
  isSavingLabelChanges,
  onDeleteLabel,
  deletingLabelId,
  labelsLocked,
  canSubmit,
  multiLabelEnabled,
  onToggleMultiLabel,
  annotationMode,
  projectMode,
  onChangeAnnotationMode,
  selectedObjectId,
  geometryObjectCount,
  geometryObjects,
  hoveredObjectId,
  onHoverObject,
  onSelectObject,
  onDeleteSelectedObject,
  availableDeployments = [],
  selectedDeploymentId = null,
  selectedDeploymentName = null,
  selectedDeploymentDevicePreference = null,
  lastInferenceDeviceSelected = null,
  suggestionPredictions = [],
  suggestionBoxes = [],
  suggestionScoreThreshold = 0.3,
  onChangeSuggestionScoreThreshold,
  isSuggesting = false,
  hasCompatibleDeployment = false,
  onChangeSelectedDeploymentId,
  onSuggest,
  onApplySuggestedLabel,
}: LabelPanelProps) {
  const [manageMode, setManageMode] = useState(false);
  const [draftLabels, setDraftLabels] = useState<ManageLabelItem[]>([]);
  const [newLabelName, setNewLabelName] = useState("");

  useEffect(() => {
    setDraftLabels(
      allLabels
        .slice()
        .sort((a, b) => a.displayOrder - b.displayOrder)
        .map((label) => ({ ...label })),
    );
  }, [allLabels, manageMode]);

  useEffect(() => {
    if (!labelsLocked) return;
    setManageMode(false);
  }, [labelsLocked]);

  async function handleCreateLabelSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const labelName = newLabelName.trim();
    if (!labelName) return;
    await onCreateLabel(labelName);
    setNewLabelName("");
  }

  function moveDraft(index: number, direction: -1 | 1) {
    const nextIndex = index + direction;
    if (nextIndex < 0 || nextIndex >= draftLabels.length) return;

    const next = draftLabels.slice();
    const current = next[index];
    next[index] = next[nextIndex];
    next[nextIndex] = current;
    setDraftLabels(next);
  }

  function updateDraft(id: string, patch: Partial<ManageLabelItem>) {
    setDraftLabels((previous) => previous.map((item) => (item.id === id ? { ...item, ...patch } : item)));
  }

  async function handleSaveLabelChanges() {
    const normalized = draftLabels.map((item, index) => ({ ...item, displayOrder: index }));
    await onSaveLabelChanges(normalized);
    setManageMode(false);
  }

  function modeLabelText() {
    if (annotationMode === "bbox") return "Bounding box mode";
    if (annotationMode === "segmentation") return "Segmentation mode";
    return "Classification mode";
  }

  const taskModeLocked = projectMode !== "labels";
  const selectedLabelNames = allLabels
    .filter((label) => selectedLabelIds.includes(label.id))
    .sort((a, b) => a.displayOrder - b.displayOrder)
    .map((label) => label.name);

  return (
    <aside className="labels-panel" aria-label="Label tools" data-testid="label-panel">
      <div className="label-tabs">
        <button
          type="button"
          className={annotationMode === "labels" ? "tab-button active" : "tab-button"}
          onClick={() => onChangeAnnotationMode("labels")}
          disabled={projectMode !== "labels"}
          title={projectMode !== "labels" ? "Project mode is locked to a non-label task type" : undefined}
        >
          Labels
        </button>
        <button
          type="button"
          className={annotationMode === "bbox" ? "tab-button active" : "tab-button"}
          onClick={() => onChangeAnnotationMode("bbox")}
          disabled={projectMode !== "bbox"}
          title={projectMode !== "bbox" ? "Project mode is locked to a different task type" : undefined}
        >
          Bounding Boxes
        </button>
        <button
          type="button"
          className={annotationMode === "segmentation" ? "tab-button active" : "tab-button"}
          onClick={() => onChangeAnnotationMode("segmentation")}
          disabled={projectMode !== "segmentation"}
          title={projectMode !== "segmentation" ? "Project mode is locked to a different task type" : undefined}
        >
          Segmentation
        </button>
      </div>

      <div className="label-manage-toolbar">
        <button
          type="button"
          className="ghost-button"
          onClick={() => setManageMode((value) => !value)}
          disabled={labelsLocked}
        >
          {manageMode ? "Close Manage" : "Manage Labels"}
        </button>
        <button
          type="button"
          className={multiLabelEnabled ? "ghost-button active-toggle" : "ghost-button"}
          onClick={onToggleMultiLabel}
          disabled={!manageMode || taskModeLocked || labelsLocked}
        >
          Task Multi-label: {multiLabelEnabled ? "On" : "Off"}
        </button>
        {annotationMode !== "labels" ? (
          <button
            type="button"
            className="ghost-button danger-button"
            onClick={onDeleteSelectedObject}
            disabled={!selectedObjectId}
            title={selectedObjectId ?? undefined}
          >
            Delete Selected
          </button>
        ) : null}
        {manageMode ? (
          <button
            type="button"
            className="primary-button"
            onClick={handleSaveLabelChanges}
            disabled={isSavingLabelChanges || labelsLocked}
          >
            {isSavingLabelChanges ? "Saving..." : "Save Labels"}
          </button>
        ) : null}
      </div>
      {labelsLocked ? (
        <p className="label-lock-notice">
          Labels are locked for this task because dataset versions already exist. Create a new task to evolve classes safely.
        </p>
      ) : null}
      {annotationMode === "labels" ? (
        <button
          type="button"
          className="ghost-button danger-button subtle-danger"
          onClick={onClearLabels}
          disabled={selectedLabelIds.length === 0}
        >
          Clear Selected Labels
        </button>
      ) : null}

      {annotationMode !== "labels" ? (
        <>
          <div className="label-section-head">
            <h4>Objects</h4>
            <span>{modeLabelText()}</span>
          </div>
          <p className="labels-empty">
            {geometryObjectCount} object{geometryObjectCount === 1 ? "" : "s"} on this image.
            {" "}
            {selectedObjectId ? "Object selected on canvas." : "Select or draw an object, then choose a class."}
          </p>
          {geometryObjects.length === 0 ? (
            <p className="labels-empty">No objects yet on this image.</p>
          ) : (
            <ul className="geometry-object-list" data-testid="geometry-object-list">
              {geometryObjects.map((object, index) => {
                const isSelected = selectedObjectId === object.id;
                const isHovered = hoveredObjectId === object.id;
                const classColor = getClassColor(object.categoryId);
                return (
                  <li key={object.id}>
                    <button
                      type="button"
                      className={`geometry-object-item${isSelected ? " active" : ""}${isHovered ? " is-hovered" : ""}`}
                      onClick={() => onSelectObject(object.id)}
                      onMouseEnter={() => onHoverObject(object.id)}
                      onMouseLeave={() => onHoverObject(null)}
                      title={object.id}
                      style={
                        {
                          "--geometry-item-bg": classColor.chipBackground,
                          "--geometry-item-border": classColor.chipBorder,
                          "--geometry-item-active-bg": classColor.chipActiveBackground,
                          "--geometry-item-text": classColor.chipText,
                        } as CSSProperties
                      }
                      data-testid="geometry-object-item"
                      data-object-id={object.id}
                      data-category-id={object.categoryId}
                      data-selected={isSelected ? "true" : "false"}
                    >
                      <span className="geometry-object-index">{index + 1}</span>
                      <span>{`${index + 1} ${object.categoryName}`}</span>
                      <span>{object.kind === "bbox" ? "Bounding Box" : "Segmentation"}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </>
      ) : null}

      {manageMode ? (
        <form className="label-create-form" onSubmit={handleCreateLabelSubmit}>
          <input
            name="label_name"
            className="label-create-input"
            placeholder="Create label (e.g. Mushroom)"
            maxLength={48}
            value={newLabelName}
            onChange={(event) => setNewLabelName(event.target.value)}
            disabled={labelsLocked}
          />
          <button type="submit" className="ghost-button" disabled={isCreatingLabel || labelsLocked}>
            {isCreatingLabel ? "Adding..." : "Add"}
          </button>
        </form>
      ) : null}

      {manageMode ? (
        <ul className="label-manage-list">
          {draftLabels.map((label, index) => (
            <li key={label.id} className="label-manage-item">
              <input
                className="label-manage-input"
                value={label.name}
                onChange={(event) => updateDraft(label.id, { name: event.target.value })}
                disabled={labelsLocked}
              />
              <label className="label-manage-active">
                <input
                  type="checkbox"
                  checked={label.isActive}
                  onChange={(event) => updateDraft(label.id, { isActive: event.target.checked })}
                  disabled={labelsLocked}
                />
                Active
              </label>
              <div className="label-manage-order">
                <button type="button" className="ghost-icon-button" onClick={() => moveDraft(index, -1)} disabled={labelsLocked}>
                  ^
                </button>
                <button type="button" className="ghost-icon-button" onClick={() => moveDraft(index, 1)} disabled={labelsLocked}>
                  v
                </button>
              </div>
              <button
                type="button"
                className="ghost-button danger-button"
                onClick={() => void onDeleteLabel(label.id, label.name)}
                disabled={labelsLocked || deletingLabelId === label.id}
              >
                {deletingLabelId === label.id ? "Deleting..." : "Delete"}
              </button>
            </li>
          ))}
        </ul>
      ) : labels.length === 0 ? (
        <p className="labels-empty">No active labels yet. Add one above to start annotating.</p>
      ) : (
        <>
          <div className="label-section-head">
            <h4>Classes</h4>
            {annotationMode === "labels" ? null : (
              <span>{selectedObjectId ? "Click to assign selected object." : "Click to set default for new objects."}</span>
            )}
          </div>
          {annotationMode === "labels" ? (
            <p className="label-selection-summary">
              Assigned: {selectedLabelNames.length > 0 ? selectedLabelNames.join(", ") : "none"}
            </p>
          ) : null}
          <ol className="label-list" data-testid="label-list">
            {labels.map((label, index) => {
              const classColor = getClassColor(label.id);
              const isActive = selectedLabelIds.includes(label.id);
              return (
                <li key={label.id}>
                  <button
                    type="button"
                    onClick={() => onToggleLabel(label.id)}
                    className={isActive ? "label-item active" : "label-item"}
                    style={
                      {
                        "--class-chip-bg": classColor.chipBackground,
                        "--class-chip-border": classColor.chipBorder,
                        "--class-chip-active-bg": classColor.chipActiveBackground,
                        "--class-chip-text": classColor.chipText,
                      } as CSSProperties
                    }
                    data-testid="label-chip"
                    data-category-id={label.id}
                    data-selected={isActive ? "true" : "false"}
                  >
                    <span className="label-index">{index + 1}</span>
                    <span>{label.name}</span>
                    {isActive ? <span className="label-check">Assigned</span> : null}
                  </button>
                </li>
              );
            })}
          </ol>
        </>
      )}

      {annotationMode === "labels" || annotationMode === "bbox" ? (
        <section className="placeholder-card">
          <h4>Suggestions</h4>
          {hasCompatibleDeployment ? (
            <>
              <label className="project-field">
                <span>Model</span>
                <select
                  value={selectedDeploymentId ?? ""}
                  onChange={(event) => onChangeSelectedDeploymentId?.(event.target.value)}
                >
                  {availableDeployments.map((deployment) => (
                    <option key={deployment.deployment_id} value={deployment.deployment_id}>
                      {deployment.name}
                    </option>
                  ))}
                </select>
              </label>
              <p className="labels-empty">
                Model: {selectedDeploymentName ?? "-"} | preference: {selectedDeploymentDevicePreference ?? "-"} | last device: {lastInferenceDeviceSelected ?? "-"}
              </p>
              {annotationMode === "bbox" ? (
                <label className="project-field">
                  <span>Confidence threshold</span>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    value={suggestionScoreThreshold}
                    onChange={(event) => onChangeSuggestionScoreThreshold?.(Number.parseFloat(event.target.value) || 0)}
                  />
                </label>
              ) : null}
              <div className="label-actions">
                <button type="button" className="ghost-button" onClick={onSuggest} disabled={isSuggesting || !onSuggest}>
                  {isSuggesting ? "Suggesting..." : "Suggest"}
                </button>
                {annotationMode === "labels" ? (
                  <button
                    type="button"
                    className="primary-button"
                    onClick={() => onApplySuggestedLabel?.(suggestionPredictions[0].class_id)}
                    disabled={suggestionPredictions.length === 0 || !onApplySuggestedLabel}
                  >
                    Apply top-1
                  </button>
                ) : null}
              </div>
              {annotationMode === "labels" && suggestionPredictions.length > 0 ? (
                <ol className="label-list">
                  {suggestionPredictions.map((row) => (
                    <li key={`${row.class_id}-${row.class_name}`}>
                      <button type="button" className="label-item" onClick={() => onApplySuggestedLabel?.(row.class_id)}>
                        <span>{row.class_name}</span>
                        <span className="label-check">{row.score.toFixed(3)}</span>
                      </button>
                    </li>
                  ))}
                </ol>
              ) : annotationMode === "bbox" && suggestionBoxes.length > 0 ? (
                <ol className="label-list">
                  {suggestionBoxes.map((row, index) => (
                    <li key={`${row.class_id}-${index}`}>
                      <button type="button" className="label-item">
                        <span>{row.class_name}</span>
                        <span className="label-check">{row.score.toFixed(3)}</span>
                      </button>
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="labels-empty">No suggestions yet.</p>
              )}
            </>
          ) : (
            <p className="labels-empty">No deployed models for this task. Open Deploy tab to configure one.</p>
          )}
        </section>
      ) : null}

      <div className="label-actions">
        <button type="button" className="ghost-button" onClick={onToggleEditMode}>
          {editMode ? "Exit Edit" : "Edit"}
        </button>
        <button
          type="button"
          className="primary-button"
          onClick={onSubmit}
          disabled={isSaving || !canSubmit}
          data-testid="annotation-submit-button"
        >
          {isSaving ? "Saving..." : `Submit${pendingCount > 0 ? ` (${pendingCount})` : ""}`}
        </button>
      </div>

      <section className="placeholder-card annotation-shortcuts">
        <h4>Shortcut Readiness</h4>
        <p className="labels-empty">E = select | W = draw box | Del = delete | 1-9 = assign class | A / D = previous / next image</p>
      </section>
    </aside>
  );
}
