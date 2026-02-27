"use client";

import { type CSSProperties, type FormEvent, useEffect, useState } from "react";
import { getClassColor } from "../lib/workspace/classColors";

export type AnnotationMode = "labels" | "bbox" | "segmentation";

interface LabelItem {
  id: number;
  name: string;
  isActive: boolean;
  displayOrder: number;
}

interface ManageLabelItem {
  id: number;
  name: string;
  isActive: boolean;
  displayOrder: number;
}

interface GeometryObjectListItem {
  id: string;
  kind: "bbox" | "polygon";
  categoryId: number;
  categoryName: string;
}

interface LabelPanelProps {
  labels: LabelItem[];
  allLabels: LabelItem[];
  selectedLabelIds: number[];
  onToggleLabel: (id: number) => void;
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

  function updateDraft(id: number, patch: Partial<ManageLabelItem>) {
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
    <aside className="labels-panel" aria-label="Label tools">
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
        <button type="button" className="ghost-button" onClick={() => setManageMode((value) => !value)}>
          {manageMode ? "Close Manage" : "Manage Labels"}
        </button>
        <button
          type="button"
          className={multiLabelEnabled ? "ghost-button active-toggle" : "ghost-button"}
          onClick={onToggleMultiLabel}
          disabled={!manageMode || taskModeLocked}
        >
          Project Multi-label: {multiLabelEnabled ? "On" : "Off"}
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
          <button type="button" className="primary-button" onClick={handleSaveLabelChanges} disabled={isSavingLabelChanges}>
            {isSavingLabelChanges ? "Saving..." : "Save Labels"}
          </button>
        ) : null}
      </div>
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
          <p className="labels-empty">
            {modeLabelText()}.
            {" "}
            {geometryObjectCount} object{geometryObjectCount === 1 ? "" : "s"} on this image.
            {" "}
            {selectedObjectId ? `Selected: ${selectedObjectId}` : "Select or draw an object, then choose a class."}
          </p>
          {geometryObjects.length === 0 ? (
            <p className="labels-empty">No objects yet on this image.</p>
          ) : (
            <ul className="geometry-object-list">
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
                    >
                      <span className="geometry-object-index">{index + 1}</span>
                      <span>{object.kind === "bbox" ? "BBox" : "Segment"}</span>
                      <span>{object.categoryName}</span>
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
          />
          <button type="submit" className="ghost-button" disabled={isCreatingLabel}>
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
              />
              <label className="label-manage-active">
                <input
                  type="checkbox"
                  checked={label.isActive}
                  onChange={(event) => updateDraft(label.id, { isActive: event.target.checked })}
                />
                Active
              </label>
              <div className="label-manage-order">
                <button type="button" className="ghost-icon-button" onClick={() => moveDraft(index, -1)}>
                  ^
                </button>
                <button type="button" className="ghost-icon-button" onClick={() => moveDraft(index, 1)}>
                  v
                </button>
              </div>
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
          <ol className="label-list">
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

      <div className="label-actions">
        <button type="button" className="ghost-button" onClick={onToggleEditMode}>
          {editMode ? "Exit Edit" : "Edit"}
        </button>
        <button type="button" className="primary-button" onClick={onSubmit} disabled={isSaving || !canSubmit}>
          {isSaving ? "Saving..." : `Submit${pendingCount > 0 ? ` (${pendingCount})` : ""}`}
        </button>
      </div>
    </aside>
  );
}
