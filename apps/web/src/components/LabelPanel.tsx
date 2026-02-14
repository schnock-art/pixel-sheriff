"use client";

import { type FormEvent, useEffect, useState } from "react";

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

interface LabelPanelProps {
  labels: LabelItem[];
  allLabels: LabelItem[];
  selectedLabelIds: number[];
  onToggleLabel: (id: number) => void;
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
}

export function LabelPanel({
  labels,
  allLabels,
  selectedLabelIds,
  onToggleLabel,
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

  return (
    <aside className="labels-panel" aria-label="Label tools">
      <div className="label-tabs">
        <button type="button" className="tab-button active">
          Labels
        </button>
        <button type="button" className="tab-button">
          Bounding Boxes
        </button>
        <button type="button" className="tab-button">
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
          disabled={!editMode}
        >
          Multi-label: {multiLabelEnabled ? "On" : "Off"}
        </button>
        {manageMode ? (
          <button type="button" className="primary-button" onClick={handleSaveLabelChanges} disabled={isSavingLabelChanges}>
            {isSavingLabelChanges ? "Saving..." : "Save Labels"}
          </button>
        ) : null}
      </div>

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
                  ↑
                </button>
                <button type="button" className="ghost-icon-button" onClick={() => moveDraft(index, 1)}>
                  ↓
                </button>
              </div>
            </li>
          ))}
        </ul>
      ) : labels.length === 0 ? (
        <p className="labels-empty">No active labels yet. Add one above to start annotating.</p>
      ) : (
        <ol className="label-list">
          {labels.map((label, index) => (
            <li key={label.id}>
              <button
                type="button"
                onClick={() => onToggleLabel(label.id)}
                className={selectedLabelIds.includes(label.id) ? "label-item active" : "label-item"}
              >
                <span className="label-index">{index + 1}</span>
                <span>{label.name}</span>
              </button>
            </li>
          ))}
        </ol>
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
