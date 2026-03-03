"use client";

import { useEffect, useState } from "react";

export interface CreateProjectDraft {
  name: string;
  taskType: "classification_single" | "bbox" | "segmentation";
}

interface ProjectCreateModalProps {
  open: boolean;
  isSubmitting: boolean;
  onClose: () => void;
  onCreate: (draft: CreateProjectDraft) => Promise<void>;
}

const INITIAL_DRAFT: CreateProjectDraft = {
  name: "",
  taskType: "classification_single",
};

export function ProjectCreateModal({ open, isSubmitting, onClose, onCreate }: ProjectCreateModalProps) {
  const [draft, setDraft] = useState<CreateProjectDraft>(INITIAL_DRAFT);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setDraft(INITIAL_DRAFT);
    setError(null);
  }, [open]);

  if (!open) return null;

  async function handleCreate() {
    const name = draft.name.trim();
    if (!name) {
      setError("Project name is required.");
      return;
    }

    setError(null);
    try {
      await onCreate({ ...draft, name });
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : "Failed to create project.");
    }
  }

  return (
    <div className="project-modal-backdrop">
      <div className="project-modal" role="dialog" aria-modal="true" aria-label="Create Project">
        <h3>Create Project</h3>
        <label className="project-field">
          <span>Project Name</span>
          <input
            value={draft.name}
            onChange={(event) => setDraft((previous) => ({ ...previous, name: event.target.value }))}
            placeholder="lofoten"
            disabled={isSubmitting}
          />
        </label>
        <label className="project-field">
          <span>Task Type</span>
          <select
            value={draft.taskType}
            onChange={(event) =>
              setDraft((previous) => ({
                ...previous,
                taskType: event.target.value as CreateProjectDraft["taskType"],
              }))
            }
            disabled={isSubmitting}
          >
            <option value="classification_single">Classification (single label)</option>
            <option value="bbox">Bounding Boxes</option>
            <option value="segmentation">Segmentation</option>
          </select>
        </label>
        {error ? <p className="project-field-error">{error}</p> : null}
        <div className="project-modal-actions">
          <button type="button" className="ghost-button" onClick={onClose} disabled={isSubmitting}>
            Cancel
          </button>
          <button type="button" className="primary-button" onClick={() => void handleCreate()} disabled={isSubmitting}>
            {isSubmitting ? "Creating..." : "Create Project"}
          </button>
        </div>
      </div>
    </div>
  );
}

