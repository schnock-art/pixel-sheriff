import type { TaskKind } from "../../../lib/api";

export function ProjectAssetsTaskModal({
  open,
  newTaskName,
  newTaskKind,
  newTaskLabelMode,
  isCreatingTask,
  onSetNewTaskName,
  onSetNewTaskKind,
  onSetNewTaskLabelMode,
  onClose,
  onCreate,
}: {
  open: boolean;
  newTaskName: string;
  newTaskKind: TaskKind;
  newTaskLabelMode: "single_label" | "multi_label";
  isCreatingTask: boolean;
  onSetNewTaskName: (value: string) => void;
  onSetNewTaskKind: (value: TaskKind) => void;
  onSetNewTaskLabelMode: (value: "single_label" | "multi_label") => void;
  onClose: () => void;
  onCreate: () => void;
}) {
  if (!open) return null;

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Create Task">
      <section className="placeholder-card" style={{ maxWidth: 520, margin: "10vh auto 0", display: "grid", gap: 12 }}>
        <h3 style={{ margin: 0 }}>Create Task</h3>
        <label className="project-field">
          <span>Name</span>
          <input value={newTaskName} onChange={(event) => onSetNewTaskName(event.target.value)} maxLength={120} />
        </label>
        <label className="project-field">
          <span>Kind</span>
          <select value={newTaskKind} onChange={(event) => onSetNewTaskKind(event.target.value as TaskKind)}>
            <option value="classification">Classification</option>
            <option value="bbox">Bounding boxes</option>
            <option value="segmentation">Segmentation</option>
          </select>
        </label>
        {newTaskKind === "classification" ? (
          <label className="project-field">
            <span>Label mode</span>
            <select value={newTaskLabelMode} onChange={(event) => onSetNewTaskLabelMode(event.target.value as "single_label" | "multi_label")}>
              <option value="single_label">Single label</option>
              <option value="multi_label">Multi label</option>
            </select>
          </label>
        ) : null}
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
          <button type="button" className="ghost-button" onClick={onClose} disabled={isCreatingTask}>
            Cancel
          </button>
          <button type="button" className="primary-button" onClick={onCreate} disabled={isCreatingTask || !newTaskName.trim()}>
            {isCreatingTask ? "Creating..." : "Create Task"}
          </button>
        </div>
      </section>
    </div>
  );
}
