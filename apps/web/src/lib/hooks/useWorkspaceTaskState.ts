import { useEffect, useState } from "react";

import { createTask, listDatasetVersions, type Task, type TaskKind } from "../api";

const PROJECT_ACTIVE_TASK_STORAGE_KEY_PREFIX = "pixel-sheriff:project-active-task:v1:";

export function useWorkspaceTaskState({
  selectedProjectId,
  selectedProjectDefaultTaskId,
  tasks,
  requestedTaskId,
  syncTaskInUrl,
  refetchTasks,
  setMessage,
}: {
  selectedProjectId: string | null;
  selectedProjectDefaultTaskId: string | null | undefined;
  tasks: Task[];
  requestedTaskId: string | null;
  syncTaskInUrl: (taskId: string) => void;
  refetchTasks: () => Promise<unknown>;
  setMessage: (message: string | null) => void;
}) {
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [isTaskModalOpen, setIsTaskModalOpen] = useState(false);
  const [newTaskName, setNewTaskName] = useState("");
  const [newTaskKind, setNewTaskKind] = useState<TaskKind>("classification");
  const [newTaskLabelMode, setNewTaskLabelMode] = useState<"single_label" | "multi_label">("single_label");
  const [isCreatingTask, setIsCreatingTask] = useState(false);
  const [isTaskLabelsLocked, setIsTaskLabelsLocked] = useState(false);

  useEffect(() => {
    if (!selectedProjectId || tasks.length === 0) {
      setSelectedTaskId(null);
      return;
    }
    const validIds = new Set(tasks.map((task) => task.id));
    const storageKey = `${PROJECT_ACTIVE_TASK_STORAGE_KEY_PREFIX}${selectedProjectId}`;
    const storedTaskId = typeof window !== "undefined" ? window.localStorage.getItem(storageKey) : null;
    const defaultTaskId = selectedProjectDefaultTaskId ?? tasks.find((task) => task.is_default)?.id ?? tasks[0]?.id ?? null;
    const nextTaskId =
      [requestedTaskId, storedTaskId, defaultTaskId].find(
        (value): value is string => Boolean(value && validIds.has(value)),
      ) ?? null;
    if (!nextTaskId) return;
    setSelectedTaskId((previous) => (previous === nextTaskId ? previous : nextTaskId));
    if (typeof window !== "undefined") {
      window.localStorage.setItem(storageKey, nextTaskId);
    }
    if (requestedTaskId !== nextTaskId) {
      syncTaskInUrl(nextTaskId);
    }
  }, [requestedTaskId, selectedProjectDefaultTaskId, selectedProjectId, syncTaskInUrl, tasks]);

  useEffect(() => {
    let active = true;
    async function loadTaskLockState() {
      if (!selectedProjectId || !selectedTaskId) {
        if (!active) return;
        setIsTaskLabelsLocked(false);
        return;
      }
      try {
        const listed = await listDatasetVersions(selectedProjectId, selectedTaskId);
        if (!active) return;
        const items = Array.isArray(listed.items) ? listed.items : [];
        setIsTaskLabelsLocked(items.length > 0);
      } catch {
        if (!active) return;
        setIsTaskLabelsLocked(false);
      }
    }
    void loadTaskLockState();
    return () => {
      active = false;
    };
  }, [selectedProjectId, selectedTaskId]);

  function handleSelectTask(nextTaskId: string) {
    if (!selectedProjectId || !nextTaskId) return;
    if (!tasks.some((task) => task.id === nextTaskId)) return;
    setSelectedTaskId(nextTaskId);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(`${PROJECT_ACTIVE_TASK_STORAGE_KEY_PREFIX}${selectedProjectId}`, nextTaskId);
    }
    syncTaskInUrl(nextTaskId);
  }

  function handleOpenCreateTaskModal() {
    if (!selectedProjectId) {
      setMessage("Select a project before creating tasks.");
      return;
    }
    setNewTaskName("");
    setNewTaskKind("classification");
    setNewTaskLabelMode("single_label");
    setIsTaskModalOpen(true);
  }

  async function handleCreateTask() {
    if (!selectedProjectId) {
      setMessage("Select a project before creating tasks.");
      return;
    }
    const taskName = newTaskName.trim();
    if (!taskName) {
      setMessage("Task name is required.");
      return;
    }
    try {
      setIsCreatingTask(true);
      const created = await createTask(selectedProjectId, {
        name: taskName,
        kind: newTaskKind,
        label_mode: newTaskKind === "classification" ? newTaskLabelMode : undefined,
      });
      await refetchTasks();
      handleSelectTask(created.id);
      setIsTaskModalOpen(false);
      setMessage(`Created task "${created.name}".`);
    } catch (error) {
      setMessage(error instanceof Error ? `Failed to create task: ${error.message}` : "Failed to create task.");
    } finally {
      setIsCreatingTask(false);
    }
  }

  return {
    selectedTaskId,
    selectedTask: tasks.find((task) => task.id === selectedTaskId) ?? null,
    isTaskLabelsLocked,
    handleSelectTask,
    isTaskModalOpen,
    setIsTaskModalOpen,
    newTaskName,
    setNewTaskName,
    newTaskKind,
    setNewTaskKind,
    newTaskLabelMode,
    setNewTaskLabelMode,
    isCreatingTask,
    handleOpenCreateTaskModal,
    handleCreateTask,
  };
}
