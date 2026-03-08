"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { useParams, usePathname, useRouter, useSearchParams } from "next/navigation";

import { ProjectCreateModal, type CreateProjectDraft } from "../ProjectCreateModal";
import { ProjectAssetsTaskModal } from "../project-assets/ProjectAssetsTaskModal";
import { useProjectNavigationGuard } from "../ProjectNavigationContext";
import { createProject, listExperiments, listProjectModels, type Project, type Task } from "../../../lib/api";
import { useAssets } from "../../../lib/hooks/useAssets";
import { useLabels } from "../../../lib/hooks/useLabels";
import { useProject } from "../../../lib/hooks/useProject";
import { useTasks } from "../../../lib/hooks/useTasks";
import { useWorkspaceTaskState } from "../../../lib/hooks/useWorkspaceTaskState";
import {
  buildProjectSectionHref,
  deriveProjectSectionFromPathname,
  type ProjectShellSection,
} from "../../../lib/workspace/projectRouting";

const LAST_PROJECT_STORAGE_KEY = "pixel-sheriff:last-project-id:v1";

export interface ProjectWorkspaceStats {
  imageCount: number | null;
  classCount: number | null;
  modelCount: number | null;
  experimentCount: number | null;
}

interface ProjectShellContextValue {
  projectId: string;
  project: Project | null;
  projects: Project[];
  currentSection: ProjectShellSection;
  tasks: Task[];
  selectedTaskId: string | null;
  selectedTask: Task | null;
  isTaskLabelsLocked: boolean;
  projectStats: ProjectWorkspaceStats;
  hasUnsavedDrafts: boolean;
  shellMessage: string | null;
  setShellMessage: (message: string | null) => void;
  refetchProjects: () => Promise<unknown>;
  selectProject: (projectId: string) => void;
  selectTask: (taskId: string) => void;
  openCreateProject: () => void;
  openCreateTask: () => void;
  navigateToSection: (section: ProjectShellSection) => void;
}

const ProjectShellContext = createContext<ProjectShellContextValue | null>(null);

export function ProjectShellProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  const params = useParams<{ projectId: string }>();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const projectId = decodeURIComponent(params?.projectId ?? "");
  const currentSection = deriveProjectSectionFromPathname(pathname) as ProjectShellSection;
  const { hasUnsavedDrafts, guardedNavigate } = useProjectNavigationGuard();
  const { data: projects, isLoading: isProjectsLoading, refetch: refetchProjects } = useProject();
  const [isCreateProjectOpen, setIsCreateProjectOpen] = useState(false);
  const [isCreatingProject, setIsCreatingProject] = useState(false);
  const [shellMessage, setShellMessage] = useState<string | null>(null);
  const [projectStats, setProjectStats] = useState<ProjectWorkspaceStats>({
    imageCount: null,
    classCount: null,
    modelCount: null,
    experimentCount: null,
  });

  const project = useMemo(
    () => projects.find((item) => item.id === projectId) ?? null,
    [projectId, projects],
  );

  const requestedTaskId = searchParams.get("taskId");
  const { data: tasks, refetch: refetchTasks } = useTasks(projectId || null);
  const syncTaskInUrl = (taskId: string) => {
    const nextParams = new URLSearchParams(searchParams.toString());
    nextParams.set("taskId", taskId);
    const query = nextParams.toString();
    router.replace(query ? `${pathname}?${query}` : pathname);
  };
  const taskState = useWorkspaceTaskState({
    selectedProjectId: projectId || null,
    selectedProjectDefaultTaskId: project?.default_task_id,
    tasks,
    requestedTaskId,
    syncTaskInUrl,
    refetchTasks,
    setMessage: setShellMessage,
  });
  const selectedTaskId = taskState.selectedTaskId;
  const selectedTask = taskState.selectedTask;

  const { data: assets } = useAssets(projectId || null, selectedTaskId);
  const { data: labels } = useLabels(projectId || null, selectedTaskId);

  useEffect(() => {
    if (isProjectsLoading) return;
    if (projects.length === 0) {
      router.replace("/projects");
      return;
    }
    if (project) return;

    const storedProjectId = typeof window !== "undefined" ? window.localStorage.getItem(LAST_PROJECT_STORAGE_KEY) : null;
    const fallbackProject =
      (storedProjectId && projects.find((item) => item.id === storedProjectId)) ?? projects[0];
    router.replace(buildProjectSectionHref(fallbackProject.id, currentSection));
  }, [currentSection, isProjectsLoading, project, projects, router]);

  useEffect(() => {
    if (!project) return;
    if (typeof window === "undefined") return;
    window.localStorage.setItem(LAST_PROJECT_STORAGE_KEY, project.id);
  }, [project]);

  useEffect(() => {
    if (!projectId) {
      setProjectStats({ imageCount: null, classCount: null, modelCount: null, experimentCount: null });
      return;
    }

    let active = true;
    async function loadStats() {
      try {
        const [models, experiments] = await Promise.all([listProjectModels(projectId), listExperiments(projectId)]);
        if (!active) return;
        setProjectStats((previous) => ({
          ...previous,
          modelCount: models.length,
          experimentCount: experiments.items?.length ?? 0,
        }));
      } catch {
        if (!active) return;
        setProjectStats((previous) => ({
          ...previous,
          modelCount: null,
          experimentCount: null,
        }));
      }
    }

    void loadStats();
    return () => {
      active = false;
    };
  }, [pathname, projectId]);

  useEffect(() => {
    setProjectStats((previous) => ({
      ...previous,
      imageCount: assets.length,
      classCount: labels.length,
    }));
  }, [assets.length, labels.length]);

  useEffect(() => {
    if (!shellMessage) return;
    const timeout = window.setTimeout(() => setShellMessage(null), 5000);
    return () => window.clearTimeout(timeout);
  }, [shellMessage]);

  async function handleCreateProject(draft: CreateProjectDraft) {
    setIsCreatingProject(true);
    try {
      const created = await createProject({
        name: draft.name,
        task_type: draft.taskType,
      });
      await refetchProjects();
      if (typeof window !== "undefined") {
        window.localStorage.setItem(LAST_PROJECT_STORAGE_KEY, created.id);
      }
      setIsCreateProjectOpen(false);
      guardedNavigate(() => {
        router.push(buildProjectSectionHref(created.id, "datasets"));
      });
    } catch (error) {
      setShellMessage(error instanceof Error ? `Failed to create project: ${error.message}` : "Failed to create project.");
    } finally {
      setIsCreatingProject(false);
    }
  }

  function selectProject(nextProjectId: string) {
    if (!nextProjectId || nextProjectId === projectId) return;
    guardedNavigate(() => {
      router.push(buildProjectSectionHref(nextProjectId, currentSection));
    });
  }

  function navigateToSection(section: ProjectShellSection) {
    if (!projectId || section === currentSection) return;
    guardedNavigate(() => {
      router.push(buildProjectSectionHref(projectId, section));
    });
  }

  const value = useMemo<ProjectShellContextValue>(
    () => ({
      projectId,
      project,
      projects,
      currentSection,
      tasks,
      selectedTaskId,
      selectedTask,
      isTaskLabelsLocked: taskState.isTaskLabelsLocked,
      projectStats,
      hasUnsavedDrafts,
      shellMessage,
      setShellMessage,
      refetchProjects,
      selectProject,
      selectTask: taskState.handleSelectTask,
      openCreateProject: () => setIsCreateProjectOpen(true),
      openCreateTask: taskState.handleOpenCreateTaskModal,
      navigateToSection,
    }),
    [
      currentSection,
      hasUnsavedDrafts,
      project,
      projectId,
      projectStats,
      projects,
      selectedTask,
      selectedTaskId,
      taskState.isTaskLabelsLocked,
      shellMessage,
      tasks,
      taskState.handleSelectTask,
      taskState.handleOpenCreateTaskModal,
      refetchProjects,
    ],
  );

  return (
    <ProjectShellContext.Provider value={value}>
      {children}
      <ProjectCreateModal
        open={isCreateProjectOpen}
        isSubmitting={isCreatingProject}
        onClose={() => setIsCreateProjectOpen(false)}
        onCreate={handleCreateProject}
      />
      <ProjectAssetsTaskModal
        open={taskState.isTaskModalOpen}
        newTaskName={taskState.newTaskName}
        newTaskKind={taskState.newTaskKind}
        newTaskLabelMode={taskState.newTaskLabelMode}
        isCreatingTask={taskState.isCreatingTask}
        onSetNewTaskName={taskState.setNewTaskName}
        onSetNewTaskKind={taskState.setNewTaskKind}
        onSetNewTaskLabelMode={taskState.setNewTaskLabelMode}
        onClose={() => taskState.setIsTaskModalOpen(false)}
        onCreate={() => void taskState.handleCreateTask()}
      />
    </ProjectShellContext.Provider>
  );
}

export function useProjectShell() {
  const context = useContext(ProjectShellContext);
  if (!context) {
    throw new Error("useProjectShell must be used within ProjectShellProvider");
  }
  return context;
}
