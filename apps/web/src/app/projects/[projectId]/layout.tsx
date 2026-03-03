"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useParams, usePathname, useRouter } from "next/navigation";

import { ProjectCreateModal, type CreateProjectDraft } from "../../../components/workspace/ProjectCreateModal";
import {
  ProjectNavigationGuardProvider,
  useProjectNavigationGuard,
  type ProjectShellSection,
} from "../../../components/workspace/ProjectNavigationContext";
import { createProject } from "../../../lib/api";
import { useAssets } from "../../../lib/hooks/useAssets";
import { useLabels } from "../../../lib/hooks/useLabels";
import { useProject } from "../../../lib/hooks/useProject";
import { buildProjectSectionHref, deriveProjectSectionFromPathname } from "../../../lib/workspace/projectRouting";

const LAST_PROJECT_STORAGE_KEY = "pixel-sheriff:last-project-id:v1";
const PROJECT_STATUS_REFRESH_EVENT = "pixel-sheriff:project-status-refresh";

interface ProjectRouteLayoutProps {
  children: ReactNode;
}

interface ProjectStatusRefreshDetail {
  projectId: string;
  labeledImageCount: number;
  classCount: number;
}

function ProjectRouteLayout({ children }: ProjectRouteLayoutProps) {
  const router = useRouter();
  const params = useParams<{ projectId: string }>();
  const pathname = usePathname();
  const { hasUnsavedDrafts, guardedNavigate } = useProjectNavigationGuard();
  const { data: projects, isLoading: isProjectsLoading, refetch: refetchProjects } = useProject();
  const projectId = decodeURIComponent(params?.projectId ?? "");
  const currentSection = deriveProjectSectionFromPathname(pathname) as ProjectShellSection;

  const selectedProject = useMemo(
    () => projects.find((project) => project.id === projectId) ?? null,
    [projectId, projects],
  );

  const { annotations } = useAssets(selectedProject?.id ?? null);
  const { data: labels } = useLabels(selectedProject?.id ?? null);

  const [isProjectMenuOpen, setIsProjectMenuOpen] = useState(false);
  const [isCreateProjectOpen, setIsCreateProjectOpen] = useState(false);
  const [isCreatingProject, setIsCreatingProject] = useState(false);
  const [statusOverride, setStatusOverride] = useState<ProjectStatusRefreshDetail | null>(null);

  function navigateTo(href: string) {
    if (typeof window !== "undefined") {
      window.location.assign(href);
      return;
    }
    router.push(href);
  }

  useEffect(() => {
    if (isProjectsLoading) return;

    if (projects.length === 0) {
      router.replace("/projects");
      return;
    }

    if (selectedProject) return;

    let preferredProjectId: string | null = null;
    if (typeof window !== "undefined") {
      const stored = window.localStorage.getItem(LAST_PROJECT_STORAGE_KEY);
      preferredProjectId = stored && stored.trim() ? stored : null;
    }

    const fallbackProject =
      (preferredProjectId && projects.find((project) => project.id === preferredProjectId)) ?? projects[0];

    router.replace(buildProjectSectionHref(fallbackProject.id, currentSection));
  }, [currentSection, isProjectsLoading, projects, router, selectedProject]);

  useEffect(() => {
    if (!selectedProject) return;
    if (typeof window === "undefined") return;
    window.localStorage.setItem(LAST_PROJECT_STORAGE_KEY, selectedProject.id);
  }, [selectedProject]);

  useEffect(() => {
    setStatusOverride(null);
  }, [projectId]);

  useEffect(() => {
    function handleStatusRefresh(event: Event) {
      const customEvent = event as CustomEvent<ProjectStatusRefreshDetail>;
      const detail = customEvent.detail;
      if (!detail || detail.projectId !== projectId) return;
      setStatusOverride(detail);
    }

    window.addEventListener(PROJECT_STATUS_REFRESH_EVENT, handleStatusRefresh as EventListener);
    return () => window.removeEventListener(PROJECT_STATUS_REFRESH_EVENT, handleStatusRefresh as EventListener);
  }, [projectId]);

  const labeledImageCount = useMemo(() => {
    const labeledAssetIds = new Set<string>();
    for (const annotation of annotations) {
      if (annotation.status !== "unlabeled") labeledAssetIds.add(annotation.asset_id);
    }
    return labeledAssetIds.size;
  }, [annotations]);

  const classCount = labels.length;
  const displayedLabeledImageCount =
    statusOverride && statusOverride.projectId === projectId ? statusOverride.labeledImageCount : labeledImageCount;
  const displayedClassCount = statusOverride && statusOverride.projectId === projectId ? statusOverride.classCount : classCount;

  async function handleCreateProject(draft: CreateProjectDraft) {
    setIsCreatingProject(true);
    try {
      const created = await createProject({
        name: draft.name,
        task_type: draft.taskType,
      });
      await refetchProjects();
      if (typeof window !== "undefined") window.localStorage.setItem(LAST_PROJECT_STORAGE_KEY, created.id);
      setIsCreateProjectOpen(false);
      setIsProjectMenuOpen(false);
      guardedNavigate(() => {
        navigateTo(buildProjectSectionHref(created.id, "datasets"));
      });
    } finally {
      setIsCreatingProject(false);
    }
  }

  function handleProjectSelect(nextProjectId: string) {
    setIsProjectMenuOpen(false);
    if (!nextProjectId || nextProjectId === projectId) return;
    guardedNavigate(() => {
      navigateTo(buildProjectSectionHref(nextProjectId, currentSection));
    });
  }

  function handleTabSelect(section: ProjectShellSection) {
    if (section === currentSection) return;
    guardedNavigate(() => {
      navigateTo(buildProjectSectionHref(projectId, section));
    });
  }

  if (!selectedProject) {
    return (
      <main className="workspace-shell project-page-shell">
        <section className="workspace-frame project-content-frame">
          <header className="project-loading">Loading project workspace...</header>
        </section>
      </main>
    );
  }

  return (
    <>
      <main className="workspace-shell project-shell-wrap">
        <section className="workspace-frame project-shell-frame">
          <header className="project-shell-topbar">
            <div className="project-selector-wrap">
              <button
                type="button"
                className="project-selector-button"
                onClick={() => setIsProjectMenuOpen((open) => !open)}
                aria-expanded={isProjectMenuOpen}
              >
                Project: {selectedProject.name} <span aria-hidden>v</span>
              </button>
              {isProjectMenuOpen ? (
                <div className="project-selector-menu">
                  <button
                    type="button"
                    className="project-selector-item create"
                    onClick={() => {
                      setIsProjectMenuOpen(false);
                      setIsCreateProjectOpen(true);
                    }}
                  >
                    + Create Project
                  </button>
                  {projects.map((project) => (
                    <button
                      key={project.id}
                      type="button"
                      className={`project-selector-item${project.id === projectId ? " active" : ""}`}
                      onClick={() => handleProjectSelect(project.id)}
                    >
                      {project.name}
                    </button>
                  ))}
                </div>
              ) : null}
            </div>

            <nav className="project-shell-nav" aria-label="Project sections">
              <button
                type="button"
                className={currentSection === "datasets" ? "project-nav-tab active" : "project-nav-tab"}
                onClick={() => handleTabSelect("datasets")}
              >
                Datasets
              </button>
              <button
                type="button"
                className={currentSection === "models" ? "project-nav-tab active" : "project-nav-tab"}
                onClick={() => handleTabSelect("models")}
              >
                Models
              </button>
              <button
                type="button"
                className={currentSection === "experiments" ? "project-nav-tab active" : "project-nav-tab"}
                onClick={() => handleTabSelect("experiments")}
              >
                Experiments
              </button>
              <button type="button" className="project-nav-tab disabled" disabled title="Coming soon">
                Deploy
              </button>
            </nav>
          </header>
          <div className="project-shell-status" role="status" aria-live="polite">
            Status: {displayedLabeledImageCount} images labeled | {displayedClassCount} classes | 0 models | 0 experiments
            {hasUnsavedDrafts ? <span className="project-status-dirty"> | unsaved drafts</span> : null}
          </div>
          <div className="project-shell-content">{children}</div>
        </section>
      </main>

      <ProjectCreateModal
        open={isCreateProjectOpen}
        isSubmitting={isCreatingProject}
        onClose={() => setIsCreateProjectOpen(false)}
        onCreate={handleCreateProject}
      />
    </>
  );
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <ProjectNavigationGuardProvider>
      <ProjectRouteLayout>{children}</ProjectRouteLayout>
    </ProjectNavigationGuardProvider>
  );
}

