"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { ProjectCreateModal, type CreateProjectDraft } from "../../components/workspace/ProjectCreateModal";
import { createProject } from "../../lib/api";
import { useProject } from "../../lib/hooks/useProject";
import { buildProjectSectionHref } from "../../lib/workspace/projectRouting";

const LAST_PROJECT_STORAGE_KEY = "pixel-sheriff:last-project-id:v1";

export default function ProjectsEntryPage() {
  const router = useRouter();
  const { data: projects, isLoading, refetch } = useProject();
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);

  const preferredProjectId = useMemo(() => {
    if (typeof window === "undefined") return null;
    const raw = window.localStorage.getItem(LAST_PROJECT_STORAGE_KEY);
    return raw && raw.trim() ? raw : null;
  }, []);

  useEffect(() => {
    if (isLoading) return;
    if (projects.length === 0) return;

    const preferred =
      (preferredProjectId && projects.find((project) => project.id === preferredProjectId)) ?? projects[0];
    router.replace(buildProjectSectionHref(preferred.id, "datasets"));
  }, [isLoading, preferredProjectId, projects, router]);

  async function handleCreateProject(draft: CreateProjectDraft) {
    setIsCreating(true);
    try {
      const created = await createProject({ name: draft.name, task_type: draft.taskType });
      await refetch();
      if (typeof window !== "undefined") {
        window.localStorage.setItem(LAST_PROJECT_STORAGE_KEY, created.id);
      }
      setIsCreateOpen(false);
      router.push(buildProjectSectionHref(created.id, "datasets"));
    } finally {
      setIsCreating(false);
    }
  }

  if (isLoading || projects.length > 0) {
    return (
      <main className="workspace-shell project-page-shell">
        <section className="workspace-frame project-content-frame">
          <header className="project-loading">Loading projects...</header>
        </section>
      </main>
    );
  }

  return (
    <>
      <main className="workspace-shell project-page-shell">
        <section className="workspace-frame project-content-frame project-empty-state">
          <h1>Projects</h1>
          <p>Create your first isolated CV workspace to manage datasets, models, and experiments in one context.</p>
          <button type="button" className="primary-button" onClick={() => setIsCreateOpen(true)}>
            + Create Project
          </button>
        </section>
      </main>
      <ProjectCreateModal
        open={isCreateOpen}
        isSubmitting={isCreating}
        onClose={() => setIsCreateOpen(false)}
        onCreate={handleCreateProject}
      />
    </>
  );
}

