"use client";

import { useMemo } from "react";

import { useProjectShell } from "./ProjectShellContext";

function statText(label: string, value: number | null) {
  return `${label}: ${value == null ? "-" : value}`;
}

export function ProjectRibbon() {
  const {
    project,
    projects,
    currentSection,
    tasks,
    selectedTaskId,
    selectedTask,
    projectStats,
    hasUnsavedDrafts,
    shellMessage,
    selectProject,
    selectTask,
    openCreateProject,
    openCreateTask,
    navigateToSection,
  } = useProjectShell();

  const tabs = useMemo(
    () => [
      { key: "datasets", label: "Labeling" },
      { key: "dataset", label: "Dataset" },
      { key: "models", label: "Models" },
      { key: "experiments", label: "Experiments" },
      { key: "deploy", label: "Deploy" },
    ] as const,
    [],
  );

  return (
    <header className="project-ribbon" data-testid="project-ribbon">
      <div className="project-ribbon-main">
        <div className="project-ribbon-brand">
          <p className="project-ribbon-kicker">Pixel Sheriff</p>
          <h1>{project?.name ?? "Project Workspace"}</h1>
        </div>
        <div className="project-ribbon-controls">
          <label className="project-ribbon-field">
            <span>Project</span>
            <div className="project-ribbon-inline">
              <select
                data-testid="project-selector"
                value={project?.id ?? ""}
                onChange={(event) => selectProject(event.target.value)}
                disabled={projects.length === 0}
              >
                {projects.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
              <button type="button" className="ghost-button" onClick={openCreateProject} data-testid="project-create-button">
                + Project
              </button>
            </div>
          </label>
          <label className="project-ribbon-field">
            <span>Task</span>
            <div className="project-ribbon-inline">
              <select
                data-testid="task-selector"
                value={selectedTaskId ?? ""}
                onChange={(event) => selectTask(event.target.value)}
                disabled={tasks.length === 0}
              >
                {tasks.length === 0 ? <option value="">No tasks</option> : null}
                {tasks.map((task) => (
                  <option key={task.id} value={task.id}>
                    {task.name} [{task.kind}]
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="ghost-button"
                onClick={openCreateTask}
                disabled={!project}
                data-testid="task-create-button"
              >
                + Task
              </button>
            </div>
          </label>
        </div>
      </div>

      <div className="project-ribbon-navrow">
        <nav className="project-ribbon-tabs" aria-label="Project workflow" data-testid="workflow-tabs">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              type="button"
              className={currentSection === tab.key ? "project-ribbon-tab active" : "project-ribbon-tab"}
              onClick={() => navigateToSection(tab.key)}
              data-testid={`workflow-tab-${tab.key}`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
        <div className="project-ribbon-stats" role="status" aria-live="polite" data-testid="project-ribbon-stats">
          <span className="project-ribbon-stat">{statText("Images", projectStats.imageCount)}</span>
          <span className="project-ribbon-stat">{statText("Classes", projectStats.classCount)}</span>
          <span className="project-ribbon-stat">{statText("Models", projectStats.modelCount)}</span>
          <span className="project-ribbon-stat">{statText("Experiments", projectStats.experimentCount)}</span>
          {selectedTask ? <span className="project-ribbon-stat subtle">Task: {selectedTask.kind}</span> : null}
          {hasUnsavedDrafts ? <span className="project-ribbon-stat warning">Unsaved drafts</span> : null}
        </div>
      </div>

      {shellMessage ? <p className="project-ribbon-message">{shellMessage}</p> : null}
    </header>
  );
}
