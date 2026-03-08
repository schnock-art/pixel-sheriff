"use client";

import type { ReactNode } from "react";

import { ProjectNavigationGuardProvider } from "../../../components/workspace/ProjectNavigationContext";
import { ProjectRibbon } from "../../../components/workspace/project-shell/ProjectRibbon";
import { ProjectShellProvider } from "../../../components/workspace/project-shell/ProjectShellContext";

function ProjectRouteLayout({ children }: { children: ReactNode }) {
  return (
    <main className="workspace-shell project-shell-wrap">
      <section className="workspace-frame project-shell-frame">
        <ProjectRibbon />
        <div className="project-shell-content">{children}</div>
      </section>
    </main>
  );
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <ProjectNavigationGuardProvider>
      <ProjectShellProvider>
        <ProjectRouteLayout>{children}</ProjectRouteLayout>
      </ProjectShellProvider>
    </ProjectNavigationGuardProvider>
  );
}
