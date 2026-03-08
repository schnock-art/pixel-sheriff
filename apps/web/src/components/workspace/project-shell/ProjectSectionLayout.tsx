import type { ReactNode } from "react";

interface ProjectSectionLayoutProps {
  title: string;
  description?: string | null;
  actions?: ReactNode;
  children: ReactNode;
}

export function ProjectSectionLayout({ title, description = null, actions = null, children }: ProjectSectionLayoutProps) {
  return (
    <section className="project-section-layout">
      <header className="project-section-header polished">
        <div>
          <h2>{title}</h2>
          {description ? <p className="project-section-description">{description}</p> : null}
        </div>
        {actions ? <div className="project-section-actions">{actions}</div> : null}
      </header>
      {children}
    </section>
  );
}
