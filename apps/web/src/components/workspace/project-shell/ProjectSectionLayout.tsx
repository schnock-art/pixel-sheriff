import type { ReactNode } from "react";

import { HelpPopover } from "../HelpPopover";

interface ProjectSectionLayoutProps {
  title: string;
  description?: string | null;
  titleHelp?: ReactNode;
  actions?: ReactNode;
  className?: string;
  children: ReactNode;
}

export function ProjectSectionLayout({
  title,
  description = null,
  titleHelp = null,
  actions = null,
  className = "",
  children,
}: ProjectSectionLayoutProps) {
  return (
    <section className={`project-section-layout ${className}`.trim()}>
      <header className="project-section-header polished">
        <div>
          <div className="project-section-title-row">
            <h2>{title}</h2>
            {titleHelp ? (
              <HelpPopover label={`${title} help`} title={title}>
                {titleHelp}
              </HelpPopover>
            ) : null}
          </div>
          {description ? <p className="project-section-description">{description}</p> : null}
        </div>
        {actions ? <div className="project-section-actions">{actions}</div> : null}
      </header>
      {children}
    </section>
  );
}
