import type { ReactNode } from "react";

interface MediaImportWorkspaceModalProps {
  title: string;
  subtitle: string;
  headerAction?: ReactNode;
  controls: ReactNode;
  preview: ReactNode;
  footer: ReactNode;
}

export function MediaImportWorkspaceModal({
  title,
  subtitle,
  headerAction = null,
  controls,
  preview,
  footer,
}: MediaImportWorkspaceModalProps) {
  return (
    <div className="import-modal-backdrop">
      <div className="import-modal import-workspace-modal">
        <div className="import-workspace-head">
          <div>
            <h3>{title}</h3>
            <p>{subtitle}</p>
          </div>
          {headerAction}
        </div>
        <div className="import-workspace-body">
          <div className="import-workspace-controls">{controls}</div>
          <div className="import-workspace-preview">{preview}</div>
        </div>
        <div className="import-workspace-footer">{footer}</div>
      </div>
    </div>
  );
}
