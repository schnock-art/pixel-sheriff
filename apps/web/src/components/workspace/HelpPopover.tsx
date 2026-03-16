import type { ReactNode } from "react";

interface HelpPopoverProps {
  label?: string;
  title?: string;
  align?: "left" | "right";
  children: ReactNode;
}

export function HelpPopover({ label = "Help", title, align = "right", children }: HelpPopoverProps) {
  return (
    <details className={`help-popover is-${align}`}>
      <summary className="help-popover-trigger" aria-label={label} title={label}>
        ?
      </summary>
      <div className="help-popover-card" role="note">
        {title ? <strong>{title}</strong> : null}
        <div className="help-popover-body">{children}</div>
      </div>
    </details>
  );
}
