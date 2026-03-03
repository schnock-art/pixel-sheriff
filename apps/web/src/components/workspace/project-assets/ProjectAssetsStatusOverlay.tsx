interface ProjectAssetsStatusOverlayProps {
  message: string | null;
  messageTone: "info" | "error" | "success";
  importFailures: string[];
  onDismissMessage: () => void;
}

export function ProjectAssetsStatusOverlay({
  message,
  messageTone,
  importFailures,
  onDismissMessage,
}: ProjectAssetsStatusOverlayProps) {
  return (
    <>
      {message ? (
        <div className={`status-toast ${messageTone === "error" ? "is-error" : "is-success"}`} role="status" aria-live="polite">
          <span>{message}</span>
          <button type="button" aria-label="Dismiss message" onClick={onDismissMessage}>
            x
          </button>
        </div>
      ) : null}
      {importFailures.length > 0 ? (
        <ul className="status-errors">
          {importFailures.map((failure) => (
            <li key={failure}>{failure}</li>
          ))}
        </ul>
      ) : null}
    </>
  );
}
