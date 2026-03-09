interface SequenceStatusBadgeProps {
  sourceType?: string | null;
  status?: string | null;
  frameCount?: number | null;
}

export function SequenceStatusBadge({ sourceType, status, frameCount }: SequenceStatusBadgeProps) {
  if (!sourceType) return null;
  const sourceLabel = sourceType === "webcam" ? "CAM" : "VID";
  const statusLabel = status === "processing" ? "PROC" : status === "failed" ? "ERR" : "READY";
  return (
    <span className={`sequence-status-badge is-${status ?? "ready"}`}>
      <span>{sourceLabel}</span>
      <span>{frameCount ?? 0}</span>
      <span>{statusLabel}</span>
    </span>
  );
}
