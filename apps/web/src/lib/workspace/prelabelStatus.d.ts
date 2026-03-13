export interface PrelabelSessionStatusView {
  badgeTone: "running" | "completed" | "empty" | "failed" | "cancelled" | "idle";
  badgeLabel: string;
  description: string;
  emptyStateMessage: string;
}

export function derivePrelabelSessionStatus(
  session:
    | {
        source_label?: string | null;
        source_type?: string | null;
        device_preference?: string | null;
        status?: string | null;
        generated_proposals?: number | null;
        skipped_unmatched?: number | null;
        error_message?: string | null;
      }
    | null
    | undefined,
): PrelabelSessionStatusView | null;
