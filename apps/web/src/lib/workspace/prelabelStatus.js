function normalizeCount(value) {
  return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : 0;
}

function resolveSourceLabel(session) {
  if (!session || typeof session !== "object") return "Florence-2";
  const sourceLabel = typeof session.source_label === "string" && session.source_label.trim()
    ? session.source_label.trim()
    : session.source_type === "active_deployment"
      ? "Project model"
      : "Florence-2";
  const devicePreference = typeof session.device_preference === "string" && session.device_preference.trim()
    ? session.device_preference.trim()
    : "";
  return devicePreference ? `${sourceLabel} • pref ${devicePreference}` : sourceLabel;
}

function derivePrelabelSessionStatus(session) {
  if (!session || typeof session !== "object") return null;

  const sourceLabel = resolveSourceLabel(session);
  const status = typeof session.status === "string" ? session.status.toLowerCase() : "";
  const generatedProposals = normalizeCount(session.generated_proposals);
  const skippedUnmatched = normalizeCount(session.skipped_unmatched);
  const errorMessage = typeof session.error_message === "string" && session.error_message.trim()
    ? session.error_message.trim()
    : "";

  if (status === "failed") {
    return {
      badgeTone: "failed",
      badgeLabel: "Failed",
      description: `${sourceLabel} • processing failed`,
      emptyStateMessage: errorMessage || "AI prelabels failed on this sequence.",
    };
  }

  if (status === "cancelled") {
    return {
      badgeTone: "cancelled",
      badgeLabel: "Cancelled",
      description: `${sourceLabel} • session cancelled`,
      emptyStateMessage: "This AI prelabel session was cancelled.",
    };
  }

  if (status === "queued") {
    return {
      badgeTone: "running",
      badgeLabel: "Queued",
      description: `${sourceLabel} • waiting to start`,
      emptyStateMessage: "Waiting for sampled frame results.",
    };
  }

  if (status === "running") {
    return {
      badgeTone: "running",
      badgeLabel: "Processing",
      description: `${sourceLabel} • sampling frames`,
      emptyStateMessage: "Waiting for sampled frame results.",
    };
  }

  if (status === "completed" && generatedProposals === 0) {
    if (skippedUnmatched > 0) {
      return {
        badgeTone: "empty",
        badgeLabel: "No Matching Labels",
        description: `${sourceLabel} • detections did not match task classes`,
        emptyStateMessage: "AI prelabels completed, but detections did not match your task classes.",
      };
    }
    return {
      badgeTone: "empty",
      badgeLabel: "No Detections",
      description: `${sourceLabel} • completed without proposals`,
      emptyStateMessage: "AI prelabels completed, but no proposals were generated on sampled frames.",
    };
  }

  if (status === "completed") {
    return {
      badgeTone: "completed",
      badgeLabel: "Ready To Review",
      description: `${sourceLabel} • proposals are ready for review`,
      emptyStateMessage: "No pending proposals on this frame.",
    };
  }

  return {
    badgeTone: "idle",
    badgeLabel: "Unknown",
    description: `${sourceLabel} • status unavailable`,
    emptyStateMessage: "AI prelabel status is unavailable.",
  };
}

module.exports = {
  derivePrelabelSessionStatus,
};
