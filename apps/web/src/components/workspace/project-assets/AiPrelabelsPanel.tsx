import type { PrelabelProposal, PrelabelSession } from "../../../lib/api";
import { derivePrelabelSessionStatus } from "../../../lib/workspace/prelabelStatus.js";

interface AiPrelabelsPanelProps {
  session: PrelabelSession | null;
  proposals: PrelabelProposal[];
  selectedProposalId: string | null;
  onSelectProposal: (proposalId: string | null) => void;
  onAcceptSelected: () => void;
  onRejectSelected: () => void;
  onAcceptCurrentFrame: () => void;
  onRejectCurrentFrame: () => void;
  onAcceptFullSession: () => void;
  onEditSelected: () => void;
  isLoading: boolean;
  isApplying: boolean;
  errorMessage?: string | null;
}

function formatPercent(value: number): string {
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

function describeDebugDetectionStatus(detection: NonNullable<PrelabelSession["debug_detections"]>[number]): string {
  if (detection.status === "matched") {
    return detection.resolved_category_name ? `matched to ${detection.resolved_category_name}` : "matched";
  }
  if (detection.status === "discarded") {
    return detection.resolved_category_name
      ? `matched to ${detection.resolved_category_name}, box discarded`
      : "box discarded";
  }
  return "no class match";
}

export function AiPrelabelsPanel({
  session,
  proposals,
  selectedProposalId,
  onSelectProposal,
  onAcceptSelected,
  onRejectSelected,
  onAcceptCurrentFrame,
  onRejectCurrentFrame,
  onAcceptFullSession,
  onEditSelected,
  isLoading,
  isApplying,
  errorMessage = null,
}: AiPrelabelsPanelProps) {
  const selectedProposal = proposals.find((proposal) => proposal.id === selectedProposalId) ?? null;
  const progressDenominator = Math.max(session?.enqueued_assets ?? 0, 1);
  const progressValue = session ? (session.processed_assets ?? 0) / progressDenominator : 0;
  const sessionStatusView = derivePrelabelSessionStatus(session);
  const debugDetections = session ? [...(session.debug_detections ?? [])].slice(-8).reverse() : [];

  return (
    <section className="ai-prelabels-panel" aria-label="AI prelabels panel" data-testid="ai-prelabels-panel">
      <div className="ai-prelabels-head">
        <div>
          <h4>AI Prelabels</h4>
          {session && sessionStatusView ? (
            <div className="ai-prelabels-status-row">
              <span
                className={`ai-prelabels-status-badge is-${sessionStatusView.badgeTone}`}
                data-testid="ai-prelabels-status-badge"
              >
                {sessionStatusView.badgeLabel}
              </span>
              <span className="ai-prelabels-status-text">{sessionStatusView.description}</span>
            </div>
          ) : (
            <p>No AI prelabel session on this sequence.</p>
          )}
        </div>
        {session ? <span className="ai-prelabels-count">{proposals.length} pending</span> : null}
      </div>

      {session ? (
        <>
          <div className="ai-prelabels-progress">
            <div className="ai-prelabels-progress-bar">
              <span style={{ width: `${Math.max(4, Math.round(progressValue * 100))}%` }} />
            </div>
            <div className="ai-prelabels-progress-meta">
              <span>
                {session.processed_assets}/{session.enqueued_assets} sampled frames processed
              </span>
              <span>{session.generated_proposals} proposals</span>
              <span>{session.skipped_unmatched} skipped</span>
            </div>
          </div>

          <div className="ai-prelabels-actions">
            <button
              type="button"
              className="ghost-button"
              onClick={onAcceptSelected}
              disabled={!selectedProposal || isApplying}
              data-testid="ai-prelabels-accept-selected"
            >
              Accept Selected
            </button>
            <button
              type="button"
              className="ghost-button"
              onClick={onRejectSelected}
              disabled={!selectedProposal || isApplying}
              data-testid="ai-prelabels-reject-selected"
            >
              Reject Selected
            </button>
            <button
              type="button"
              className="ghost-button"
              onClick={onEditSelected}
              disabled={!selectedProposal || isApplying}
              data-testid="ai-prelabels-edit-selected"
            >
              Edit Selected
            </button>
            <button
              type="button"
              className="ghost-button"
              onClick={onAcceptCurrentFrame}
              disabled={proposals.length === 0 || isApplying}
              data-testid="ai-prelabels-accept-frame"
            >
              Accept Frame
            </button>
            <button
              type="button"
              className="ghost-button"
              onClick={onRejectCurrentFrame}
              disabled={proposals.length === 0 || isApplying}
              data-testid="ai-prelabels-reject-frame"
            >
              Reject Frame
            </button>
            <button
              type="button"
              className="primary-button"
              onClick={onAcceptFullSession}
              disabled={isApplying || session.status === "cancelled" || session.status === "failed" || session.generated_proposals <= 0}
              data-testid="ai-prelabels-accept-session"
            >
              Accept Session
            </button>
          </div>

          {isLoading ? <p className="labels-empty">Loading AI proposals…</p> : null}
          {errorMessage ? <p className="import-field-error">{errorMessage}</p> : null}
          {!isLoading && proposals.length === 0 ? (
            <p
              className={`labels-empty${sessionStatusView?.badgeTone === "empty" ? " labels-empty-note" : ""}`}
              data-testid="ai-prelabels-empty-state"
            >
              {sessionStatusView?.emptyStateMessage ?? "Waiting for sampled frame results."}
            </p>
          ) : null}

          {debugDetections.length > 0 ? (
            <div className="ai-prelabels-debug" data-testid="ai-prelabels-debug">
              <h5>Model Predictions</h5>
              <ul className="ai-prelabels-debug-list">
                {debugDetections.map((detection, index) => (
                  <li key={`${detection.asset_id}-${detection.label_text}-${index}`} className="ai-prelabels-debug-item">
                    <span className="ai-prelabels-debug-top">
                      <strong>{detection.label_text}</strong>
                      <span>{formatPercent(detection.confidence)}</span>
                    </span>
                    <span className="ai-prelabels-debug-meta">
                      {typeof detection.asset_frame_index === "number" ? `Frame ${detection.asset_frame_index + 1}` : "Sampled frame"} • {describeDebugDetectionStatus(detection)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          {proposals.length > 0 ? (
            <ul className="ai-prelabels-list">
              {proposals.map((proposal, index) => {
                const isSelected = proposal.id === selectedProposalId;
                return (
                  <li key={proposal.id}>
                    <button
                      type="button"
                      className={`ai-prelabels-item${isSelected ? " active" : ""}`}
                      onClick={() => onSelectProposal(isSelected ? null : proposal.id)}
                      data-testid="ai-prelabels-proposal"
                      data-proposal-id={proposal.id}
                      data-selected={isSelected ? "true" : "false"}
                    >
                      <span className="ai-prelabels-item-top">
                        <strong>{proposal.label_text}</strong>
                        <span>{formatPercent(proposal.confidence)}</span>
                      </span>
                      <span className="ai-prelabels-item-meta">
                        #{index + 1} • {proposal.bbox.map((value) => value.toFixed(0)).join(", ")}
                      </span>
                      <span className="ai-prelabels-item-meta">
                        Prompt: {proposal.prompt_text || proposal.label_text}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          ) : null}
        </>
      ) : (
        <p className="labels-empty">Start a video import or webcam capture with bbox prelabels enabled to review pending AI boxes here.</p>
      )}
    </section>
  );
}
