interface SequenceToolbarProps {
  currentFrameLabel: string;
  currentTimestampLabel: string;
  isPlaying: boolean;
  pendingFrameCount?: number;
  pendingProposalCount?: number;
  onFirst: () => void;
  onPrev: () => void;
  onTogglePlayback: () => void;
  onNext: () => void;
  onLast: () => void;
  onNextPending?: () => void;
}

export function SequenceToolbar({
  currentFrameLabel,
  currentTimestampLabel,
  isPlaying,
  pendingFrameCount = 0,
  pendingProposalCount = 0,
  onFirst,
  onPrev,
  onTogglePlayback,
  onNext,
  onLast,
  onNextPending,
}: SequenceToolbarProps) {
  return (
    <div className="sequence-toolbar" data-testid="sequence-toolbar">
      <div className="sequence-toolbar-controls">
        <button type="button" className="ghost-button" onClick={onFirst}>
          |&lt;&lt;
        </button>
        <button type="button" className="ghost-button" onClick={onPrev}>
          &lt;&lt;
        </button>
        <button type="button" className="primary-button" onClick={onTogglePlayback}>
          {isPlaying ? "Pause" : "Play"}
        </button>
        <button type="button" className="ghost-button" onClick={onNext}>
          &gt;&gt;
        </button>
        <button type="button" className="ghost-button" onClick={onLast}>
          &gt;&gt;|
        </button>
        <button
          type="button"
          className="ghost-button"
          onClick={onNextPending}
          disabled={!onNextPending || pendingFrameCount <= 0}
          data-testid="sequence-next-pending-button"
        >
          Next AI
        </button>
      </div>
      <div className="sequence-toolbar-meta">
        <span>{currentFrameLabel}</span>
        <span>{currentTimestampLabel}</span>
        <span>{pendingFrameCount} frame{pendingFrameCount === 1 ? "" : "s"} with AI</span>
        <span>{pendingProposalCount} pending box{pendingProposalCount === 1 ? "" : "es"}</span>
      </div>
    </div>
  );
}
