interface SequenceToolbarProps {
  currentFrameLabel: string;
  currentTimestampLabel: string;
  isPlaying: boolean;
  onFirst: () => void;
  onPrev: () => void;
  onTogglePlayback: () => void;
  onNext: () => void;
  onLast: () => void;
}

export function SequenceToolbar({
  currentFrameLabel,
  currentTimestampLabel,
  isPlaying,
  onFirst,
  onPrev,
  onTogglePlayback,
  onNext,
  onLast,
}: SequenceToolbarProps) {
  return (
    <div className="sequence-toolbar">
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
      </div>
      <div className="sequence-toolbar-meta">
        <span>{currentFrameLabel}</span>
        <span>{currentTimestampLabel}</span>
      </div>
    </div>
  );
}
