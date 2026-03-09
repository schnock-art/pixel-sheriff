import { useEffect } from "react";

import { resolveWorkspaceHotkeyAction } from "../workspace/hotkeys";

interface LabelRow {
  id: string;
}

type AnnotationMode = "labels" | "bbox" | "segmentation";

interface UseWorkspaceHotkeysParams {
  activeLabelRows: LabelRow[];
  annotationMode: AnnotationMode;
  assetRowsLength: number;
  selectedObjectId: string | null;
  onPrev: () => void;
  onNext: () => void;
  onJumpPrev?: () => void;
  onJumpNext?: () => void;
  onTogglePlayback?: () => void;
  onLabelHotkey: (labelId: string, selectedObjectId: string | null) => void;
}

export function useWorkspaceHotkeys({
  activeLabelRows,
  annotationMode,
  assetRowsLength,
  selectedObjectId,
  onPrev,
  onNext,
  onJumpPrev,
  onJumpNext,
  onTogglePlayback,
  onLabelHotkey,
}: UseWorkspaceHotkeysParams) {
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const action = resolveWorkspaceHotkeyAction(event, { activeLabelCount: activeLabelRows.length });
      if (!action) return;
      const actionType = (action as { type: string }).type;

      if (actionType === "navigate_prev") {
        event.preventDefault();
        onPrev();
        return;
      }
      if (actionType === "navigate_next") {
        event.preventDefault();
        onNext();
        return;
      }
      if (actionType === "navigate_jump_prev") {
        if (!onJumpPrev) return;
        event.preventDefault();
        onJumpPrev();
        return;
      }
      if (actionType === "navigate_jump_next") {
        if (!onJumpNext) return;
        event.preventDefault();
        onJumpNext();
        return;
      }
      if (actionType === "toggle_playback") {
        if (!onTogglePlayback) return;
        event.preventDefault();
        onTogglePlayback();
        return;
      }

      const label = activeLabelRows[(action as { labelIndex: number }).labelIndex];
      if (!label) return;

      event.preventDefault();
      if (annotationMode === "labels") {
        onLabelHotkey(label.id, null);
      } else {
        onLabelHotkey(label.id, selectedObjectId);
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeLabelRows, annotationMode, assetRowsLength, onJumpNext, onJumpPrev, onLabelHotkey, onNext, onPrev, onTogglePlayback, selectedObjectId]);
}
