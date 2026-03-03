import { useEffect } from "react";

import { resolveWorkspaceHotkeyAction } from "../workspace/hotkeys";

interface LabelRow {
  id: number;
}

type AnnotationMode = "labels" | "bbox" | "segmentation";

interface UseWorkspaceHotkeysParams {
  activeLabelRows: LabelRow[];
  annotationMode: AnnotationMode;
  assetRowsLength: number;
  selectedObjectId: string | null;
  onPrev: () => void;
  onNext: () => void;
  onLabelHotkey: (labelId: number, selectedObjectId: string | null) => void;
}

export function useWorkspaceHotkeys({
  activeLabelRows,
  annotationMode,
  assetRowsLength,
  selectedObjectId,
  onPrev,
  onNext,
  onLabelHotkey,
}: UseWorkspaceHotkeysParams) {
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const action = resolveWorkspaceHotkeyAction(event, { activeLabelCount: activeLabelRows.length });
      if (!action) return;

      if (action.type === "navigate_prev") {
        event.preventDefault();
        onPrev();
        return;
      }
      if (action.type === "navigate_next") {
        event.preventDefault();
        onNext();
        return;
      }

      const label = activeLabelRows[action.labelIndex];
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
  }, [activeLabelRows, annotationMode, assetRowsLength, onLabelHotkey, onNext, onPrev, selectedObjectId]);
}
