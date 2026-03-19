import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import type { PreviewInferenceResponse } from "../../../lib/api";
import { getClassColor } from "../../../lib/workspace/classColors";
import { computeImageViewport, resolveImageBasis, toViewportCoords } from "../../../lib/workspace/geometry";

interface ModalMediaPreviewProps {
  title: string;
  subtitle: string;
  emptyMessage: string;
  mediaBasis: { width?: number | null; height?: number | null } | null;
  mediaReady: boolean;
  liveBadgeText?: string | null;
  preview: {
    status: "idle" | "loading" | "ready" | "error";
    response: PreviewInferenceResponse | null;
    error: { message: string } | null;
  };
  metaItems?: string[];
  children: ReactNode;
}

function formatConfidence(score: number) {
  return `${Math.round(score * 100)}%`;
}

export function ModalMediaPreview({
  title,
  subtitle,
  emptyMessage,
  mediaBasis,
  mediaReady,
  liveBadgeText = null,
  preview,
  metaItems = [],
  children,
}: ModalMediaPreviewProps) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!stageRef.current) return;

    function measure() {
      if (!stageRef.current) return;
      const rect = stageRef.current.getBoundingClientRect();
      setStageSize({ width: rect.width, height: rect.height });
    }

    measure();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", measure);
      return () => window.removeEventListener("resize", measure);
    }

    const observer = new ResizeObserver(() => measure());
    observer.observe(stageRef.current);
    return () => observer.disconnect();
  }, []);

  const resolvedMediaBasis = useMemo(
    () =>
      resolveImageBasis(
        {
          width: preview.response?.preview_width ?? null,
          height: preview.response?.preview_height ?? null,
        },
        mediaBasis,
      ),
    [mediaBasis, preview.response?.preview_height, preview.response?.preview_width],
  );

  const viewport = useMemo(() => {
    if (!resolvedMediaBasis) return null;
    return computeImageViewport(stageSize.width, stageSize.height, resolvedMediaBasis.width, resolvedMediaBasis.height);
  }, [resolvedMediaBasis, stageSize.height, stageSize.width]);

  const matchedBoxes = useMemo(
    () =>
      preview.response?.task === "bbox"
        ? preview.response.boxes.filter((box) => box.matched && Array.isArray(box.bbox) && box.bbox.length === 4)
        : [],
    [preview.response],
  );
  const unmatchedDebug = useMemo(
    () =>
      preview.response?.task === "bbox"
        ? preview.response.debug.filter((item) => item.status === "unmatched")
        : [],
    [preview.response],
  );
  const predictions = preview.response?.task === "classification" ? preview.response.predictions : [];

  return (
    <section className="placeholder-card modal-media-preview-panel" data-testid="media-preview-panel">
      <div className="modal-media-preview-head">
        <div>
          <h4>{title}</h4>
          <p>{subtitle}</p>
        </div>
        {preview.response?.source_label ? (
          <div className="modal-media-preview-badges">
            <span className="modal-media-preview-badge">{preview.response.source_label}</span>
            {preview.response.device_selected ? (
              <span className="modal-media-preview-badge is-device">{preview.response.device_selected.toUpperCase()}</span>
            ) : null}
          </div>
        ) : null}
      </div>

      <div className={`modal-media-preview-stage${mediaReady ? " is-ready" : ""}`} ref={stageRef} data-testid="media-preview-stage">
        {children}
        {viewport && mediaReady
          ? matchedBoxes.map((box, index) => {
              const topLeft = toViewportCoords(box.bbox[0], box.bbox[1], viewport);
              const color = getClassColor(box.class_id || box.class_name || index);
              return (
                <div
                  key={`${box.class_id ?? box.class_name}-${index}`}
                  className="modal-media-preview-box"
                  data-testid="media-preview-overlay-box"
                  style={{
                    left: `${topLeft.x}px`,
                    top: `${topLeft.y}px`,
                    width: `${box.bbox[2] * viewport.scale}px`,
                    height: `${box.bbox[3] * viewport.scale}px`,
                    borderColor: color.overlayStroke,
                    background: `hsl(${color.hue} 85% 55% / 0.14)`,
                  }}
                >
                  <span
                    className="modal-media-preview-box-badge"
                    style={{
                      background: color.chipBackground,
                      borderColor: color.chipBorder,
                      color: color.chipText,
                    }}
                  >
                    {box.class_name} {formatConfidence(box.score)}
                  </span>
                </div>
              );
            })
          : null}
        {liveBadgeText ? <span className="modal-media-preview-live">{liveBadgeText}</span> : null}
        {preview.status === "loading" ? <span className="modal-media-preview-analysis">Analyzing preview...</span> : null}
        {!mediaReady ? <div className="modal-media-preview-empty">{emptyMessage}</div> : null}
      </div>

      <div className="modal-media-preview-results">
        {preview.error ? <p className="import-field-error">{preview.error.message}</p> : null}
        {preview.response?.task === "bbox" ? (
          <>
            <p>
              {matchedBoxes.length > 0
                ? `${matchedBoxes.length} AI overlay${matchedBoxes.length === 1 ? "" : "s"} ready on this preview frame.`
                : unmatchedDebug.length > 0
                  ? "No overlays matched the current task classes."
                  : preview.status === "ready"
                    ? "No detections met the current preview settings."
                    : "Preview a frame to see AI overlays here."}
            </p>
            {unmatchedDebug.length > 0 ? (
              <ul className="modal-media-preview-debug-list" data-testid="media-preview-debug-list">
                {unmatchedDebug.slice(0, 4).map((item, index) => (
                  <li key={`${item.label_text}-${index}`}>
                    {item.label_text} {formatConfidence(item.confidence)}
                  </li>
                ))}
              </ul>
            ) : null}
          </>
        ) : predictions.length > 0 ? (
          <div className="modal-media-preview-predictions" data-testid="media-preview-predictions">
            {predictions.map((item) => {
              const color = getClassColor(item.class_id);
              return (
                <span
                  key={item.class_id}
                  className="modal-media-preview-prediction"
                  data-testid="media-preview-classification-item"
                  style={{
                    background: color.chipBackground,
                    borderColor: color.chipBorder,
                    color: color.chipText,
                  }}
                >
                  {item.class_name} {formatConfidence(item.score)}
                </span>
              );
            })}
          </div>
        ) : (
          <p>{preview.status === "ready" ? "No predictions were returned for this preview frame." : "Preview a frame to inspect the active model output."}</p>
        )}
      </div>

      {metaItems.length > 0 ? (
        <ul className="modal-media-preview-meta">
          {metaItems.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
