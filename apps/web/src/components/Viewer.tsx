import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent, type PointerEvent } from "react";

import {
  bboxContainsPoint,
  bboxFromPoints,
  computeImageViewport,
  distanceBetweenPoints,
  flattenPoints,
  polygonContainsPoint,
  toImageCoords,
  toViewportCoords,
} from "../lib/workspace/geometry";
import { getClassColor } from "../lib/workspace/classColors";
import type { AnnotationMode } from "./LabelPanel";
import { Pagination } from "./Pagination";

interface ViewerAsset {
  id: string;
  uri: string;
  width: number | null;
  height: number | null;
}

interface GeometryBBoxObject {
  id: string;
  kind: "bbox";
  category_id: number;
  bbox: number[];
}

interface GeometryPolygonObject {
  id: string;
  kind: "polygon";
  category_id: number;
  segmentation: number[][];
}

type GeometryObject = GeometryBBoxObject | GeometryPolygonObject;
type BBoxHandle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

interface ImageBasis {
  width: number;
  height: number;
}

interface ViewerProps {
  currentAsset: ViewerAsset | null;
  totalAssets: number;
  currentIndex: number;
  pageStatuses?: Array<"labeled" | "unlabeled">;
  pageDirtyFlags?: boolean[];
  annotationMode: AnnotationMode;
  geometryObjects: GeometryObject[];
  selectedObjectId: string | null;
  hoveredObjectId: string | null;
  defaultCategoryId: number | null;
  onSelectObject: (objectId: string | null) => void;
  onHoverObject: (objectId: string | null) => void;
  onUpsertObject: (object: GeometryObject) => void;
  onDeleteSelectedObject: () => void;
  onImageBasisChange: (imageBasis: ImageBasis | null) => void;
  onSelectIndex: (index: number) => void;
  onPrev: () => void;
  onNext: () => void;
}

function createObjectId(prefix: string) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function clampBBoxWithinImage(bbox: number[], imageWidth: number, imageHeight: number): number[] {
  const width = Math.max(1, Math.min(bbox[2], imageWidth));
  const height = Math.max(1, Math.min(bbox[3], imageHeight));
  const maxX = Math.max(0, imageWidth - width);
  const maxY = Math.max(0, imageHeight - height);
  const x = Math.min(maxX, Math.max(0, bbox[0]));
  const y = Math.min(maxY, Math.max(0, bbox[1]));
  return [x, y, width, height];
}

function getBBoxHandlePoints(bbox: number[]): Array<{ handle: BBoxHandle; x: number; y: number }> {
  const x = bbox[0];
  const y = bbox[1];
  const right = bbox[0] + bbox[2];
  const bottom = bbox[1] + bbox[3];
  const midX = x + bbox[2] / 2;
  const midY = y + bbox[3] / 2;
  return [
    { handle: "nw", x, y },
    { handle: "n", x: midX, y },
    { handle: "ne", x: right, y },
    { handle: "e", x: right, y: midY },
    { handle: "se", x: right, y: bottom },
    { handle: "s", x: midX, y: bottom },
    { handle: "sw", x, y: bottom },
    { handle: "w", x, y: midY },
  ];
}

function resolveBBoxHandle(
  point: { x: number; y: number },
  bbox: number[],
  thresholdInImageUnits: number,
): BBoxHandle | null {
  const handles = getBBoxHandlePoints(bbox);
  let best: { handle: BBoxHandle; distance: number } | null = null;
  for (const handle of handles) {
    const distance = distanceBetweenPoints(point, handle);
    if (distance > thresholdInImageUnits) continue;
    if (!best || distance < best.distance) {
      best = { handle: handle.handle, distance };
    }
  }
  return best?.handle ?? null;
}

function resizeBBoxFromHandle(
  original: number[],
  handle: BBoxHandle,
  point: { x: number; y: number },
  imageBasis: ImageBasis | null,
  minSize: number = 2,
): number[] {
  const originX = original[0];
  const originY = original[1];
  const originRight = original[0] + original[2];
  const originBottom = original[1] + original[3];

  let left = originX;
  let top = originY;
  let right = originRight;
  let bottom = originBottom;

  if (handle === "nw" || handle === "w" || handle === "sw") {
    const maxLeft = originRight - minSize;
    left = Math.min(point.x, maxLeft);
  }
  if (handle === "ne" || handle === "e" || handle === "se") {
    const minRight = originX + minSize;
    right = Math.max(point.x, minRight);
  }
  if (handle === "nw" || handle === "n" || handle === "ne") {
    const maxTop = originBottom - minSize;
    top = Math.min(point.y, maxTop);
  }
  if (handle === "sw" || handle === "s" || handle === "se") {
    const minBottom = originY + minSize;
    bottom = Math.max(point.y, minBottom);
  }

  if (imageBasis) {
    left = Math.max(0, left);
    top = Math.max(0, top);
    right = Math.min(imageBasis.width, right);
    bottom = Math.min(imageBasis.height, bottom);

    if (right - left < minSize) {
      if (handle === "nw" || handle === "w" || handle === "sw") left = right - minSize;
      else right = left + minSize;
    }
    if (bottom - top < minSize) {
      if (handle === "nw" || handle === "n" || handle === "ne") top = bottom - minSize;
      else bottom = top + minSize;
    }

    left = Math.max(0, left);
    top = Math.max(0, top);
    right = Math.min(imageBasis.width, right);
    bottom = Math.min(imageBasis.height, bottom);
  }

  return [left, top, Math.max(minSize, right - left), Math.max(minSize, bottom - top)];
}

export function Viewer({
  currentAsset,
  totalAssets,
  currentIndex,
  pageStatuses,
  pageDirtyFlags,
  annotationMode,
  geometryObjects,
  selectedObjectId,
  hoveredObjectId,
  defaultCategoryId,
  onSelectObject,
  onHoverObject,
  onUpsertObject,
  onDeleteSelectedObject,
  onImageBasisChange,
  onSelectIndex,
  onPrev,
  onNext,
}: ViewerProps) {
  const hasImage = Boolean(currentAsset?.uri);
  const maxIndex = Math.max(totalAssets - 1, 0);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });
  const [naturalSize, setNaturalSize] = useState<ImageBasis | null>(null);
  const [bboxDraft, setBboxDraft] = useState<{ start: { x: number; y: number }; end: { x: number; y: number } } | null>(null);
  const [polygonDraft, setPolygonDraft] = useState<Array<{ x: number; y: number }>>([]);
  const [activePointerId, setActivePointerId] = useState<number | null>(null);
  const [bboxMoveState, setBboxMoveState] = useState<{
    objectId: string;
    pointerId: number;
    anchor: { x: number; y: number };
    original: number[];
  } | null>(null);
  const [bboxResizeState, setBboxResizeState] = useState<{
    objectId: string;
    pointerId: number;
    handle: BBoxHandle;
    original: number[];
  } | null>(null);

  const imageBasis = useMemo(() => {
    if (currentAsset && typeof currentAsset.width === "number" && currentAsset.width > 0 && typeof currentAsset.height === "number" && currentAsset.height > 0) {
      return { width: currentAsset.width, height: currentAsset.height };
    }
    return naturalSize;
  }, [currentAsset, naturalSize]);

  const viewport = useMemo(() => {
    if (!imageBasis) return null;
    return computeImageViewport(canvasSize.width, canvasSize.height, imageBasis.width, imageBasis.height);
  }, [canvasSize.height, canvasSize.width, imageBasis]);

  useEffect(() => {
    setBboxDraft(null);
    setPolygonDraft([]);
    setActivePointerId(null);
    setBboxMoveState(null);
    setBboxResizeState(null);
  }, [currentAsset?.id, annotationMode]);

  useEffect(() => {
    if (!imageBasis) onImageBasisChange(null);
    else onImageBasisChange(imageBasis);
  }, [imageBasis, onImageBasisChange]);

  useEffect(() => {
    function measure() {
      if (!canvasRef.current) return;
      const rect = canvasRef.current.getBoundingClientRect();
      setCanvasSize({ width: rect.width, height: rect.height });
    }
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [currentAsset?.id]);

  function jump(delta: number) {
    const nextIndex = Math.min(maxIndex, Math.max(0, currentIndex + delta));
    onSelectIndex(nextIndex);
  }

  function toImagePoint(event: PointerEvent<SVGSVGElement>) {
    if (!viewport || !imageBasis) return null;
    const svgRect = event.currentTarget.getBoundingClientRect();
    return toImageCoords(event.clientX - svgRect.left, event.clientY - svgRect.top, viewport, imageBasis.width, imageBasis.height);
  }

  function hitTestObject(point: { x: number; y: number }): string | null {
    for (let index = geometryObjects.length - 1; index >= 0; index -= 1) {
      const object = geometryObjects[index];
      if (object.kind === "bbox" && bboxContainsPoint(object.bbox, point.x, point.y, 2)) {
        return object.id;
      }
      if (object.kind === "polygon") {
        for (const segment of object.segmentation) {
          if (polygonContainsPoint(segment, point.x, point.y)) return object.id;
        }
      }
    }
    return null;
  }

  const finalizePolygon = useCallback(() => {
    if (annotationMode !== "segmentation") return;
    if (!imageBasis || !defaultCategoryId) return;
    if (polygonDraft.length < 3) return;
    const object: GeometryPolygonObject = {
      id: createObjectId("poly"),
      kind: "polygon",
      category_id: defaultCategoryId,
      segmentation: [flattenPoints(polygonDraft)],
    };
    onUpsertObject(object);
    onSelectObject(object.id);
    setPolygonDraft([]);
  }, [annotationMode, defaultCategoryId, imageBasis, onSelectObject, onUpsertObject, polygonDraft]);

  useEffect(() => {
    if (annotationMode === "labels") return;

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setBboxDraft(null);
        setPolygonDraft([]);
        setBboxMoveState(null);
        setBboxResizeState(null);
        onHoverObject(null);
        return;
      }
      if ((event.key === "Delete" || event.key === "Backspace") && selectedObjectId) {
        event.preventDefault();
        onDeleteSelectedObject();
        return;
      }
      if (annotationMode === "segmentation" && event.key === "Enter") {
        event.preventDefault();
        finalizePolygon();
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [annotationMode, finalizePolygon, onDeleteSelectedObject, onHoverObject, selectedObjectId]);

  function handlePointerDown(event: PointerEvent<SVGSVGElement>) {
    if (annotationMode === "labels" || !imageBasis || !viewport) return;
    if (event.button !== 0) return;
    const point = toImagePoint(event);
    if (!point) return;

    if (annotationMode === "bbox") {
      if (selectedObjectId) {
        const selectedObject = geometryObjects.find(
          (object): object is GeometryBBoxObject => object.id === selectedObjectId && object.kind === "bbox",
        );
        if (selectedObject) {
          const handleThreshold = viewport ? 10 / viewport.scale : 10;
          const resizeHandle = resolveBBoxHandle(point, selectedObject.bbox, handleThreshold);
          if (resizeHandle) {
            try {
              event.currentTarget.setPointerCapture(event.pointerId);
              setActivePointerId(event.pointerId);
            } catch {
              setActivePointerId(null);
            }
            setBboxResizeState({
              objectId: selectedObject.id,
              pointerId: event.pointerId,
              handle: resizeHandle,
              original: selectedObject.bbox.slice(),
            });
            setBboxMoveState(null);
            setBboxDraft(null);
            onSelectObject(selectedObject.id);
            onHoverObject(selectedObject.id);
            return;
          }
        }
      }

      const hitObjectId = hitTestObject(point);
      if (hitObjectId) {
        onSelectObject(hitObjectId);
        onHoverObject(hitObjectId);
        const hitObject = geometryObjects.find((object) => object.id === hitObjectId);
        if (hitObject?.kind === "bbox") {
          try {
            event.currentTarget.setPointerCapture(event.pointerId);
            setActivePointerId(event.pointerId);
          } catch {
            setActivePointerId(null);
          }
          setBboxMoveState({
            objectId: hitObjectId,
            pointerId: event.pointerId,
            anchor: point,
            original: hitObject.bbox.slice(),
          });
        }
        setBboxResizeState(null);
        setBboxDraft(null);
        return;
      }
      setBboxMoveState(null);
      setBboxResizeState(null);
      setBboxDraft({ start: point, end: point });
      try {
        event.currentTarget.setPointerCapture(event.pointerId);
        setActivePointerId(event.pointerId);
      } catch {
        setActivePointerId(null);
      }
      onSelectObject(null);
      onHoverObject(null);
      return;
    }

    if (annotationMode === "segmentation") {
      if (event.detail > 1 && polygonDraft.length >= 3) {
        finalizePolygon();
        return;
      }

      if (polygonDraft.length === 0) {
        const hitObjectId = hitTestObject(point);
        if (hitObjectId) {
          onSelectObject(hitObjectId);
          onHoverObject(hitObjectId);
          return;
        }
      }

      if (polygonDraft.length >= 3) {
        const firstPoint = polygonDraft[0];
        const closeThreshold = viewport ? 18 / viewport.scale : 18;
        if (distanceBetweenPoints(firstPoint, point) <= closeThreshold) {
          finalizePolygon();
          return;
        }
      }

      setPolygonDraft((previous) => [...previous, point]);
      onSelectObject(null);
      onHoverObject(null);
    }
  }

  function handlePointerMove(event: PointerEvent<SVGSVGElement>) {
    if (annotationMode === "labels") return;
    const point = toImagePoint(event);
    if (!point) return;

    if (bboxResizeState && annotationMode === "bbox") {
      if (event.pointerId !== bboxResizeState.pointerId) return;
      const objectToResize = geometryObjects.find((object) => object.id === bboxResizeState.objectId);
      if (!objectToResize || objectToResize.kind !== "bbox") return;

      const resizedBbox = resizeBBoxFromHandle(
        bboxResizeState.original,
        bboxResizeState.handle,
        point,
        imageBasis,
        2,
      );

      onUpsertObject({ ...objectToResize, bbox: resizedBbox });
      return;
    }

    if (bboxMoveState && annotationMode === "bbox") {
      if (event.pointerId !== bboxMoveState.pointerId) return;
      const objectToMove = geometryObjects.find((object) => object.id === bboxMoveState.objectId);
      if (!objectToMove || objectToMove.kind !== "bbox") return;

      const deltaX = point.x - bboxMoveState.anchor.x;
      const deltaY = point.y - bboxMoveState.anchor.y;
      const movedBbox = [
        bboxMoveState.original[0] + deltaX,
        bboxMoveState.original[1] + deltaY,
        bboxMoveState.original[2],
        bboxMoveState.original[3],
      ];
      const clamped = imageBasis
        ? clampBBoxWithinImage(movedBbox, imageBasis.width, imageBasis.height)
        : movedBbox;

      onUpsertObject({ ...objectToMove, bbox: clamped });
      return;
    }

    if (bboxDraft && annotationMode === "bbox") {
      if (activePointerId !== null && event.pointerId !== activePointerId) return;
      setBboxDraft((previous) => (previous ? { ...previous, end: point } : previous));
      return;
    }

    const hitObjectId = hitTestObject(point);
    onHoverObject(hitObjectId);
  }

  function handlePointerUp(event: PointerEvent<SVGSVGElement>) {
    if (annotationMode !== "bbox") return;
    if (activePointerId !== null && event.pointerId !== activePointerId) return;

    if (bboxResizeState) {
      if (event.pointerId !== bboxResizeState.pointerId) return;
      if (activePointerId !== null) {
        try {
          event.currentTarget.releasePointerCapture(activePointerId);
        } catch {
          // Ignore release errors when capture is already gone.
        }
      }
      setActivePointerId(null);
      setBboxResizeState(null);
      return;
    }

    if (bboxMoveState) {
      if (event.pointerId !== bboxMoveState.pointerId) return;
      if (activePointerId !== null) {
        try {
          event.currentTarget.releasePointerCapture(activePointerId);
        } catch {
          // Ignore release errors when capture is already gone.
        }
      }
      setActivePointerId(null);
      setBboxMoveState(null);
      return;
    }

    if (!bboxDraft) return;

    if (activePointerId !== null) {
      try {
        event.currentTarget.releasePointerCapture(activePointerId);
      } catch {
        // Ignore release errors when capture is already gone.
      }
      setActivePointerId(null);
    }

    if (!defaultCategoryId) {
      setBboxDraft(null);
      return;
    }
    const bbox = bboxFromPoints(bboxDraft.start, bboxDraft.end);
    setBboxDraft(null);
    if (bbox[2] < 2 || bbox[3] < 2) return;
    const object: GeometryBBoxObject = {
      id: createObjectId("bbox"),
      kind: "bbox",
      category_id: defaultCategoryId,
      bbox,
    };
    onUpsertObject(object);
    onSelectObject(object.id);
  }

  function handlePointerCancel(event: PointerEvent<SVGSVGElement>) {
    if (activePointerId !== null) {
      try {
        event.currentTarget.releasePointerCapture(activePointerId);
      } catch {
        // Ignore release errors when capture is already gone.
      }
    }
    setActivePointerId(null);
    setBboxDraft(null);
    setBboxMoveState(null);
    setBboxResizeState(null);
  }

  function handleDoubleClick(event: MouseEvent<SVGSVGElement>) {
    if (annotationMode !== "segmentation") return;
    if (polygonDraft.length < 3) return;
    event.preventDefault();
    finalizePolygon();
  }

  function renderGeometryObject(object: GeometryObject) {
    const isSelected = object.id === selectedObjectId;
    const isHovered = object.id === hoveredObjectId;
    const className = `geometry-shape${isSelected ? " is-selected" : ""}${isHovered ? " is-hovered" : ""}`;
    const color = getClassColor(object.category_id);
    const overlayStyle = {
      fill: `hsl(${color.hue} 85% 55% / ${isSelected ? "0.34" : isHovered ? "0.28" : "0.22"})`,
      stroke: color.overlayStroke,
      strokeWidth: isSelected ? 3 : isHovered ? 2.5 : 2,
    };

    if (object.kind === "bbox") {
      const topLeft = toViewportCoords(object.bbox[0], object.bbox[1], viewport!);
      const width = object.bbox[2] * viewport!.scale;
      const height = object.bbox[3] * viewport!.scale;
      const handles = getBBoxHandlePoints(object.bbox).map((handlePoint) => ({
        ...handlePoint,
        viewport: toViewportCoords(handlePoint.x, handlePoint.y, viewport!),
      }));
      return (
        <g key={object.id} className={className}>
          <rect x={topLeft.x} y={topLeft.y} width={width} height={height} style={overlayStyle} />
          {annotationMode === "bbox" && isSelected
            ? handles.map((handlePoint) => (
                <rect
                  key={`${object.id}-${handlePoint.handle}`}
                  className={`geometry-handle geometry-handle-${handlePoint.handle}`}
                  x={handlePoint.viewport.x - 4}
                  y={handlePoint.viewport.y - 4}
                  width={8}
                  height={8}
                  rx={2}
                  ry={2}
                />
              ))
            : null}
        </g>
      );
    }

    return object.segmentation.map((segment, index) => {
      const points = [];
      for (let pointIndex = 0; pointIndex < segment.length; pointIndex += 2) {
        const point = toViewportCoords(segment[pointIndex], segment[pointIndex + 1], viewport!);
        points.push(`${point.x},${point.y}`);
      }
      return (
        <g key={`${object.id}-${index}`} className={className}>
          <polygon points={points.join(" ")} style={overlayStyle} />
        </g>
      );
    });
  }

  function renderDraft() {
    if (!viewport) return null;
    const draftColor = getClassColor(defaultCategoryId ?? 1);
    if (annotationMode === "bbox" && bboxDraft) {
      const bbox = bboxFromPoints(bboxDraft.start, bboxDraft.end);
      const topLeft = toViewportCoords(bbox[0], bbox[1], viewport);
      return (
        <g className="geometry-shape is-draft">
          <rect
            x={topLeft.x}
            y={topLeft.y}
            width={bbox[2] * viewport.scale}
            height={bbox[3] * viewport.scale}
            style={{ fill: `hsl(${draftColor.hue} 85% 55% / 0.14)`, stroke: draftColor.overlayStroke }}
          />
        </g>
      );
    }

    if (annotationMode === "segmentation" && polygonDraft.length > 0) {
      const points = polygonDraft.map((point) => {
        const projected = toViewportCoords(point.x, point.y, viewport);
        return `${projected.x},${projected.y}`;
      });
      return (
        <g className="geometry-shape is-draft">
          <polyline points={points.join(" ")} style={{ fill: "none", stroke: draftColor.overlayStroke }} />
          {points.map((point, index) => {
            const [x, y] = point.split(",");
            return <circle key={`${point}-${index}`} cx={x} cy={y} r={3} style={{ fill: draftColor.overlayStroke }} />;
          })}
        </g>
      );
    }
    return null;
  }

  return (
    <section className="viewer-panel" aria-label="Image viewer">
      <div className={hasImage ? "viewer-canvas has-image" : "viewer-canvas"} role="img" aria-label="Traffic scene with annotations" ref={canvasRef}>
        {currentAsset?.uri ? (
          <img
            src={currentAsset.uri}
            alt={`Asset ${currentIndex + 1}`}
            className="viewer-image"
            onLoad={(event) => {
              const image = event.currentTarget;
              if (image.naturalWidth > 0 && image.naturalHeight > 0) {
                setNaturalSize({ width: image.naturalWidth, height: image.naturalHeight });
              }
            }}
          />
        ) : null}
        {hasImage && viewport ? (
          <svg
            className={annotationMode === "labels" ? "viewer-geometry-overlay is-readonly" : "viewer-geometry-overlay"}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerCancel}
            onDoubleClick={handleDoubleClick}
            onPointerLeave={() => onHoverObject(null)}
          >
            {geometryObjects.map((object) => renderGeometryObject(object))}
            {renderDraft()}
          </svg>
        ) : null}
        <div className="skyline" />
        <div className="road" />
        <div className="car car-main" />
        <div className="car car-left" />
        <div className="car car-right" />
      </div>
      {annotationMode === "bbox" && bboxDraft ? (
        <p className="viewer-draft-warning" role="status" aria-live="polite">
          Draft box not saved yet. Release mouse to create the object. Drag boxes to move; drag corner/edge handles to resize.
        </p>
      ) : null}
      {annotationMode === "segmentation" && polygonDraft.length > 0 ? (
        <p className="viewer-draft-warning" role="status" aria-live="polite">
          Draft polygon not saved yet. Click near the first point, double-click, or press Enter to close.
        </p>
      ) : null}

      <div className="viewer-controls">
        <Pagination
          total={Math.max(totalAssets, 1)}
          current={Math.max(currentIndex, 0)}
          onSelect={onSelectIndex}
          statuses={pageStatuses}
          dirtyFlags={pageDirtyFlags}
        />
        <div className="viewer-nav">
          <button type="button" className="ghost-icon-button" aria-label="Back 10 frames" onClick={() => jump(-10)}>
            -10
          </button>
          <button type="button" className="ghost-icon-button" aria-label="Back 5 frames" onClick={() => jump(-5)}>
            -5
          </button>
          <button type="button" className="ghost-icon-button" aria-label="Previous frame" onClick={onPrev}>
            {"<"}
          </button>
          <button type="button" className="ghost-icon-button" aria-label="Next frame" onClick={onNext}>
            {">"}
          </button>
          <button type="button" className="ghost-icon-button" aria-label="Forward 5 frames" onClick={() => jump(5)}>
            +5
          </button>
          <button type="button" className="ghost-icon-button" aria-label="Forward 10 frames" onClick={() => jump(10)}>
            +10
          </button>
        </div>
      </div>
    </section>
  );
}
