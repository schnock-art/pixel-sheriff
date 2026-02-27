export interface ImageViewport {
  scale: number;
  width: number;
  height: number;
  offsetX: number;
  offsetY: number;
}

export interface Point2D {
  x: number;
  y: number;
}

export function clamp(value: number, min: number, max: number): number;
export function computeImageViewport(
  containerWidth: number,
  containerHeight: number,
  imageWidth: number,
  imageHeight: number,
): ImageViewport | null;
export function toImageCoords(
  clientX: number,
  clientY: number,
  viewport: ImageViewport,
  imageWidth: number,
  imageHeight: number,
): Point2D;
export function toViewportCoords(imageX: number, imageY: number, viewport: ImageViewport): Point2D;
export function bboxFromPoints(startPoint: Point2D, endPoint: Point2D): number[];
export function bboxContainsPoint(bbox: number[], x: number, y: number, padding?: number): boolean;
export function polygonArea(flatPoints: number[]): number;
export function polygonBounds(flatPoints: number[]): number[];
export function polygonContainsPoint(flatPoints: number[], x: number, y: number): boolean;
export function flattenPoints(points: Point2D[]): number[];
export function distanceBetweenPoints(a: Point2D, b: Point2D): number;
