function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalizeImageBasis(imageBasis) {
  if (!imageBasis || typeof imageBasis !== "object") return null;
  const width = Number(imageBasis.width);
  const height = Number(imageBasis.height);
  if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) {
    return null;
  }
  return { width: Math.round(width), height: Math.round(height) };
}

function resolveImageBasis(preferredImageBasis, fallbackImageBasis) {
  return normalizeImageBasis(preferredImageBasis) ?? normalizeImageBasis(fallbackImageBasis);
}

function computeImageViewport(containerWidth, containerHeight, imageWidth, imageHeight) {
  if (containerWidth <= 0 || containerHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) {
    return null;
  }
  const scale = Math.min(containerWidth / imageWidth, containerHeight / imageHeight);
  const width = imageWidth * scale;
  const height = imageHeight * scale;
  const offsetX = (containerWidth - width) / 2;
  const offsetY = (containerHeight - height) / 2;
  return { scale, width, height, offsetX, offsetY };
}

function toImageCoords(clientX, clientY, viewport, imageWidth, imageHeight) {
  const x = (clientX - viewport.offsetX) / viewport.scale;
  const y = (clientY - viewport.offsetY) / viewport.scale;
  return {
    x: clamp(x, 0, imageWidth),
    y: clamp(y, 0, imageHeight),
  };
}

function toViewportCoords(imageX, imageY, viewport) {
  return {
    x: viewport.offsetX + imageX * viewport.scale,
    y: viewport.offsetY + imageY * viewport.scale,
  };
}

function bboxFromPoints(startPoint, endPoint) {
  const left = Math.min(startPoint.x, endPoint.x);
  const top = Math.min(startPoint.y, endPoint.y);
  const right = Math.max(startPoint.x, endPoint.x);
  const bottom = Math.max(startPoint.y, endPoint.y);
  return [left, top, Math.max(0, right - left), Math.max(0, bottom - top)];
}

function bboxContainsPoint(bbox, x, y, padding = 0) {
  if (!Array.isArray(bbox) || bbox.length !== 4) return false;
  return (
    x >= bbox[0] - padding &&
    y >= bbox[1] - padding &&
    x <= bbox[0] + bbox[2] + padding &&
    y <= bbox[1] + bbox[3] + padding
  );
}

function polygonArea(flatPoints) {
  if (!Array.isArray(flatPoints) || flatPoints.length < 6 || flatPoints.length % 2 !== 0) return 0;
  const pointCount = flatPoints.length / 2;
  let area = 0;
  for (let index = 0; index < pointCount; index += 1) {
    const nextIndex = (index + 1) % pointCount;
    const x1 = flatPoints[index * 2];
    const y1 = flatPoints[index * 2 + 1];
    const x2 = flatPoints[nextIndex * 2];
    const y2 = flatPoints[nextIndex * 2 + 1];
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area) / 2;
}

function polygonBounds(flatPoints) {
  if (!Array.isArray(flatPoints) || flatPoints.length < 6 || flatPoints.length % 2 !== 0) return [0, 0, 0, 0];
  const xs = [];
  const ys = [];
  for (let index = 0; index < flatPoints.length; index += 2) {
    xs.push(flatPoints[index]);
    ys.push(flatPoints[index + 1]);
  }
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  return [minX, minY, Math.max(0, maxX - minX), Math.max(0, maxY - minY)];
}

function polygonContainsPoint(flatPoints, x, y) {
  if (!Array.isArray(flatPoints) || flatPoints.length < 6 || flatPoints.length % 2 !== 0) return false;
  let inside = false;
  let previousIndex = flatPoints.length - 2;
  for (let index = 0; index < flatPoints.length; index += 2) {
    const xi = flatPoints[index];
    const yi = flatPoints[index + 1];
    const xj = flatPoints[previousIndex];
    const yj = flatPoints[previousIndex + 1];
    const intersects = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-8) + xi;
    if (intersects) inside = !inside;
    previousIndex = index;
  }
  return inside;
}

function flattenPoints(points) {
  const flat = [];
  for (const point of points) {
    flat.push(point.x, point.y);
  }
  return flat;
}

function distanceBetweenPoints(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

module.exports = {
  clamp,
  normalizeImageBasis,
  resolveImageBasis,
  computeImageViewport,
  toImageCoords,
  toViewportCoords,
  bboxFromPoints,
  bboxContainsPoint,
  polygonArea,
  polygonBounds,
  polygonContainsPoint,
  flattenPoints,
  distanceBetweenPoints,
};
