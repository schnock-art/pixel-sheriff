const test = require("node:test");
const assert = require("node:assert/strict");

const {
  bboxContainsPoint,
  bboxFromPoints,
  computeImageViewport,
  flattenPoints,
  polygonArea,
  polygonBounds,
  polygonContainsPoint,
  toImageCoords,
  toViewportCoords,
} = require("../src/lib/workspace/geometry.js");
const { buildAnnotationUpsertInput } = require("../src/lib/workspace/annotationSubmission.js");

test("geometry viewport conversion round-trips image and canvas coordinates", () => {
  const viewport = computeImageViewport(1000, 700, 500, 500);
  assert.ok(viewport);
  const imagePoint = toImageCoords(500, 350, viewport, 500, 500);
  const projected = toViewportCoords(imagePoint.x, imagePoint.y, viewport);
  assert.equal(Math.round(projected.x), 500);
  assert.equal(Math.round(projected.y), 350);
});

test("bbox and polygon helpers compute containment and area", () => {
  const bbox = bboxFromPoints({ x: 10, y: 20 }, { x: 40, y: 45 });
  assert.deepEqual(bbox, [10, 20, 30, 25]);
  assert.equal(bboxContainsPoint(bbox, 15, 30), true);
  assert.equal(bboxContainsPoint(bbox, 100, 100), false);

  const polygon = flattenPoints([
    { x: 0, y: 0 },
    { x: 10, y: 0 },
    { x: 10, y: 10 },
    { x: 0, y: 10 },
  ]);
  assert.equal(polygonArea(polygon), 100);
  assert.deepEqual(polygonBounds(polygon), [0, 0, 10, 10]);
  assert.equal(polygonContainsPoint(polygon, 5, 5), true);
  assert.equal(polygonContainsPoint(polygon, 20, 20), false);
});

test("annotation submission includes geometry payload and image basis", () => {
  const upsertInput = buildAnnotationUpsertInput({
    assetId: "asset-geo",
    currentStatus: "unlabeled",
    selectedLabelIds: [2],
    activeLabelRows: [{ id: 2, name: "car" }],
    imageBasis: { width: 640, height: 480 },
    objects: [{ id: "bbox-1", kind: "bbox", category_id: 2, bbox: [10, 10, 50, 20] }],
  });

  assert.notEqual(upsertInput, null);
  assert.equal(upsertInput.status, "labeled");
  assert.equal(upsertInput.payload_json.version, "2.0");
  assert.equal(upsertInput.payload_json.image_basis.width, 640);
  assert.equal(upsertInput.payload_json.objects.length, 1);
});
