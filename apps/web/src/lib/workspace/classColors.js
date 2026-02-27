function normalizedHueForClassId(classId) {
  if (typeof classId !== "number" || !Number.isFinite(classId)) return 210;
  const prime = 57;
  const hue = Math.abs(Math.round(classId) * prime) % 360;
  return hue;
}

function getClassColor(classId) {
  const hue = normalizedHueForClassId(classId);
  return {
    hue,
    chipBackground: `hsl(${hue} 88% 95%)`,
    chipBorder: `hsl(${hue} 68% 58%)`,
    chipText: `hsl(${hue} 46% 28%)`,
    chipActiveBackground: `hsl(${hue} 92% 88%)`,
    overlayStroke: `hsl(${hue} 70% 48%)`,
    overlayFill: `hsl(${hue} 85% 55% / 0.22)`,
  };
}

module.exports = {
  normalizedHueForClassId,
  getClassColor,
};
