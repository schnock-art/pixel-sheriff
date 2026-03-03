function normalizedHueForClassId(classId) {
  if (typeof classId === "number" && Number.isFinite(classId)) {
    const prime = 57;
    return Math.abs(Math.round(classId) * prime) % 360;
  }
  if (typeof classId !== "string" || classId.trim() === "") return 210;
  let hash = 0;
  for (let index = 0; index < classId.length; index += 1) {
    hash = ((hash << 5) - hash + classId.charCodeAt(index)) | 0;
  }
  const hue = Math.abs(hash) % 360;
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
