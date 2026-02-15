const IMAGE_EXTENSIONS = new Set(["jpg", "jpeg", "png", "gif", "bmp", "webp", "tif", "tiff"]);

function isImageCandidate(file) {
  if (typeof file.type === "string" && file.type.toLowerCase().startsWith("image/")) return true;
  const extension = (file.name || "").split(".").pop()?.toLowerCase() ?? "";
  return IMAGE_EXTENSIONS.has(extension);
}

function buildTargetRelativePath(file, targetFolder) {
  const normalizedFolder = targetFolder.replaceAll("\\", "/").replace(/^\/+|\/+$/g, "");
  const relative = (file.webkitRelativePath || file.name).replaceAll("\\", "/");
  const parts = relative.split("/").filter(Boolean);
  const remainder = file.webkitRelativePath ? parts.slice(1).join("/") : file.name;
  return `${normalizedFolder}/${remainder || file.name}`;
}

module.exports = {
  isImageCandidate,
  buildTargetRelativePath,
};
