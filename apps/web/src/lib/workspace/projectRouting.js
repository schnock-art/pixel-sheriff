function normalizeSection(section) {
  return section === "datasets" || section === "models" || section === "experiments" ? section : "datasets";
}

function deriveProjectSectionFromPathname(pathname) {
  if (typeof pathname !== "string" || pathname.trim() === "") return "datasets";

  const segments = pathname.split("/").filter(Boolean);
  const projectsIndex = segments.indexOf("projects");
  if (projectsIndex < 0) return "datasets";
  const candidate = segments[projectsIndex + 2] ?? "datasets";
  return normalizeSection(candidate);
}

function buildProjectSectionHref(projectId, section) {
  if (typeof projectId !== "string" || projectId.trim() === "") {
    return "/projects";
  }
  const normalizedSection = normalizeSection(section);
  return `/projects/${encodeURIComponent(projectId)}/${normalizedSection}`;
}

function buildModelBuilderHref(projectId, modelId) {
  if (typeof modelId === "string" && modelId.trim() !== "") {
    return `/projects/${encodeURIComponent(projectId)}/models/${encodeURIComponent(modelId)}`;
  }
  return `/projects/${encodeURIComponent(projectId)}/models/new`;
}

module.exports = {
  normalizeSection,
  deriveProjectSectionFromPathname,
  buildProjectSectionHref,
  buildModelBuilderHref,
};

