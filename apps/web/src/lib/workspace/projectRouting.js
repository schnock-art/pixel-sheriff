function normalizeSection(section) {
  return section === "datasets" || section === "dataset" || section === "models" || section === "experiments" || section === "deploy"
    ? section
    : "datasets";
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

function buildModelCreateHref(projectId, options) {
  if (typeof projectId !== "string" || projectId.trim() === "") {
    return "/projects";
  }

  const searchParams = new URLSearchParams();
  if (options && typeof options === "object") {
    if (typeof options.taskId === "string" && options.taskId.trim() !== "") {
      searchParams.set("taskId", options.taskId);
    }
    if (typeof options.datasetVersionId === "string" && options.datasetVersionId.trim() !== "") {
      searchParams.set("datasetVersionId", options.datasetVersionId);
    }
  }

  const query = searchParams.toString();
  const base = `/projects/${encodeURIComponent(projectId)}/models/new`;
  return query ? `${base}?${query}` : base;
}

module.exports = {
  normalizeSection,
  deriveProjectSectionFromPathname,
  buildProjectSectionHref,
  buildModelBuilderHref,
  buildModelCreateHref,
};
