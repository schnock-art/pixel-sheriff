function buildQuery(params) {
  const query = params.toString();
  return query ? `?${query}` : "";
}

function buildDatasetVersionsPath(projectId, taskId) {
  const params = new URLSearchParams();
  if (taskId) params.set("task_id", taskId);
  return `/projects/${projectId}/datasets/versions${buildQuery(params)}`;
}

function buildDatasetVersionAssetsPath(projectId, datasetVersionId, options = {}) {
  const params = new URLSearchParams();
  if (typeof options.page === "number" && Number.isFinite(options.page) && options.page >= 1) {
    params.set("page", String(Math.floor(options.page)));
  }
  if (typeof options.page_size === "number" && Number.isFinite(options.page_size) && options.page_size >= 1) {
    params.set("page_size", String(Math.floor(options.page_size)));
  }
  if (options.split) params.set("split", options.split);
  if (options.status) params.set("status", options.status);
  if (options.class_id) params.set("class_id", options.class_id);
  if (options.search) params.set("search", options.search);
  return `/projects/${projectId}/datasets/versions/${datasetVersionId}/assets${buildQuery(params)}`;
}

function buildExperimentListPath(projectId, options = {}) {
  const params = new URLSearchParams();
  if (options.modelId) params.set("model_id", options.modelId);
  return `/projects/${projectId}/experiments${buildQuery(params)}`;
}

function buildExperimentAnalyticsPath(projectId, options = {}) {
  const params = new URLSearchParams();
  if (typeof options.maxPoints === "number" && Number.isFinite(options.maxPoints) && options.maxPoints >= 1) {
    params.set("max_points", String(Math.floor(options.maxPoints)));
  }
  return `/projects/${projectId}/experiments/analytics${buildQuery(params)}`;
}

function buildExperimentLogsPath(projectId, experimentId, options = {}) {
  const params = new URLSearchParams();
  if (typeof options.fromByte === "number" && Number.isFinite(options.fromByte) && options.fromByte >= 0) {
    params.set("from_byte", String(Math.floor(options.fromByte)));
  }
  if (typeof options.maxBytes === "number" && Number.isFinite(options.maxBytes) && options.maxBytes >= 1) {
    params.set("max_bytes", String(Math.floor(options.maxBytes)));
  }
  return `/projects/${projectId}/experiments/${experimentId}/logs${buildQuery(params)}`;
}

function buildExperimentSamplesPath(projectId, experimentId, options) {
  const params = new URLSearchParams();
  params.set("mode", options.mode);
  if (typeof options.trueClassIndex === "number" && Number.isFinite(options.trueClassIndex) && options.trueClassIndex >= 0) {
    params.set("true_class_index", String(Math.floor(options.trueClassIndex)));
  }
  if (typeof options.predClassIndex === "number" && Number.isFinite(options.predClassIndex) && options.predClassIndex >= 0) {
    params.set("pred_class_index", String(Math.floor(options.predClassIndex)));
  }
  if (typeof options.limit === "number" && Number.isFinite(options.limit) && options.limit >= 1) {
    params.set("limit", String(Math.floor(options.limit)));
  }
  return `/projects/${projectId}/experiments/${experimentId}/samples?${params.toString()}`;
}

function buildExperimentEventsUrl(apiBase, projectId, experimentId, options = {}) {
  const base = apiBase.endsWith("/") ? apiBase.slice(0, -1) : apiBase;
  const params = new URLSearchParams();
  if (typeof options.fromLine === "number" && Number.isFinite(options.fromLine) && options.fromLine >= 0) {
    params.set("from_line", String(Math.floor(options.fromLine)));
  }
  if (typeof options.attempt === "number" && Number.isFinite(options.attempt) && options.attempt >= 1) {
    params.set("attempt", String(Math.floor(options.attempt)));
  }
  return `${base}/api/v1/projects/${encodeURIComponent(projectId)}/experiments/${encodeURIComponent(experimentId)}/events${buildQuery(params)}`;
}

module.exports = {
  buildDatasetVersionsPath,
  buildDatasetVersionAssetsPath,
  buildExperimentListPath,
  buildExperimentAnalyticsPath,
  buildExperimentLogsPath,
  buildExperimentSamplesPath,
  buildExperimentEventsUrl,
};
