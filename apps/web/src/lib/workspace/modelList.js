function asRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function deriveModelDatasetVersion(config, datasetVersionNameById) {
  const sourceDataset = asRecord(asRecord(config).source_dataset);
  const datasetVersionId = typeof sourceDataset.manifest_id === "string" && sourceDataset.manifest_id.trim()
    ? sourceDataset.manifest_id
    : null;

  return {
    datasetVersionId,
    datasetVersionName: datasetVersionId ? datasetVersionNameById[datasetVersionId] ?? datasetVersionId : "-",
    hasSourceDataset: Boolean(datasetVersionId),
  };
}

function experimentPriority(status) {
  if (status === "running") return 5;
  if (status === "queued") return 4;
  if (status === "completed") return 3;
  if (status === "failed") return 2;
  if (status === "canceled") return 1;
  return 0;
}

function latestExperimentForModel(experiments, modelId) {
  return experiments
    .filter((experiment) => experiment.model_id === modelId)
    .sort((left, right) => {
      const leftTime = Date.parse(left.updated_at ?? left.created_at ?? "");
      const rightTime = Date.parse(right.updated_at ?? right.created_at ?? "");
      if (Number.isFinite(rightTime) && Number.isFinite(leftTime) && rightTime !== leftTime) {
        return rightTime - leftTime;
      }
      return experimentPriority(right.status) - experimentPriority(left.status);
    })[0] ?? null;
}

function deriveModelStatus(experiments, modelId, hasSourceDataset) {
  const latest = latestExperimentForModel(experiments, modelId);
  if (!latest) return hasSourceDataset ? "ready" : "draft";
  if (latest.status === "queued" || latest.status === "running") return "training";
  if (latest.status === "completed") return "completed";
  if (latest.status === "failed" || latest.status === "canceled") return "failed";
  return hasSourceDataset ? "ready" : "draft";
}

module.exports = {
  deriveModelDatasetVersion,
  deriveModelStatus,
};
