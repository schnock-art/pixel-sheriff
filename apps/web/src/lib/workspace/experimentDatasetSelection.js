function asString(value) {
  return typeof value === "string" ? value : "";
}

function buildDatasetVersionOptions(items, configDatasetVersionId) {
  const options = [];
  for (const item of Array.isArray(items) ? items : []) {
    const version = item && typeof item === "object" ? item.version : null;
    const id = asString(version && typeof version === "object" ? version.dataset_version_id : "");
    if (!id) continue;
    const name = asString(version && typeof version === "object" ? version.name : "") || id;
    options.push({ id, name });
  }

  const currentId = asString(configDatasetVersionId);
  if (currentId && !options.some((row) => row.id === currentId)) {
    options.unshift({ id: currentId, name: `${currentId} (missing)` });
  }
  return options;
}

module.exports = {
  buildDatasetVersionOptions,
};
