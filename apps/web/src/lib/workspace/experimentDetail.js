function asRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function cloneConfig(value) {
  return JSON.parse(JSON.stringify(value ?? {}));
}

function configValidation(config) {
  const issues = [];
  const optimizer = asRecord(config?.optimizer);
  const scheduler = asRecord(config?.scheduler);
  const schedulerParams = asRecord(scheduler.params);
  const advanced = asRecord(config?.advanced);
  const lr = Number(optimizer.lr);
  const weightDecay = Number(optimizer.weight_decay);
  const momentum = optimizer.momentum == null || optimizer.momentum === "" ? null : Number(optimizer.momentum);
  const epochs = Number(config?.epochs);
  const batchSize = Number(config?.batch_size);
  if (!Number.isFinite(lr) || lr <= 0) issues.push("Learning rate must be > 0");
  if (optimizer.weight_decay != null && optimizer.weight_decay !== "" && (!Number.isFinite(weightDecay) || weightDecay < 0)) {
    issues.push("Weight decay must be >= 0");
  }
  if (optimizer.type === "sgd" && momentum != null && (!Number.isFinite(momentum) || momentum < 0)) {
    issues.push("Momentum must be >= 0");
  }
  if (scheduler.type === "step") {
    const stepSize = Number(schedulerParams.step_size);
    const gamma = Number(schedulerParams.gamma);
    if (!Number.isInteger(stepSize) || stepSize < 1) issues.push("Step scheduler step size must be >= 1");
    if (!Number.isFinite(gamma) || gamma <= 0) issues.push("Step scheduler gamma must be > 0");
  }
  if (advanced.grad_clip_norm != null && advanced.grad_clip_norm !== "") {
    const gradClip = Number(advanced.grad_clip_norm);
    if (!Number.isFinite(gradClip) || gradClip < 0) issues.push("Gradient clip norm must be >= 0");
  }
  if (!Number.isFinite(epochs) || epochs < 1) issues.push("Epochs must be >= 1");
  if (!Number.isFinite(batchSize) || batchSize < 1) issues.push("Batch size must be >= 1");
  return { isValid: issues.length === 0, issues };
}

function formatDateTime(value, options = {}) {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "-";
  const formatOptions = options.timeZone ? { timeZone: options.timeZone } : undefined;
  return parsed.toLocaleString(options.locales, formatOptions);
}

function formatDurationSeconds(value) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "-";
  const total = Math.round(value);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

function formatEtaClock(value, options = {}) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "-";
  const finishAt = new Date((typeof options.nowMs === "number" ? options.nowMs : Date.now()) + (value * 1000));
  const formatOptions = {
    hour: "2-digit",
    minute: "2-digit",
    ...(options.timeZone ? { timeZone: options.timeZone } : {}),
  };
  return finishAt.toLocaleTimeString(options.locales, formatOptions);
}

function formatCheckpoint(checkpoint) {
  if (!checkpoint || checkpoint.epoch == null) return "Not available yet";
  const metricName = checkpoint.metric_name ?? "metric";
  const value = typeof checkpoint.value === "number" ? checkpoint.value.toFixed(4) : "-";
  return `epoch ${checkpoint.epoch} | ${metricName}: ${value}`;
}

function patchNumber(value) {
  if (typeof value !== "string" || value.trim() === "") return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return null;
  return parsed;
}

function asYesNo(value) {
  if (value === true) return "yes";
  if (value === false) return "no";
  return "-";
}

module.exports = {
  asRecord,
  cloneConfig,
  configValidation,
  formatDateTime,
  formatDurationSeconds,
  formatEtaClock,
  formatCheckpoint,
  patchNumber,
  asYesNo,
};
