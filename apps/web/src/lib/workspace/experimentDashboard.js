function asNumber(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function normalizeConfusion(matrix, mode = "none") {
  const rows = Array.isArray(matrix) ? matrix : [];
  if (mode === "none") {
    return rows.map((row) => (Array.isArray(row) ? row.map((value) => asNumber(value) ?? 0) : []));
  }

  if (mode === "by_true") {
    return rows.map((row) => {
      const normalized = Array.isArray(row) ? row.map((value) => asNumber(value) ?? 0) : [];
      const rowSum = normalized.reduce((sum, value) => sum + value, 0);
      if (rowSum <= 0) return normalized.map(() => 0);
      return normalized.map((value) => value / rowSum);
    });
  }

  if (mode === "by_pred") {
    const numeric = rows.map((row) => (Array.isArray(row) ? row.map((value) => asNumber(value) ?? 0) : []));
    const width = numeric.reduce((max, row) => Math.max(max, row.length), 0);
    const colSums = Array.from({ length: width }, (_, index) => numeric.reduce((sum, row) => sum + (row[index] ?? 0), 0));
    return numeric.map((row) =>
      row.map((value, colIndex) => {
        const colSum = colSums[colIndex] ?? 0;
        if (colSum <= 0) return 0;
        return value / colSum;
      }),
    );
  }

  return rows.map((row) => (Array.isArray(row) ? row.map((value) => asNumber(value) ?? 0) : []));
}

function filterPredictionRows(rows, options = {}) {
  const mode = options.mode ?? "misclassified";
  const trueClassIndex = typeof options.trueClassIndex === "number" ? options.trueClassIndex : null;
  const predClassIndex = typeof options.predClassIndex === "number" ? options.predClassIndex : null;
  const limit = typeof options.limit === "number" && Number.isFinite(options.limit) ? Math.max(1, Math.floor(options.limit)) : 100;

  const filtered = [];
  for (const row of Array.isArray(rows) ? rows : []) {
    const trueIdx = asNumber(row?.true_class_index);
    const predIdx = asNumber(row?.pred_class_index);
    const confidence = asNumber(row?.confidence);
    if (trueIdx == null || predIdx == null || confidence == null) continue;
    if (mode === "misclassified" && trueIdx === predIdx) continue;
    if (mode === "lowest_confidence_correct" && trueIdx !== predIdx) continue;
    if (mode === "highest_confidence_wrong" && trueIdx === predIdx) continue;
    if (trueClassIndex != null && trueIdx !== trueClassIndex) continue;
    if (predClassIndex != null && predIdx !== predClassIndex) continue;
    filtered.push(row);
  }

  if (mode === "lowest_confidence_correct") {
    filtered.sort((a, b) => (asNumber(a?.confidence) ?? 0) - (asNumber(b?.confidence) ?? 0));
  } else {
    filtered.sort((a, b) => (asNumber(b?.confidence) ?? 0) - (asNumber(a?.confidence) ?? 0));
  }
  return filtered.slice(0, limit);
}

function dashboardTabsForTask(task) {
  if (task === "detection") {
    return [
      { key: "loss", label: "Loss" },
      { key: "map", label: "mAP" },
      { key: "runtime", label: "Runtime" },
    ];
  }
  if (task === "classification") {
    return [
      { key: "loss", label: "Loss" },
      { key: "accuracy", label: "Accuracy" },
      { key: "prf", label: "F1 / Precision / Recall" },
    ];
  }
  return [];
}

function dashboardSeriesForTask(task, tab) {
  if (task === "detection") {
    if (tab === "loss") {
      return [
        { key: "train_loss", label: "train loss", color: "#cc6f36" },
      ];
    }
    if (tab === "map") {
      return [
        { key: "val_map", label: "val mAP@50", color: "#2f6fca" },
        { key: "val_map_50_95", label: "val mAP@50:95", color: "#2f9d58" },
      ];
    }
    if (tab === "runtime") {
      return [
        { key: "epoch_seconds", label: "epoch seconds", color: "#5b6fd1" },
        { key: "eta_seconds", label: "eta seconds", color: "#c96262" },
      ];
    }
    return [];
  }

  if (task === "classification") {
    if (tab === "loss") {
      return [
        { key: "train_loss", label: "train loss", color: "#cc6f36" },
        { key: "val_loss", label: "val loss", color: "#c96262" },
      ];
    }
    if (tab === "accuracy") {
      return [
        { key: "train_accuracy", label: "train accuracy", color: "#2f9d58" },
        { key: "val_accuracy", label: "val accuracy", color: "#2f6fca" },
      ];
    }
    if (tab === "prf") {
      return [
        { key: "val_macro_f1", label: "val macro f1", color: "#2f6fca" },
        { key: "val_macro_precision", label: "val macro precision", color: "#2f9d58" },
        { key: "val_macro_recall", label: "val macro recall", color: "#cc6f36" },
      ];
    }
    return [];
  }

  return [];
}

module.exports = {
  dashboardSeriesForTask,
  dashboardTabsForTask,
  normalizeConfusion,
  filterPredictionRows,
};
