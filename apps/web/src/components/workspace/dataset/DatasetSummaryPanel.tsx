import type { DatasetSummaryPayload } from "../../../lib/workspace/datasetPage";

export function DatasetSummaryPanel({
  summarySource,
  summaryData,
  classDisplayName,
}: {
  summarySource: "saved" | "draft";
  summaryData: DatasetSummaryPayload | null;
  classDisplayName: (classId: string) => string;
}) {
  return (
    <section className="placeholder-card" data-testid="dataset-summary-panel">
      <h3>{summarySource === "saved" ? "Summary (Saved Version)" : "Summary (Draft Preview)"}</h3>
      {!summaryData ? <p>{summarySource === "saved" ? "Select a dataset version." : "Run preview to compute counts."}</p> : null}
      {summaryData ? (
        <div style={{ display: "grid", gap: 10 }}>
          <p>Total: {summaryData.total}</p>
          <p>
            Splits: train {summaryData.split_counts.train} | val {summaryData.split_counts.val} | test {summaryData.split_counts.test}
          </p>
          <div>
            <h4>Class Distribution</h4>
            <div style={{ display: "grid", gap: 6 }}>
              {Object.entries(summaryData.class_counts).map(([classId, count]) => (
                <div key={classId} style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 8 }}>
                  <span>{classDisplayName(classId)}</span>
                  <span>{count}</span>
                </div>
              ))}
            </div>
          </div>
          {summaryData.warnings.length > 0 ? (
            <div>
              <h4>Warnings</h4>
              <ul>
                {summaryData.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
