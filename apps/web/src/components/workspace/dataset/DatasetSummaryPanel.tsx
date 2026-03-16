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
    <section className="placeholder-card dataset-side-panel dataset-summary-panel" data-testid="dataset-summary-panel">
      <div className="dataset-panel-head">
        <h3>{summarySource === "saved" ? "Saved Summary" : "Draft Summary"}</h3>
      </div>
      {!summaryData ? <p>{summarySource === "saved" ? "Select a dataset version." : "Run preview to compute counts."}</p> : null}
      {summaryData ? (
        <div className="dataset-summary-stack">
          <p>Total: {summaryData.total}</p>
          <p>
            Splits: train {summaryData.split_counts.train} | val {summaryData.split_counts.val} | test {summaryData.split_counts.test}
          </p>
          <div>
            <h4>Class Distribution</h4>
            <div className="dataset-class-distribution">
              {Object.entries(summaryData.class_counts).map(([classId, count]) => (
                <div key={classId} className="dataset-class-row">
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
