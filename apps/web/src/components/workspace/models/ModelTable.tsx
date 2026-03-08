import Link from "next/link";

export interface ModelTableRowView {
  id: string;
  href: string;
  name: string;
  taskLabel: string;
  datasetVersionName: string;
  backboneName: string;
  numClasses: number;
  status: "draft" | "ready" | "training" | "completed" | "failed";
  createdAt: string;
}

function formatDate(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "-";
  return parsed.toLocaleDateString();
}

export function ModelTable({ rows }: { rows: ModelTableRowView[] }) {
  return (
    <div className="models-table-wrap polished">
      <table className="models-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Task</th>
            <th>Dataset Version</th>
            <th>Backbone</th>
            <th>Classes</th>
            <th>Status</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>
                <Link href={row.href}>{row.name}</Link>
              </td>
              <td>{row.taskLabel}</td>
              <td>{row.datasetVersionName}</td>
              <td>{row.backboneName}</td>
              <td>{row.numClasses}</td>
              <td>
                <span className={`model-status-chip is-${row.status}`}>{row.status}</span>
              </td>
              <td>{formatDate(row.createdAt)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
