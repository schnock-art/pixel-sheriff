interface DatasetItem {
  id: string;
  name: string;
}

interface AssetGridProps {
  datasets: DatasetItem[];
  selectedDatasetId: string | null;
  onSelectDataset: (id: string) => void;
}

export function AssetGrid({ datasets, selectedDatasetId, onSelectDataset }: AssetGridProps) {
  if (datasets.length === 0) {
    return (
      <section className="dataset-list" aria-label="Datasets">
        <p className="dataset-empty">No datasets yet. Use Import to add one.</p>
      </section>
    );
  }

  return (
    <section className="dataset-list" aria-label="Datasets">
      <ul>
        {datasets.map((dataset) => (
          <li key={dataset.id}>
            <button
              type="button"
              onClick={() => onSelectDataset(dataset.id)}
              className={dataset.id === selectedDatasetId ? "dataset-item active" : "dataset-item"}
            >
              <span className="folder-icon" aria-hidden />
              {dataset.name}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
