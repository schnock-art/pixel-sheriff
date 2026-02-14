interface FiltersProps {
  query: string;
  onQueryChange: (value: string) => void;
}

export function Filters({ query, onQueryChange }: FiltersProps) {
  return (
    <section className="dataset-filters" aria-label="Dataset filters">
      <div className="search-row">
        <span className="search-icon" aria-hidden />
        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          className="search-input"
          placeholder="Folder"
          aria-label="Search datasets"
        />
        <button type="button" className="tiny-button" aria-label="More options">
          ...
        </button>
      </div>
    </section>
  );
}
