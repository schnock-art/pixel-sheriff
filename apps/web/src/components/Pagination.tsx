interface PaginationProps {
  total: number;
  current: number;
  onSelect: (index: number) => void;
  statuses?: Array<"labeled" | "unlabeled">;
}

export function Pagination({ total, current, onSelect, statuses }: PaginationProps) {
  const pages = Array.from({ length: total }, (_, index) => index + 1);

  return (
    <nav className="viewer-pagination" aria-label="Asset pages">
      {pages.map((page) => {
        const pageIndex = page - 1;
        const isActive = pageIndex === current;
        const statusClass = statuses?.[pageIndex] === "labeled" ? "is-labeled" : "is-unlabeled";
        return (
          <button
            key={page}
            type="button"
            onClick={() => onSelect(pageIndex)}
            className={`page-chip ${statusClass}${isActive ? " active" : ""}`}
            title={statuses?.[pageIndex] === "labeled" ? "Labeled" : "Unlabeled"}
          >
            {page}
          </button>
        );
      })}
    </nav>
  );
}
