interface PaginationProps {
  total: number;
  current: number;
  onSelect: (index: number) => void;
}

export function Pagination({ total, current, onSelect }: PaginationProps) {
  const pages = Array.from({ length: total }, (_, index) => index + 1);

  return (
    <nav className="viewer-pagination" aria-label="Asset pages">
      {pages.map((page) => (
        <button key={page} type="button" onClick={() => onSelect(page - 1)} className={page - 1 === current ? "page-chip active" : "page-chip"}>
          {page}
        </button>
      ))}
    </nav>
  );
}
