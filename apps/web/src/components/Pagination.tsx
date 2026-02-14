import { useEffect, useMemo, useRef, useState } from "react";

interface PaginationProps {
  total: number;
  current: number;
  onSelect: (index: number) => void;
  statuses?: Array<"labeled" | "unlabeled">;
}

type PageToken = { type: "page"; page: number } | { type: "ellipsis"; key: string };

function buildPageTokens(total: number, current: number, maxVisiblePages: number): PageToken[] {
  if (total <= 0) return [];
  if (total <= maxVisiblePages) {
    return Array.from({ length: total }, (_, index) => ({ type: "page", page: index + 1 }));
  }

  const currentPage = current + 1;
  const interiorBudget = Math.max(maxVisiblePages - 2, 1);
  let start = Math.max(2, currentPage - Math.floor((interiorBudget - 1) / 2));
  let end = Math.min(total - 1, start + interiorBudget - 1);
  start = Math.max(2, end - interiorBudget + 1);

  const tokens: PageToken[] = [{ type: "page", page: 1 }];

  if (start > 2) {
    tokens.push({ type: "ellipsis", key: "left" });
  }

  for (let page = start; page <= end; page += 1) {
    tokens.push({ type: "page", page });
  }

  if (end < total - 1) {
    tokens.push({ type: "ellipsis", key: "right" });
  }

  tokens.push({ type: "page", page: total });
  return tokens;
}

export function Pagination({ total, current, onSelect, statuses }: PaginationProps) {
  const navRef = useRef<HTMLElement | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);

  useEffect(() => {
    const node = navRef.current;
    if (!node) return;

    const measure = () => setContainerWidth(node.clientWidth);
    measure();

    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", measure);
      return () => window.removeEventListener("resize", measure);
    }

    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const maxVisiblePages = useMemo(() => {
    if (total <= 0) return 0;
    if (containerWidth <= 0) return Math.min(total, 11);

    const estimatedChipWidth = 38;
    const availableForChips = Math.max(containerWidth - 28, 0);
    const chipCapacity = Math.floor(availableForChips / estimatedChipWidth);
    return Math.max(7, Math.min(total, chipCapacity));
  }, [containerWidth, total]);

  const tokens = buildPageTokens(total, current, maxVisiblePages);

  return (
    <nav ref={navRef} className="viewer-pagination" aria-label="Asset pages">
      <button type="button" className="page-chip page-jump" onClick={() => onSelect(0)} disabled={current === 0}>
        First
      </button>
      {tokens.map((token) => {
        if (token.type === "ellipsis") {
          return (
            <span key={token.key} className="page-ellipsis" aria-hidden="true">
              ...
            </span>
          );
        }

        const page = token.page;
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
      <button type="button" className="page-chip page-jump" onClick={() => onSelect(total - 1)} disabled={current >= total - 1}>
        Last
      </button>
    </nav>
  );
}
