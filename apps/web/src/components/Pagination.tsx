import { useEffect, useMemo, useRef, useState } from "react";
import { buildPageTokens, estimateMaxVisiblePages } from "../lib/workspace/pagination";

interface PaginationProps {
  total: number;
  current: number;
  onSelect: (index: number) => void;
  statuses?: Array<"labeled" | "unlabeled">;
  dirtyFlags?: boolean[];
}

export function Pagination({ total, current, onSelect, statuses, dirtyFlags }: PaginationProps) {
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
    return estimateMaxVisiblePages(total, containerWidth);
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
        const isDirty = Boolean(dirtyFlags?.[pageIndex]);
        const statusTitle = statuses?.[pageIndex] === "labeled" ? "Labeled" : "Unlabeled";
        const title = isDirty ? `${statusTitle} (staged)` : statusTitle;
        return (
          <button
            key={page}
            type="button"
            onClick={() => onSelect(pageIndex)}
            className={`page-chip ${statusClass}${isDirty ? " is-dirty" : ""}${isActive ? " active" : ""}`}
            title={title}
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
