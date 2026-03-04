import { useState } from "react";

import type { AnnotationStatus } from "../api";

export type DatasetPreviewViewMode = "list" | "grid";

export function useDatasetBrowserState() {
  const [selectedDatasetVersionId, setSelectedDatasetVersionIdState] = useState<string | null>(null);
  const [splitFilter, setSplitFilterState] = useState<"all" | "train" | "val" | "test">("all");
  const [statusFilter, setStatusFilterState] = useState<"all" | AnnotationStatus>("all");
  const [searchText, setSearchTextState] = useState("");
  const [page, setPage] = useState(1);
  const [viewMode, setViewMode] = useState<DatasetPreviewViewMode>("list");

  function setSelectedDatasetVersionId(value: string | null) {
    setSelectedDatasetVersionIdState(value);
    setPage(1);
  }

  function setSplitFilter(value: "all" | "train" | "val" | "test") {
    setSplitFilterState(value);
    setPage(1);
  }

  function setStatusFilter(value: "all" | AnnotationStatus) {
    setStatusFilterState(value);
    setPage(1);
  }

  function setSearchText(value: string) {
    setSearchTextState(value);
    setPage(1);
  }

  return {
    selectedDatasetVersionId,
    setSelectedDatasetVersionId,
    splitFilter,
    setSplitFilter,
    statusFilter,
    setStatusFilter,
    searchText,
    setSearchText,
    page,
    setPage,
    viewMode,
    setViewMode,
  };
}
