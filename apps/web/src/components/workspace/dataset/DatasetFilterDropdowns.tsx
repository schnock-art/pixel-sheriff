"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type { AnnotationStatus } from "../../../lib/api";
import {
  ALL_STATUSES,
  buildDescendantsByPath,
  buildFolderTree,
  folderCheckState,
  toggleFolderPathSelection,
  toggleStatusSelection,
} from "../../../lib/workspace/datasetPage";

interface FolderTreeNode {
  name: string;
  path: string;
  children: FolderTreeNode[];
}

function FolderTreeRow({
  node,
  depth,
  selectedPaths,
  descendantsByPath,
  collapsed,
  onToggleCollapsed,
  onToggleChecked,
}: {
  node: FolderTreeNode;
  depth: number;
  selectedPaths: string[];
  descendantsByPath: Record<string, string[]>;
  collapsed: Record<string, boolean>;
  onToggleCollapsed: (path: string) => void;
  onToggleChecked: (path: string, checked: boolean) => void;
}) {
  const checkState = folderCheckState(node.path, selectedPaths, descendantsByPath);
  const isCollapsed = Boolean(collapsed[node.path]);
  const checkboxRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!checkboxRef.current) return;
    checkboxRef.current.indeterminate = checkState === "indeterminate";
  }, [checkState]);

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 6, paddingLeft: depth * 12 }}>
        {node.children.length > 0 ? (
          <button
            type="button"
            className="ghost-button"
            style={{ width: 22, height: 22, padding: 0 }}
            onClick={() => onToggleCollapsed(node.path)}
            aria-label={isCollapsed ? "Expand folder" : "Collapse folder"}
          >
            {isCollapsed ? ">" : "v"}
          </button>
        ) : (
          <span style={{ width: 22, display: "inline-block", flexShrink: 0 }} />
        )}
        <input
          ref={checkboxRef}
          type="checkbox"
          checked={checkState === "checked"}
          onChange={(event) => onToggleChecked(node.path, event.target.checked)}
        />
        <span>{node.name}</span>
      </div>
      {!isCollapsed && node.children.length > 0
        ? node.children.map((child) => (
            <FolderTreeRow
              key={child.path}
              node={child}
              depth={depth + 1}
              selectedPaths={selectedPaths}
              descendantsByPath={descendantsByPath}
              collapsed={collapsed}
              onToggleCollapsed={onToggleCollapsed}
              onToggleChecked={onToggleChecked}
            />
          ))
        : null}
    </div>
  );
}

export function FolderMultiSelectDropdown({
  label,
  folderPaths,
  selectedPaths,
  opposingSelectedPaths,
  onSelectedChange,
  onOpposingChange,
}: {
  label: string;
  folderPaths: string[];
  selectedPaths: string[];
  opposingSelectedPaths: string[];
  onSelectedChange: (value: string[]) => void;
  onOpposingChange: (value: string[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const containerRef = useRef<HTMLDivElement>(null);
  const tree = useMemo(() => buildFolderTree(folderPaths), [folderPaths]);
  const descendantsByPath = useMemo(() => buildDescendantsByPath(folderPaths), [folderPaths]);

  useEffect(() => {
    if (!open) return;
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [open]);

  function toggleChecked(path: string, checked: boolean) {
    const next = toggleFolderPathSelection({
      selectedPaths,
      opposingSelectedPaths,
      path,
      checked,
      descendantsByPath,
    });
    onSelectedChange(next.selectedPaths);
    onOpposingChange(next.opposingSelectedPaths);
  }

  return (
    <div ref={containerRef} style={{ position: "relative" }}>
      <button type="button" className="ghost-button" onClick={() => setOpen((value) => !value)}>
        {label}: {selectedPaths.length} selected
      </button>
      {open ? (
        <div
          style={{
            position: "absolute",
            zIndex: 20,
            top: "calc(100% + 6px)",
            left: 0,
            right: 0,
            maxHeight: 260,
            overflow: "auto",
            border: "1px solid var(--line, #d8dce6)",
            borderRadius: 8,
            background: "var(--frame, #f8f9fc)",
            padding: 8,
            boxShadow: "0 6px 16px rgba(0,0,0,0.12)",
          }}
        >
          {tree.length === 0 ? <p style={{ margin: 0 }}>No folders found.</p> : null}
          {tree.map((node) => (
            <FolderTreeRow
              key={node.path}
              node={node}
              depth={0}
              selectedPaths={selectedPaths}
              descendantsByPath={descendantsByPath}
              collapsed={collapsed}
              onToggleCollapsed={(path) => setCollapsed((previous) => ({ ...previous, [path]: !previous[path] }))}
              onToggleChecked={toggleChecked}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function StatusMultiSelectDropdown({
  label,
  selected,
  otherSelected,
  onSelectedChange,
  onOtherSelectedChange,
}: {
  label: string;
  selected: AnnotationStatus[];
  otherSelected: AnnotationStatus[];
  onSelectedChange: (value: AnnotationStatus[]) => void;
  onOtherSelectedChange: (value: AnnotationStatus[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [open]);

  return (
    <div ref={containerRef} style={{ position: "relative" }}>
      <button type="button" className="ghost-button" onClick={() => setOpen((value) => !value)}>
        {label}: {selected.length === 0 ? "none" : selected.join(", ")}
      </button>
      {open ? (
        <div
          style={{
            position: "absolute",
            zIndex: 20,
            top: "calc(100% + 6px)",
            left: 0,
            right: 0,
            border: "1px solid var(--line, #d8dce6)",
            borderRadius: 8,
            background: "var(--frame, #f8f9fc)",
            padding: 8,
            boxShadow: "0 6px 16px rgba(0,0,0,0.12)",
          }}
        >
          <div style={{ display: "grid", gap: 6 }}>
            {ALL_STATUSES.map((status) => (
              <label key={status} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <input
                  type="checkbox"
                  checked={selected.includes(status as AnnotationStatus)}
                  onChange={(event) => {
                    const next = toggleStatusSelection({
                      selected,
                      otherSelected,
                      status: status as AnnotationStatus,
                      checked: event.target.checked,
                    });
                    onSelectedChange(next.selected as AnnotationStatus[]);
                    onOtherSelectedChange(next.otherSelected as AnnotationStatus[]);
                  }}
                />
                <span>{status}</span>
              </label>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
