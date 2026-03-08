export type CanvasTool = "select" | "bbox" | "polygon" | "pan";

interface CanvasToolbarProps {
  annotationMode: "labels" | "bbox" | "segmentation";
  activeTool: CanvasTool;
  onSelectTool: (tool: CanvasTool) => void;
  onResetView: () => void;
}

function isToolEnabled(tool: CanvasTool, annotationMode: CanvasToolbarProps["annotationMode"]) {
  if (tool === "select") return true;
  if (tool === "pan") return false;
  if (annotationMode === "bbox") return tool === "bbox";
  if (annotationMode === "segmentation") return tool === "polygon";
  return false;
}

export function CanvasToolbar({ annotationMode, activeTool, onSelectTool, onResetView }: CanvasToolbarProps) {
  const tools: Array<{ id: CanvasTool; label: string }> = [
    { id: "select", label: "Select" },
    { id: "bbox", label: "Draw Box" },
    { id: "polygon", label: "Polygon" },
    { id: "pan", label: "Pan" },
  ];

  return (
    <div className="canvas-toolbar">
      <div className="canvas-toolbar-group">
        {tools.map((tool) => {
          const enabled = isToolEnabled(tool.id, annotationMode);
          return (
            <button
              key={tool.id}
              type="button"
              className={activeTool === tool.id ? "ghost-button active-toggle" : "ghost-button"}
              onClick={() => onSelectTool(tool.id)}
              disabled={!enabled}
            >
              {tool.label}
            </button>
          );
        })}
      </div>
      <div className="canvas-toolbar-group">
        <button type="button" className="ghost-button" onClick={onResetView}>
          Fit
        </button>
        <button type="button" className="ghost-button" onClick={onResetView}>
          Reset
        </button>
      </div>
    </div>
  );
}
