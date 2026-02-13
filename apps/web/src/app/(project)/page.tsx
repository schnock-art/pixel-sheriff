import { AssetGrid } from "../../components/AssetGrid";
import { Filters } from "../../components/Filters";
import { LabelPanel } from "../../components/LabelPanel";
import { Viewer } from "../../components/Viewer";

export default function ProjectPage() {
  return (
    <main>
      <h1>Project Workspace</h1>
      <Filters />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr 1fr", gap: 12 }}>
        <AssetGrid />
        <Viewer />
        <LabelPanel />
      </div>
    </main>
  );
}
