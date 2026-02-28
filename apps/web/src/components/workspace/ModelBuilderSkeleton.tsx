interface ModelBuilderSkeletonProps {
  title: string;
}

const STEPS = ["Dataset", "Input", "Backbone", "Neck", "Head", "Loss", "Outputs", "Export"];

export function ModelBuilderSkeleton({ title }: ModelBuilderSkeletonProps) {
  return (
    <main className="workspace-shell project-page-shell">
      <section className="workspace-frame project-content-frame">
        <header className="project-section-header">
          <h2>{title}</h2>
        </header>
        <div className="model-builder-grid">
          <aside className="model-builder-steps">
            <h3>Builder Steps</h3>
            <ol>
              {STEPS.map((step, index) => (
                <li key={step}>{index + 1}. {step}</li>
              ))}
            </ol>
          </aside>
          <section className="model-builder-center">
            <h3>Configuration Area</h3>
            <div className="placeholder-card">
              <p>Model configuration editor will be added in a follow-up phase.</p>
            </div>
          </section>
          <aside className="model-builder-summary">
            <h3>Model Summary</h3>
            <ul>
              <li>Task: Placeholder</li>
              <li>Classes: Placeholder</li>
              <li>Params: Placeholder</li>
              <li>Est VRAM: Placeholder</li>
              <li>ONNX: Placeholder</li>
            </ul>
          </aside>
        </div>
        <footer className="model-builder-footer">
          <button type="button" className="ghost-button" disabled>
            Save
          </button>
          <button type="button" className="primary-button" disabled>
            Train Model
          </button>
        </footer>
      </section>
    </main>
  );
}

