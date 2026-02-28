interface ExperimentDetailPageProps {
  params: {
    experimentId: string;
  };
}

export default function ExperimentDetailPage({ params }: ExperimentDetailPageProps) {
  const experimentId = decodeURIComponent(params.experimentId);

  return (
    <main className="workspace-shell project-page-shell">
      <section className="workspace-frame project-content-frame placeholder-page">
        <header className="project-section-header">
          <h2>Experiment: {experimentId}</h2>
          <button type="button" className="ghost-button" disabled>
            Download ONNX
          </button>
        </header>
        <div className="experiment-grid">
          <article className="placeholder-card"><h3>Training Curves</h3><p>Loss and metric charts will appear here.</p></article>
          <article className="placeholder-card"><h3>Confusion Matrix</h3><p>Class-level evaluation will appear here.</p></article>
          <article className="placeholder-card"><h3>Example Predictions</h3><p>Qualitative outputs will appear here.</p></article>
          <article className="placeholder-card"><h3>Best Checkpoint</h3><p>Best epoch and artifact info will appear here.</p></article>
        </div>
      </section>
    </main>
  );
}

