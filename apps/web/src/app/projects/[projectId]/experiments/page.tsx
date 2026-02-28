interface ExperimentsPageProps {
  params: {
    projectId: string;
  };
}

export default function ExperimentsPage({ params }: ExperimentsPageProps) {
  const projectId = decodeURIComponent(params.projectId);

  return (
    <main className="workspace-shell project-page-shell">
      <section className="workspace-frame project-content-frame placeholder-page">
        <header className="project-section-header">
          <h2>Experiments</h2>
          <button type="button" className="primary-button" disabled title="Coming soon">
            + New Experiment
          </button>
        </header>
        <div className="placeholder-card">
          <h3>No experiments yet</h3>
          <p>Train a model to create reproducible experiment runs for this project ({projectId}).</p>
        </div>
      </section>
    </main>
  );
}

