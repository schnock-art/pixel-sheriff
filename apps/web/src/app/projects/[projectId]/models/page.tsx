import Link from "next/link";

interface ModelsPageProps {
  params: {
    projectId: string;
  };
}

export default function ModelsPage({ params }: ModelsPageProps) {
  const projectId = decodeURIComponent(params.projectId);

  return (
    <main className="workspace-shell project-page-shell">
      <section className="workspace-frame project-content-frame placeholder-page">
        <header className="project-section-header">
          <h2>Models</h2>
          <Link href={`/projects/${encodeURIComponent(projectId)}/models/new`} className="primary-button">
            + New Model
          </Link>
        </header>
        <div className="placeholder-card">
          <h3>No models yet</h3>
          <p>Create a model to configure architecture, train experiments, and export deployment artifacts.</p>
        </div>
      </section>
    </main>
  );
}

