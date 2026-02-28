import Link from "next/link";

import { readModelSummary } from "../../lib/workspace/modelSummary";

interface ModelBuilderSkeletonProps {
  title: string;
  backHref?: string | null;
  backLabel?: string;
  modelName?: string | null;
  config?: Record<string, unknown> | null;
  isLoading?: boolean;
  errorMessage?: string | null;
}

const STEPS = ["Dataset", "Input", "Backbone", "Neck", "Head", "Loss", "Outputs", "Export"];

export function ModelBuilderSkeleton({
  title,
  backHref = null,
  backLabel = "Back to Models",
  modelName,
  config,
  isLoading = false,
  errorMessage = null,
}: ModelBuilderSkeletonProps) {
  const summary = readModelSummary(config ?? {});

  return (
    <main className="workspace-shell project-page-shell">
      <section className="workspace-frame project-content-frame">
        <header className="project-section-header">
          <h2>{title}</h2>
          {backHref ? (
            <Link href={backHref} className="ghost-button">
              {backLabel}
            </Link>
          ) : null}
        </header>
        <div className="model-builder-grid">
          <aside className="model-builder-steps">
            <h3>Builder Steps</h3>
            <ol>
              {STEPS.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ol>
          </aside>
          <section className="model-builder-center">
            <h3>Configuration Area</h3>
            <div className="placeholder-card">
              <p>Model Builder coming soon.</p>
              {isLoading ? <p>Loading model config...</p> : null}
              {errorMessage ? <p className="project-field-error">{errorMessage}</p> : null}
            </div>
          </section>
          <aside className="model-builder-summary">
            <h3>Model Summary</h3>
            <dl className="model-summary-list">
              <div>
                <dt>Name</dt>
                <dd>{modelName ?? "-"}</dd>
              </div>
              <div>
                <dt>Task</dt>
                <dd>{summary.task}</dd>
              </div>
              <div>
                <dt>Classes</dt>
                <dd>{summary.numClasses}</dd>
              </div>
              <div>
                <dt>Class Names</dt>
                <dd>{summary.classNamesText}</dd>
              </div>
              <div>
                <dt>Input</dt>
                <dd>{summary.inputSizeText}</dd>
              </div>
              <div>
                <dt>Resize</dt>
                <dd>{summary.resizePolicy}</dd>
              </div>
              <div>
                <dt>Normalization</dt>
                <dd>{summary.normalizationType}</dd>
              </div>
              <div>
                <dt>Architecture</dt>
                <dd>{summary.architectureFamily}</dd>
              </div>
              <div>
                <dt>Backbone</dt>
                <dd>{summary.backboneName}</dd>
              </div>
              <div>
                <dt>Neck</dt>
                <dd>{summary.neckType}</dd>
              </div>
              <div>
                <dt>Head</dt>
                <dd>{summary.headType}</dd>
              </div>
              <div>
                <dt>Output</dt>
                <dd>{summary.primaryOutputFormat}</dd>
              </div>
              <div>
                <dt>ONNX</dt>
                <dd>{summary.onnxEnabled ? `Enabled (opset ${summary.onnxOpset})` : "Disabled"}</dd>
              </div>
              <div>
                <dt>Dynamic</dt>
                <dd>
                  batch={summary.dynamicBatch ? "on" : "off"}, h/w={summary.dynamicHeightWidth ? "on" : "off"}
                </dd>
              </div>
            </dl>
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
