import Link from "next/link";
import type { ReactNode } from "react";

import { readModelSummary } from "../../lib/workspace/modelSummary";
import { ProjectSectionLayout } from "./project-shell/ProjectSectionLayout";

interface ModelBuilderSkeletonProps {
  title: string;
  backHref?: string | null;
  backLabel?: string;
  modelName?: string | null;
  datasetVersionName?: string | null;
  config?: Record<string, unknown> | null;
  isLoading?: boolean;
  errorMessage?: string | null;
  editorContent?: ReactNode;
  isDirty?: boolean;
  isValid?: boolean;
  isSaving?: boolean;
  onSave?: (() => void) | null;
  saveDisabled?: boolean;
  saveError?: string | null;
  validationPanel?: ReactNode;
  onTrainModel?: (() => void) | null;
  trainDisabled?: boolean;
  trainButtonLabel?: string;
}

const STEPS = ["Dataset", "Input", "Backbone", "Neck", "Head", "Loss", "Outputs", "Export"];

export function ModelBuilderSkeleton({
  title,
  backHref = null,
  backLabel = "Back to Models",
  modelName,
  datasetVersionName = null,
  config,
  isLoading = false,
  errorMessage = null,
  editorContent = null,
  isDirty = false,
  isValid = true,
  isSaving = false,
  onSave = null,
  saveDisabled = true,
  saveError = null,
  validationPanel = null,
  onTrainModel = null,
  trainDisabled = true,
  trainButtonLabel = "Train Model",
}: ModelBuilderSkeletonProps) {
  const summary = readModelSummary(config ?? {});

  return (
    <ProjectSectionLayout
      title={title}
      actions={
        backHref ? (
          <Link href={backHref} className="ghost-button">
            {backLabel}
          </Link>
        ) : null
      }
    >
        <div className="model-builder-grid">
          <aside className="model-builder-steps">
            <h3>Builder Steps</h3>
            <ol>
              {STEPS.map((step) => (
                <li key={step}>
                  <a href={`#model-step-${step.toLowerCase()}`}>{step}</a>
                </li>
              ))}
            </ol>
          </aside>
          <section className="model-builder-center">
            <div className="model-builder-center-head">
              <h3>Configuration Area</h3>
              {isDirty ? <span className="model-builder-unsaved">Unsaved changes</span> : null}
            </div>
            {isLoading ? (
              <div className="placeholder-card">
                <p>Loading model config...</p>
              </div>
            ) : (
              <div className="model-builder-editor">{editorContent ?? <p className="labels-empty">No editable controls configured.</p>}</div>
            )}
            {errorMessage ? <p className="project-field-error">{errorMessage}</p> : null}
            {saveError ? <p className="project-field-error">{saveError}</p> : null}
            {!isValid ? <p className="project-field-error">Draft config is invalid. Fix validation issues before saving.</p> : null}
            {validationPanel ? <div className="model-builder-validation-panel">{validationPanel}</div> : null}
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
                <dt>Dataset Version</dt>
                <dd>{datasetVersionName ?? summary.datasetVersionId}</dd>
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
          <button type="button" className="ghost-button" disabled={saveDisabled || !onSave} onClick={onSave ? () => onSave() : undefined}>
            {isSaving ? "Saving..." : "Save"}
          </button>
          <button type="button" className="primary-button" disabled={trainDisabled || !onTrainModel} onClick={onTrainModel ? () => onTrainModel() : undefined}>
            {trainButtonLabel}
          </button>
        </footer>
    </ProjectSectionLayout>
  );
}
