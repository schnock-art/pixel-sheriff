"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { ModelBuilderSkeleton } from "../../../../../components/workspace/ModelBuilderSkeleton";
import { useProjectNavigationGuard } from "../../../../../components/workspace/ProjectNavigationContext";
import {
  ApiError,
  createExperiment,
  getProjectModel,
  listDatasetVersions,
  listExperiments,
  updateProjectModel,
  type ProjectExperimentSummary,
} from "../../../../../lib/api";
import familiesMetadata from "../../../../../lib/metadata/families.v1.json";
import { validateModelConfigDraft } from "../../../../../lib/schema/validator";
import {
  cloneModelConfig,
  isModelConfigDirty,
  setArchitectureFamily,
  setBackbone,
  setDynamicShapeFlags,
  setEmbeddingAuxEnabled,
  setEmbeddingProjection,
  setSourceDataset,
  setSquareInputSize,
} from "../../../../../lib/workspace/modelConfigEditor";

interface ModelDetailPageProps {
  params: {
    projectId: string;
    modelId: string;
  };
}

type ModelConfig = Record<string, unknown>;

interface DatasetVersionRecord {
  id: string;
  name: string;
  task: string;
  label_mode?: "single_label" | "multi_label" | null;
  num_classes: number;
  class_order: string[];
  class_names: Record<string, string>;
}

interface FamilyInputSizeRule {
  shape?: string;
  mode?: string;
  min_square_size?: number;
  step?: number;
  recommended_square_size?: number;
  required_square_size?: number;
}

const INPUT_SIZE_PRESETS = [224, 320, 384, 512, 640] as const;
const RESIZE_POLICY_OPTIONS = ["letterbox", "stretch", "longest_side_pad"] as const;
const NORMALIZATION_OPTIONS = ["imagenet", "none", "custom"] as const;
const EMBEDDING_DIM_OPTIONS = [128, 256, 512] as const;
const EMBEDDING_NORMALIZE_OPTIONS = ["none", "l2"] as const;

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function parseApiErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof ApiError) {
    if (error.responseBody) {
      try {
        const parsed = JSON.parse(error.responseBody) as { error?: { message?: unknown } };
        const message = parsed.error?.message;
        if (typeof message === "string" && message.trim()) return message;
      } catch {
        return error.responseBody;
      }
    }
    return error.message;
  }
  return error instanceof Error ? error.message : fallback;
}

function normalizeModelTask(task: string | null | undefined): string | null {
  if (typeof task !== "string") return null;
  const normalized = task.trim().toLowerCase();
  if (!normalized) return null;
  if (normalized === "bbox" || normalized === "detection") return "detection";
  if (normalized === "classification_single" || normalized === "classification") return "classification";
  if (normalized === "segmentation") return "segmentation";
  return normalized;
}

function tasksMatch(left: string | null | undefined, right: string | null | undefined): boolean {
  const normalizedLeft = normalizeModelTask(left);
  const normalizedRight = normalizeModelTask(right);
  return normalizedLeft !== null && normalizedLeft === normalizedRight;
}

function getFamilyInputSizeRule(family: { input_size?: FamilyInputSizeRule } | null | undefined): FamilyInputSizeRule | null {
  if (!family?.input_size || typeof family.input_size !== "object") return null;
  return family.input_size;
}

function isAllowedSquareSize(rule: FamilyInputSizeRule | null, size: number): boolean {
  if (!Number.isFinite(size) || size < 1) return false;
  if (!rule || rule.shape !== "square") return true;
  if (rule.mode === "fixed") {
    return size === rule.required_square_size;
  }
  if (rule.mode === "range") {
    const minimum = typeof rule.min_square_size === "number" ? rule.min_square_size : 1;
    const step = typeof rule.step === "number" && rule.step > 0 ? rule.step : 1;
    return size >= minimum && (size - minimum) % step === 0;
  }
  return true;
}

function formatFamilyInputSizeHint(rule: FamilyInputSizeRule | null): string | null {
  if (!rule || rule.shape !== "square") return null;
  if (rule.mode === "fixed" && typeof rule.required_square_size === "number") {
    return `Required for this family: ${rule.required_square_size} x ${rule.required_square_size}.`;
  }
  if (rule.mode === "range") {
    const minimum = typeof rule.min_square_size === "number" ? rule.min_square_size : 1;
    const step = typeof rule.step === "number" && rule.step > 0 ? rule.step : 1;
    const recommended = typeof rule.recommended_square_size === "number" ? rule.recommended_square_size : null;
    const recommendedText = recommended ? ` Recommended: ${recommended} x ${recommended}.` : "";
    if (step === 1) {
      return `Allowed for this family: any square >= ${minimum}.${recommendedText}`;
    }
    return `Allowed for this family: square sizes >= ${minimum} in steps of ${step}.${recommendedText}`;
  }
  return null;
}

export default function ModelDetailPage({ params }: ModelDetailPageProps) {
  const router = useRouter();
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);
  const modelId = useMemo(() => decodeURIComponent(params.modelId), [params.modelId]);
  const { setHasUnsavedDrafts, guardedNavigate } = useProjectNavigationGuard();

  const [modelName, setModelName] = useState<string | null>(null);
  const [savedConfig, setSavedConfig] = useState<ModelConfig | null>(null);
  const [draftConfig, setDraftConfig] = useState<ModelConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [toastTone, setToastTone] = useState<"success" | "error">("success");
  const [trainError, setTrainError] = useState<string | null>(null);
  const [isLaunchingTrain, setIsLaunchingTrain] = useState(false);
  const [showTrainChoiceModal, setShowTrainChoiceModal] = useState(false);
  const [latestExperiment, setLatestExperiment] = useState<ProjectExperimentSummary | null>(null);
  const [allDatasetVersions, setAllDatasetVersions] = useState<DatasetVersionRecord[]>([]);

  const validation = useMemo(() => validateModelConfigDraft(draftConfig ?? {}), [draftConfig]);
  const isValid = validation.isValid;
  const isDirty = useMemo(() => isModelConfigDirty(savedConfig ?? {}, draftConfig ?? {}), [savedConfig, draftConfig]);

  useEffect(() => {
    setHasUnsavedDrafts(isDirty);
  }, [isDirty, setHasUnsavedDrafts]);

  useEffect(() => () => setHasUnsavedDrafts(false), [setHasUnsavedDrafts]);

  useEffect(() => {
    if (!toastMessage) return;
    const timeout = window.setTimeout(() => setToastMessage(null), 5000);
    return () => window.clearTimeout(timeout);
  }, [toastMessage]);

  useEffect(() => {
    let isMounted = true;

    async function loadModel() {
      setIsLoading(true);
      setErrorMessage(null);
      setSaveError(null);
      try {
        const nextRecord = await getProjectModel(projectId, modelId);
        if (!isMounted) return;
        const clonedConfig = cloneModelConfig(nextRecord.config_json);
        setModelName(nextRecord.name);
        setSavedConfig(clonedConfig);
        setDraftConfig(clonedConfig);
      } catch (error) {
        if (isMounted) setErrorMessage(parseApiErrorMessage(error, "Failed to load model"));
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }

    void loadModel();

    return () => {
      isMounted = false;
    };
  }, [modelId, projectId]);

  useEffect(() => {
    let isMounted = true;

    async function loadDatasetVersions() {
      try {
        const listed = await listDatasetVersions(projectId);
        if (!isMounted) return;
        const versions: DatasetVersionRecord[] = listed.items
          .map((envelope) => {
            const v = envelope.version as Record<string, unknown>;
            return {
              id: typeof v.dataset_version_id === "string" ? v.dataset_version_id : "",
              name: typeof v.name === "string" ? v.name : "",
              task: typeof v.task === "string" ? v.task : "",
              label_mode:
                typeof v.label_mode === "string" && (v.label_mode === "single_label" || v.label_mode === "multi_label")
                  ? (v.label_mode as "single_label" | "multi_label")
                  : null,
              num_classes: typeof v.num_classes === "number" ? v.num_classes : 0,
              class_order: Array.isArray(v.class_order) ? (v.class_order as string[]) : [],
              class_names: v.class_names && typeof v.class_names === "object" ? (v.class_names as Record<string, string>) : {},
            };
          })
          .filter((v) => v.id !== "");
        setAllDatasetVersions(versions);
      } catch {
        // non-fatal — selectors will just be empty
      }
    }

    void loadDatasetVersions();

    return () => {
      isMounted = false;
    };
  }, [projectId]);

  const input = asRecord(draftConfig?.input);
  const inputSize = Array.isArray(input.input_size) ? input.input_size : [];
  const inputSizeWidth = typeof inputSize[0] === "number" ? Math.floor(inputSize[0]) : 0;
  const inputSizeHeight = typeof inputSize[1] === "number" ? Math.floor(inputSize[1]) : 0;
  const isSquareInputSize = inputSizeWidth > 0 && inputSizeWidth === inputSizeHeight;
  const inputSizePresetValue =
    isSquareInputSize && INPUT_SIZE_PRESETS.includes(inputSizeWidth as (typeof INPUT_SIZE_PRESETS)[number])
      ? String(inputSizeWidth)
      : "custom";
  const customInputSizeValue = inputSizeWidth > 0 ? String(inputSizeWidth) : "";

  const normalization = asRecord(input.normalization);
  const normalizationType = typeof normalization.type === "string" ? normalization.type : "imagenet";
  const resizePolicy = typeof input.resize_policy === "string" ? input.resize_policy : "letterbox";

  const architecture = asRecord(draftConfig?.architecture);
  const backbone = asRecord(architecture.backbone);
  const backboneName = typeof backbone.name === "string" ? backbone.name : "resnet18";
  const pretrained = Boolean(backbone.pretrained);

  // Step 1 derived values
  const currentFamilyName = typeof architecture.family === "string" ? architecture.family : null;
  const currentManifestId =
    typeof asRecord(draftConfig?.source_dataset).manifest_id === "string"
      ? (asRecord(draftConfig?.source_dataset).manifest_id as string)
      : null;
  const currentVersionFromManifest = allDatasetVersions.find((v) => v.id === currentManifestId) ?? null;
  const currentFamilyFromMeta = familiesMetadata.families.find((f) => f.name === currentFamilyName) ?? null;
  const currentTask =
    currentVersionFromManifest?.task
    ?? allDatasetVersions.find((v) => tasksMatch(v.task, currentFamilyFromMeta?.task))?.task
    ?? currentFamilyFromMeta?.task
    ?? null;

  const uniqueTasks = Array.from(new Set(allDatasetVersions.map((v) => v.task))).filter(Boolean);
  const familiesForTask = currentTask
    ? familiesMetadata.families.filter((f) => tasksMatch(f.task, currentTask))
    : familiesMetadata.families;
  const versionsForTask = currentTask
    ? allDatasetVersions.filter((v) => tasksMatch(v.task, currentTask))
    : allDatasetVersions;
  const allowedBackbones = currentFamilyFromMeta?.allowed_backbones ?? [];
  const currentFamilyInputSizeRule = getFamilyInputSizeRule(currentFamilyFromMeta);
  const allowedPresetSizes = INPUT_SIZE_PRESETS.filter((value) => isAllowedSquareSize(currentFamilyInputSizeRule, value));
  const customInputAllowed = currentFamilyInputSizeRule?.mode !== "fixed";
  const inputSizeHelpText = formatFamilyInputSizeHint(currentFamilyInputSizeRule);
  const inputSizeIssue =
    validation.errors.find((issue) => issue.path === "$.input.input_size" && issue.keyword === "familyInputSize")
    ?? null;

  const outputs = asRecord(draftConfig?.outputs);
  const auxOutputs = Array.isArray(outputs.aux) ? outputs.aux : [];
  const embeddingAux = auxOutputs.find((item) => {
    const row = asRecord(item);
    return row.type === "embedding" && row.name === "embedding";
  });
  const embeddingAuxRecord = asRecord(embeddingAux);
  const embeddingProjection = asRecord(embeddingAuxRecord.projection);
  const embeddingEnabled = Boolean(embeddingAux);
  const embeddingOutDim = typeof embeddingProjection.out_dim === "number" ? Math.floor(embeddingProjection.out_dim) : 256;
  const embeddingNormalize = embeddingProjection.normalize === "none" ? "none" : "l2";

  const exportSpec = asRecord(draftConfig?.export);
  const onnx = asRecord(exportSpec.onnx);
  const dynamicShapes = asRecord(onnx.dynamic_shapes);
  const onnxEnabled = Boolean(onnx.enabled);
  const onnxOpset = typeof onnx.opset === "number" ? Math.floor(onnx.opset) : 17;
  const dynamicBatch = Boolean(dynamicShapes.batch);
  const dynamicHeightWidth = Boolean(dynamicShapes.height_width);

  function patchDraft(mutator: (next: ModelConfig) => void) {
    setDraftConfig((current) => {
      if (!current) return current;
      const next = cloneModelConfig(current);
      mutator(next);
      return next;
    });
  }

  async function handleSave() {
    if (!draftConfig) return;
    setIsSaving(true);
    setSaveError(null);
    try {
      const updated = await updateProjectModel(projectId, modelId, { config_json: draftConfig });
      const updatedConfig = cloneModelConfig(updated.config_json);
      setModelName(updated.name);
      setSavedConfig(updatedConfig);
      setDraftConfig(updatedConfig);
      setToastTone("success");
      setToastMessage("Model saved");
    } catch (error) {
      const message = parseApiErrorMessage(error, "Failed to save model");
      setSaveError(message);
      setToastTone("error");
      setToastMessage(`Save failed: ${message}`);
    } finally {
      setIsSaving(false);
    }
  }

  async function handleTrainModel() {
    setTrainError(null);
    setIsLaunchingTrain(true);
    try {
      const listed = await listExperiments(projectId, { modelId });
      const rows = listed.items ?? [];
      if (rows.length === 0) {
        const created = await createExperiment(projectId, { model_id: modelId });
        guardedNavigate(() => {
          router.push(`/projects/${encodeURIComponent(projectId)}/experiments/${encodeURIComponent(created.id)}`);
        });
        return;
      }
      const [latest] = rows;
      setLatestExperiment(latest ?? null);
      setShowTrainChoiceModal(true);
    } catch (error) {
      setTrainError(parseApiErrorMessage(error, "Failed to prepare train flow"));
    } finally {
      setIsLaunchingTrain(false);
    }
  }

  function handleContinueExperiment() {
    if (!latestExperiment) return;
    setShowTrainChoiceModal(false);
    guardedNavigate(() => {
      router.push(`/projects/${encodeURIComponent(projectId)}/experiments/${encodeURIComponent(latestExperiment.id)}`);
    });
  }

  async function handleNewRun() {
    setIsLaunchingTrain(true);
    setTrainError(null);
    try {
      const created = await createExperiment(projectId, { model_id: modelId });
      setShowTrainChoiceModal(false);
      guardedNavigate(() => {
        router.push(`/projects/${encodeURIComponent(projectId)}/experiments/${encodeURIComponent(created.id)}`);
      });
    } catch (error) {
      setTrainError(parseApiErrorMessage(error, "Failed to create experiment"));
    } finally {
      setIsLaunchingTrain(false);
    }
  }

  const editorContent = (
    <div className="model-builder-form-grid">
      <section className="model-builder-step" id="model-step-dataset">
        <h4>Step 1: Source</h4>
        <label className="project-field">
          <span>Task</span>
          <select
            value={currentTask ?? ""}
            onChange={(event) => {
              const nextTask = event.target.value;
              const nextVersions = allDatasetVersions.filter((v) => tasksMatch(v.task, nextTask));
              const nextVersion = nextVersions[0] ?? null;
              const nextFamilies = familiesMetadata.families.filter((f) => tasksMatch(f.task, nextTask));
              const nextFamily = nextFamilies[0] ?? null;
              setDraftConfig((current) => {
                if (!current) return current;
                let next = cloneModelConfig(current);
                if (nextVersion) {
                  next = setSourceDataset(next, {
                    id: nextVersion.id,
                    manifest_id: nextVersion.id,
                    task: nextVersion.task,
                    label_mode: nextVersion.label_mode,
                    num_classes: nextVersion.num_classes,
                    class_order: nextVersion.class_order,
                    class_names: nextVersion.class_names,
                  }) as ModelConfig;
                }
                if (nextFamily) {
                  next = setArchitectureFamily(next, nextFamily.name, familiesMetadata) as ModelConfig;
                }
                return next;
              });
            }}
          >
            {uniqueTasks.length === 0 ? (
              <option value="">No dataset versions</option>
            ) : (
              <>
                <option value="" disabled>Select a task</option>
                {uniqueTasks.map((task) => (
                  <option key={task} value={task}>
                    {task}
                  </option>
                ))}
              </>
            )}
          </select>
        </label>
        <label className="project-field">
          <span>Dataset Version</span>
          <select
            value={currentManifestId ?? ""}
            onChange={(event) => {
              const versionId = event.target.value;
              const version = allDatasetVersions.find((v) => v.id === versionId);
              if (!version) return;
              setDraftConfig((current) =>
                current
                  ? (setSourceDataset(current, {
                      id: version.id,
                      manifest_id: version.id,
                      task: version.task,
                      label_mode: version.label_mode,
                      num_classes: version.num_classes,
                      class_order: version.class_order,
                      class_names: version.class_names,
                    }) as ModelConfig)
                  : current,
              );
            }}
          >
            {versionsForTask.length === 0 ? (
              <option value="">No versions for this task</option>
            ) : (
              versionsForTask.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.name} ({v.task})
                </option>
              ))
            )}
          </select>
        </label>
        <label className="project-field">
          <span>Family</span>
          <select
            value={currentFamilyName ?? ""}
            onChange={(event) => {
              const nextFamilyName = event.target.value;
              setDraftConfig((current) =>
                current
                  ? (setArchitectureFamily(current, nextFamilyName, familiesMetadata) as ModelConfig)
                  : current,
              );
            }}
          >
            {familiesForTask.length === 0 ? (
              <option value="">No families for this task</option>
            ) : (
              familiesForTask.map((f) => (
                <option key={f.name} value={f.name}>
                  {f.name}
                </option>
              ))
            )}
          </select>
        </label>
      </section>

      <section className="model-builder-step" id="model-step-input">
        <h4>Step 2: Input</h4>
        <label className="project-field">
          <span>Input Size</span>
          <select
            value={inputSizePresetValue}
            onChange={(event) => {
              const value = event.target.value;
              if (value === "custom") return;
              const parsed = Number.parseInt(value, 10);
              if (!Number.isFinite(parsed) || parsed < 1) return;
              setDraftConfig((current) => (current ? setSquareInputSize(current, parsed) : current));
            }}
          >
            {allowedPresetSizes.map((value) => (
              <option key={value} value={value}>
                {value} x {value}
              </option>
            ))}
            {customInputAllowed || inputSizePresetValue === "custom" ? (
              <option value="custom" disabled={!customInputAllowed}>
                {customInputAllowed ? "Custom" : "Current size invalid"}
              </option>
            ) : null}
          </select>
        </label>
        {inputSizeHelpText ? <p className="project-field-help">{inputSizeHelpText}</p> : null}
        {inputSizeIssue ? <p className="project-field-error">{inputSizeIssue.message}</p> : null}
        {inputSizePresetValue === "custom" && customInputAllowed ? (
          <label className="project-field">
            <span>Custom Square Size</span>
            <input
              type="number"
              min={currentFamilyInputSizeRule?.min_square_size ?? 1}
              step={currentFamilyInputSizeRule?.step ?? 1}
              value={customInputSizeValue}
              onChange={(event) => {
                const parsed = Number.parseInt(event.target.value, 10);
                if (!Number.isFinite(parsed) || parsed < 1) return;
                setDraftConfig((current) => (current ? setSquareInputSize(current, parsed) : current));
              }}
            />
          </label>
        ) : null}
        <label className="project-field">
          <span>Resize Policy</span>
          <select
            value={resizePolicy}
            onChange={(event) => {
              patchDraft((next) => {
                const nextInput = asRecord(next.input);
                nextInput.resize_policy = event.target.value;
                next.input = nextInput;
              });
            }}
          >
            {RESIZE_POLICY_OPTIONS.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label className="project-field">
          <span>Normalization Type</span>
          <select
            value={normalizationType}
            onChange={(event) => {
              patchDraft((next) => {
                const nextInput = asRecord(next.input);
                const nextNormalization = asRecord(nextInput.normalization);
                nextNormalization.type = event.target.value;
                nextInput.normalization = nextNormalization;
                next.input = nextInput;
              });
            }}
          >
            {NORMALIZATION_OPTIONS.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
      </section>

      <section className="model-builder-step" id="model-step-backbone">
        <h4>Step 3: Backbone</h4>
        <label className="project-field">
          <span>Backbone Name</span>
          <select
            value={backboneName}
            onChange={(event) => {
              const nextBackboneName = event.target.value;
              setDraftConfig((current) =>
                current ? (setBackbone(current, nextBackboneName) as ModelConfig) : current,
              );
            }}
          >
            {(allowedBackbones.length > 0 ? allowedBackbones : [backboneName]).map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label className="model-builder-checkbox">
          <input
            type="checkbox"
            checked={pretrained}
            onChange={(event) => {
              patchDraft((next) => {
                const nextArchitecture = asRecord(next.architecture);
                const nextBackbone = asRecord(nextArchitecture.backbone);
                nextBackbone.pretrained = event.target.checked;
                nextArchitecture.backbone = nextBackbone;
                next.architecture = nextArchitecture;
              });
            }}
          />
          <span>Pretrained</span>
        </label>
      </section>

      <section className="model-builder-step" id="model-step-neck">
        <h4>Step 4: Neck</h4>
        <p className="project-field-help">Neck configuration is currently inherited from the selected architecture family. Family changes will keep this in sync.</p>
      </section>

      <section className="model-builder-step" id="model-step-head">
        <h4>Step 5: Head</h4>
        <p className="project-field-help">Head configuration is currently managed by the family preset and task-specific defaults.</p>
      </section>

      <section className="model-builder-step" id="model-step-loss">
        <h4>Step 6: Loss</h4>
        <p className="project-field-help">Loss settings are not exposed as editable controls yet. Current defaults are preserved when you save this model draft.</p>
      </section>

      <section className="model-builder-step" id="model-step-outputs">
        <h4>Step 7: Outputs</h4>
        <label className="model-builder-checkbox">
          <input
            type="checkbox"
            checked={embeddingEnabled}
            onChange={(event) => {
              setDraftConfig((current) => (current ? setEmbeddingAuxEnabled(current, event.target.checked) : current));
            }}
          />
          <span>Enable embedding aux output</span>
        </label>
        {embeddingEnabled ? (
          <div className="model-builder-inline-grid">
            <label className="project-field">
              <span>Embedding Dimension</span>
              <select
                value={embeddingOutDim}
                onChange={(event) => {
                  const nextOutDim = Number.parseInt(event.target.value, 10);
                  if (!Number.isFinite(nextOutDim) || nextOutDim < 1) return;
                  setDraftConfig((current) =>
                    current ? setEmbeddingProjection(current, nextOutDim, embeddingNormalize as "none" | "l2") : current,
                  );
                }}
              >
                {EMBEDDING_DIM_OPTIONS.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
            <label className="project-field">
              <span>Normalize</span>
              <select
                value={embeddingNormalize}
                onChange={(event) => {
                  const value = event.target.value === "none" ? "none" : "l2";
                  setDraftConfig((current) => (current ? setEmbeddingProjection(current, embeddingOutDim, value) : current));
                }}
              >
                {EMBEDDING_NORMALIZE_OPTIONS.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
          </div>
        ) : null}
      </section>

      <section className="model-builder-step" id="model-step-export">
        <h4>Step 8: Export</h4>
        <label className="model-builder-checkbox">
          <input
            type="checkbox"
            checked={onnxEnabled}
            onChange={(event) => {
              patchDraft((next) => {
                const nextExport = asRecord(next.export);
                const nextOnnx = asRecord(nextExport.onnx);
                nextOnnx.enabled = event.target.checked;
                nextExport.onnx = nextOnnx;
                next.export = nextExport;
              });
            }}
          />
          <span>ONNX Enabled</span>
        </label>
        <label className="project-field">
          <span>Opset</span>
          <input
            type="number"
            min={9}
            step={1}
            value={String(onnxOpset)}
            onChange={(event) => {
              const parsed = Number.parseInt(event.target.value, 10);
              patchDraft((next) => {
                const nextExport = asRecord(next.export);
                const nextOnnx = asRecord(nextExport.onnx);
                nextOnnx.opset = Number.isFinite(parsed) ? parsed : 0;
                nextExport.onnx = nextOnnx;
                next.export = nextExport;
              });
            }}
          />
        </label>
        <label className="model-builder-checkbox">
          <input
            type="checkbox"
            checked={dynamicBatch}
            onChange={(event) => {
              setDraftConfig((current) =>
                current ? setDynamicShapeFlags(current, event.target.checked, dynamicHeightWidth) : current,
              );
            }}
          />
          <span>Dynamic Shapes: batch</span>
        </label>
        <label className="model-builder-checkbox">
          <input
            type="checkbox"
            checked={dynamicHeightWidth}
            onChange={(event) => {
              setDraftConfig((current) => (current ? setDynamicShapeFlags(current, dynamicBatch, event.target.checked) : current));
            }}
          />
          <span>Dynamic Shapes: height/width</span>
        </label>
      </section>
    </div>
  );

  const devValidationPanel =
    process.env.NODE_ENV !== "production" ? (
      <div>
        <p className="model-builder-validation-title">Validation ({validation.errors.length} issue{validation.errors.length === 1 ? "" : "s"})</p>
        {validation.errors.length > 0 ? (
          <ul className="model-builder-validation-list">
            {validation.errors.map((issue, index) => (
              <li key={`${issue.path}:${issue.keyword}:${index}`}>
                <code>{issue.path}</code> - {issue.message}
              </li>
            ))}
          </ul>
        ) : (
          <p className="labels-empty">No validation issues.</p>
        )}
      </div>
    ) : null;

  return (
    <>
      <ModelBuilderSkeleton
        title={`Model: ${modelName ?? modelId}`}
        backHref={`/projects/${encodeURIComponent(projectId)}/models`}
        modelName={modelName ?? modelId}
        datasetVersionName={currentVersionFromManifest?.name ?? currentManifestId ?? "-"}
        config={draftConfig ?? savedConfig ?? {}}
        isLoading={isLoading}
        errorMessage={errorMessage}
        editorContent={editorContent}
        isDirty={isDirty}
        isValid={isValid}
        isSaving={isSaving}
        onSave={() => void handleSave()}
        saveDisabled={!isDirty || !isValid || isSaving || isLoading || !draftConfig}
        saveError={saveError}
        validationPanel={devValidationPanel}
        onTrainModel={() => void handleTrainModel()}
        trainDisabled={isLoading || !draftConfig || isLaunchingTrain}
        trainButtonLabel={isLaunchingTrain ? "Preparing..." : "Train Model"}
      />
      {trainError ? (
        <div className={`status-toast is-error`} role="status" aria-live="polite">
          <span>{trainError}</span>
          <button type="button" aria-label="Dismiss message" onClick={() => setTrainError(null)}>
            x
          </button>
        </div>
      ) : null}
      {toastMessage ? (
        <div className={`status-toast ${toastTone === "error" ? "is-error" : "is-success"}`} role="status" aria-live="polite">
          <span>{toastMessage}</span>
          <button type="button" aria-label="Dismiss message" onClick={() => setToastMessage(null)}>
            x
          </button>
        </div>
      ) : null}
      {showTrainChoiceModal ? (
        <div className="project-modal-backdrop" role="presentation">
          <div className="project-modal" role="dialog" aria-modal="true" aria-label="Choose train action">
            <h3>Existing Experiment Found</h3>
            <p className="import-selection-summary">
              Latest run: <strong>{latestExperiment?.name ?? "-"}</strong>
            </p>
            <div className="project-modal-actions">
              <button type="button" className="ghost-button" onClick={() => setShowTrainChoiceModal(false)}>
                Close
              </button>
              <button type="button" className="ghost-button" disabled={!latestExperiment} onClick={handleContinueExperiment}>
                Continue
              </button>
              <button type="button" className="primary-button" disabled={isLaunchingTrain} onClick={() => void handleNewRun()}>
                {isLaunchingTrain ? "Creating..." : "New run"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
