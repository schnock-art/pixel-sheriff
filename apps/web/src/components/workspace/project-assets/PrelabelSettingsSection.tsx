import { useEffect, useState } from "react";

import { getPrelabelSourceStatus, type PrelabelConfig, type PrelabelSourceStatus } from "../../../lib/api";


interface PrelabelSettingsSectionProps {
  enabled: boolean;
  projectId: string | null;
  taskId: string | null;
  value: PrelabelConfig | null;
  defaultPrompts: string[];
  onChange: (value: PrelabelConfig | null) => void;
  samplingLabel: string;
  samplingHint: string;
}


function normalizePromptInput(rawValue: string): string[] {
  return rawValue
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
}


export function PrelabelSettingsSection({
  enabled,
  projectId,
  taskId,
  value,
  defaultPrompts,
  onChange,
  samplingLabel,
  samplingHint,
}: PrelabelSettingsSectionProps) {
  if (!enabled) return null;

  const resolvedValue =
    value ??
    ({
      source_type: "florence2",
      prompts: defaultPrompts,
      frame_sampling: { mode: "every_n_frames", value: 15 },
      confidence_threshold: 0.25,
      max_detections_per_frame: 20,
    } satisfies PrelabelConfig);
  const [confidenceInput, setConfidenceInput] = useState(String(resolvedValue.confidence_threshold));
  const [isEditingConfidence, setIsEditingConfidence] = useState(false);
  const [sourceStatus, setSourceStatus] = useState<PrelabelSourceStatus | null>(null);
  const [sourceStatusError, setSourceStatusError] = useState<string | null>(null);
  const [isCheckingSourceStatus, setIsCheckingSourceStatus] = useState(false);

  useEffect(() => {
    if (isEditingConfidence) return;
    setConfidenceInput(String(resolvedValue.confidence_threshold));
  }, [isEditingConfidence, resolvedValue.confidence_threshold]);

  useEffect(() => {
    let isMounted = true;
    async function loadSourceStatus() {
      if (!enabled || !value || !projectId || !taskId) {
        if (!isMounted) return;
        setSourceStatus(null);
        setSourceStatusError(null);
        setIsCheckingSourceStatus(false);
        return;
      }
      try {
        if (isMounted) {
          setIsCheckingSourceStatus(true);
          setSourceStatusError(null);
        }
        const status = await getPrelabelSourceStatus(projectId, taskId, value);
        if (!isMounted) return;
        setSourceStatus(status);
      } catch (error) {
        if (!isMounted) return;
        setSourceStatus(null);
        setSourceStatusError(error instanceof Error ? error.message : "AI source unavailable");
      } finally {
        if (isMounted) setIsCheckingSourceStatus(false);
      }
    }
    void loadSourceStatus();
    return () => {
      isMounted = false;
    };
  }, [enabled, projectId, taskId, value?.source_type]);

  function update(patch: Partial<PrelabelConfig>) {
    onChange({ ...resolvedValue, ...patch });
  }

  function commitConfidenceInput(rawValue: string) {
    const normalized = rawValue.trim();
    if (normalized === "") {
      setConfidenceInput(String(resolvedValue.confidence_threshold));
      return;
    }

    const parsed = Number(normalized);
    if (!Number.isFinite(parsed)) {
      setConfidenceInput(String(resolvedValue.confidence_threshold));
      return;
    }

    const nextValue = Math.min(1, Math.max(0, parsed));
    update({ confidence_threshold: nextValue });
    setConfidenceInput(String(nextValue));
  }

  const sourceStatusTone = sourceStatusError ? "error" : sourceStatus ? "ready" : "idle";
  const sourceStatusLabel = sourceStatusError
    ? "Unavailable"
    : isCheckingSourceStatus
      ? "Checking…"
      : sourceStatus?.device_selected
        ? `Ready on ${sourceStatus.device_selected.toUpperCase()}`
        : "Idle";
  const sourceStatusMeta = sourceStatus
    ? `${sourceStatus.source_label}${sourceStatus.device_preference ? ` • pref ${sourceStatus.device_preference}` : ""}`
    : null;

  return (
    <section className="placeholder-card">
      <h4>Prelabels</h4>
      <label className="project-field">
        <span>AI model</span>
        <select
          value={value ? resolvedValue.source_type : "none"}
          onChange={(event) => {
            const nextValue = event.target.value;
            if (nextValue === "none") {
              onChange(null);
              return;
            }
            if (nextValue === "active_deployment") {
              onChange({ ...resolvedValue, source_type: "active_deployment", prompts: [] });
              return;
            }
            onChange({ ...resolvedValue, source_type: "florence2", prompts: resolvedValue.prompts.length > 0 ? resolvedValue.prompts : defaultPrompts });
          }}
        >
          <option value="none">None</option>
          <option value="florence2">Florence-2 prompt assist</option>
          <option value="active_deployment">Active project deployment</option>
        </select>
      </label>
      {value ? (
        <div className="prelabel-source-status-row">
          <span className={`prelabel-source-status-badge is-${sourceStatusTone}`}>{sourceStatusLabel}</span>
          <span className="prelabel-source-status-text">
            {sourceStatusMeta ?? (value.source_type === "florence2" ? "Florence-2" : "Active project deployment")}
          </span>
        </div>
      ) : null}
      {sourceStatusError ? <p className="import-field-error">{sourceStatusError}</p> : null}
      {value ? (
        <>
          {resolvedValue.source_type === "florence2" ? (
            <label className="project-field">
              <span>Classes / prompts</span>
              <input
                value={resolvedValue.prompts.join(", ")}
                onChange={(event) => update({ prompts: normalizePromptInput(event.target.value) })}
                placeholder="person, forklift, pallet"
              />
              <span className="import-field-hint">Comma-separated prompts. Defaults to task classes.</span>
            </label>
          ) : (
            <p className="labels-empty">Uses the active compatible bbox deployment for this task.</p>
          )}
          <div className="import-inline-grid">
            <label className="project-field">
              <span>{samplingLabel}</span>
              <input
                value={String(resolvedValue.frame_sampling.value)}
                onChange={(event) =>
                  update({
                    frame_sampling: {
                      ...resolvedValue.frame_sampling,
                      value: Number(event.target.value) || 1,
                    },
                  })
                }
                inputMode="decimal"
              />
              <span className="import-field-hint">{samplingHint}</span>
            </label>
            <label className="project-field">
              <span>Confidence threshold</span>
              <input
                value={confidenceInput}
                onChange={(event) => setConfidenceInput(event.target.value)}
                onFocus={() => setIsEditingConfidence(true)}
                onBlur={(event) => {
                  setIsEditingConfidence(false);
                  commitConfidenceInput(event.target.value);
                }}
                type="number"
                inputMode="decimal"
                step="0.01"
                min="0"
                max="1"
              />
            </label>
          </div>
          <label className="project-field">
            <span>Max detections / frame</span>
            <input
              value={String(resolvedValue.max_detections_per_frame)}
              onChange={(event) => update({ max_detections_per_frame: Math.max(1, Number(event.target.value) || 1) })}
              inputMode="numeric"
            />
          </label>
          <p className="labels-empty">
            Generates first-pass bounding boxes on sampled frames. Review is required before accepted boxes become normal annotations.
          </p>
        </>
      ) : null}
    </section>
  );
}
