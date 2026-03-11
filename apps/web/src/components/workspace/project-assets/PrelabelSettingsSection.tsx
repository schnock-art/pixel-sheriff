import type { PrelabelConfig } from "../../../lib/api";


interface PrelabelSettingsSectionProps {
  enabled: boolean;
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

  function update(patch: Partial<PrelabelConfig>) {
    onChange({ ...resolvedValue, ...patch });
  }

  return (
    <section className="placeholder-card">
      <h4>Prelabels</h4>
      <label className="project-field">
        <span>Prelabel source</span>
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
          <option value="active_deployment">Project model</option>
          <option value="florence2">AI prompt assist</option>
        </select>
      </label>
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
                value={String(resolvedValue.confidence_threshold)}
                onChange={(event) => update({ confidence_threshold: Number(event.target.value) || 0 })}
                inputMode="decimal"
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

