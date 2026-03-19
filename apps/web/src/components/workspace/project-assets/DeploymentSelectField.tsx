import type { DeploymentItem } from "../../../lib/api";

interface DeploymentSelectFieldProps {
  label: string;
  deployments: DeploymentItem[];
  activeDeploymentId: string | null;
  value: string | null;
  loading?: boolean;
  disabled?: boolean;
  emptyMessage: string;
  helpText?: string | null;
  onChange: (value: string | null) => void;
}

export function DeploymentSelectField({
  label,
  deployments,
  activeDeploymentId,
  value,
  loading = false,
  disabled = false,
  emptyMessage,
  helpText = null,
  onChange,
}: DeploymentSelectFieldProps) {
  const selectedDeployment = deployments.find((item) => item.deployment_id === value) ?? null;
  const selectedMeta = selectedDeployment
    ? `Device pref ${selectedDeployment.device_preference.toUpperCase()}${selectedDeployment.deployment_id === activeDeploymentId ? " | active" : ""}`
    : helpText;

  return (
    <>
      <label className="project-field">
        <span>{label}</span>
        <select
          value={value ?? ""}
          onChange={(event) => onChange(event.target.value || null)}
          disabled={disabled || loading || deployments.length === 0}
        >
          <option value="" disabled>
            {loading ? "Loading deployments..." : deployments.length > 0 ? "Select a deployment" : "No compatible deployments"}
          </option>
          {deployments.map((deployment) => (
            <option key={deployment.deployment_id} value={deployment.deployment_id}>
              {deployment.name}
              {deployment.deployment_id === activeDeploymentId ? " (Active)" : ""}
            </option>
          ))}
        </select>
        {selectedMeta ? <span className="import-field-hint">{selectedMeta}</span> : null}
      </label>
      {!loading && deployments.length === 0 ? <p className="labels-empty">{emptyMessage}</p> : null}
    </>
  );
}
