"use client";

import { useEffect, useMemo, useState } from "react";

import { ModelBuilderSkeleton } from "../../../../../components/workspace/ModelBuilderSkeleton";
import { ApiError, getProjectModel, type ProjectModelRecord } from "../../../../../lib/api";

interface ModelDetailPageProps {
  params: {
    projectId: string;
    modelId: string;
  };
}

export default function ModelDetailPage({ params }: ModelDetailPageProps) {
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);
  const modelId = useMemo(() => decodeURIComponent(params.modelId), [params.modelId]);

  const [record, setRecord] = useState<ProjectModelRecord | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    async function loadModel() {
      setIsLoading(true);
      setErrorMessage(null);
      try {
        const nextRecord = await getProjectModel(projectId, modelId);
        if (isMounted) setRecord(nextRecord);
      } catch (error) {
        if (!isMounted) return;
        if (error instanceof ApiError && error.responseBody) {
          setErrorMessage(error.responseBody);
        } else {
          setErrorMessage(error instanceof Error ? error.message : "Failed to load model");
        }
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }

    void loadModel();

    return () => {
      isMounted = false;
    };
  }, [modelId, projectId]);

  return (
    <ModelBuilderSkeleton
      title={`Model: ${record?.name ?? modelId}`}
      backHref={`/projects/${encodeURIComponent(projectId)}/models`}
      modelName={record?.name ?? modelId}
      config={record?.config_json ?? null}
      isLoading={isLoading}
      errorMessage={errorMessage}
    />
  );
}
