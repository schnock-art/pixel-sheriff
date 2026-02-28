import { ModelBuilderSkeleton } from "../../../../../components/workspace/ModelBuilderSkeleton";

interface ModelDetailPageProps {
  params: {
    projectId: string;
    modelId: string;
  };
}

export default function ModelDetailPage({ params }: ModelDetailPageProps) {
  const modelId = decodeURIComponent(params.modelId);
  return <ModelBuilderSkeleton title={`Model: ${modelId}`} />;
}


