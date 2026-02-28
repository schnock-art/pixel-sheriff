import { ModelBuilderSkeleton } from "../../../../../components/workspace/ModelBuilderSkeleton";

interface NewModelPageProps {
  params: {
    projectId: string;
  };
}

export default function NewModelPage({ params }: NewModelPageProps) {
  const projectId = decodeURIComponent(params.projectId);
  return <ModelBuilderSkeleton title={`Model Builder | ${projectId}`} />;
}

