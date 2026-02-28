import { redirect } from "next/navigation";

interface ProjectRootPageProps {
  params: {
    projectId: string;
  };
}

export default function ProjectRootPage({ params }: ProjectRootPageProps) {
  redirect(`/projects/${encodeURIComponent(decodeURIComponent(params.projectId))}/datasets`);
}

