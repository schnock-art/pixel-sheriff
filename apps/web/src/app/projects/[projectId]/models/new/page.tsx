"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { ApiError, createProjectModel } from "../../../../../lib/api";

interface NewModelPageProps {
  params: {
    projectId: string;
  };
}

export default function NewModelPage({ params }: NewModelPageProps) {
  const router = useRouter();
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const startedRef = useRef(false);

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    let isMounted = true;

    async function createAndRedirect() {
      try {
        const created = await createProjectModel(projectId, {});
        if (!isMounted) return;
        router.replace(`/projects/${encodeURIComponent(projectId)}/models/${encodeURIComponent(created.id)}`);
      } catch (error) {
        if (!isMounted) return;
        if (error instanceof ApiError && error.responseBody) {
          setErrorMessage(error.responseBody);
        } else {
          setErrorMessage(error instanceof Error ? error.message : "Failed to create model");
        }
      }
    }

    void createAndRedirect();

    return () => {
      isMounted = false;
    };
  }, [projectId, router]);

  return (
    <main className="workspace-shell project-page-shell">
      <section className="workspace-frame project-content-frame placeholder-page">
        <header className="project-section-header">
          <h2>New Model</h2>
        </header>
        <div className="placeholder-card">
          <p>{errorMessage ?? "Creating model draft..."}</p>
        </div>
      </section>
    </main>
  );
}
