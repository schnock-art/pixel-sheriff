# Frontend Workflow Refactor Note

Historical note for the shell/workspace split that established the current frontend structure.

## Landed structure
- Shared project shell:
  - `ProjectRibbon`
  - `ProjectShellProvider`
  - `ProjectSectionLayout`
- Labeling workspace composition:
  - `ProjectAssetsWorkspace`
  - `Viewer`
  - labeling sidebar and filmstrip subcomponents under `apps/web/src/components/workspace/project-assets/`
- Model list/detail scaffolding:
  - `ModelTable`
  - model-list enrichment helpers

## Routes affected
- `apps/web/src/app/projects/[projectId]/layout.tsx`
- `apps/web/src/app/projects/[projectId]/datasets/page.tsx`
- `apps/web/src/app/projects/[projectId]/dataset/page.tsx`
- `apps/web/src/app/projects/[projectId]/models/page.tsx`
- `apps/web/src/app/projects/[projectId]/models/new/page.tsx`
- `apps/web/src/app/projects/[projectId]/models/[modelId]/page.tsx`
- `apps/web/src/app/projects/[projectId]/experiments/page.tsx`
- `apps/web/src/app/projects/[projectId]/deploy/page.tsx`

## Current behavior this note refers to
- Labeling routes to dataset creation via `Create Dataset`.
- Dataset routes to prefilled model creation via `Train Model`.
- The workspace now supports image review plus sequence-backed video/webcam frames.
- The shell owns the shared project/task selectors and top-level section navigation.

## Remaining follow-up
- Reduce client-side enrichment heuristics as more canonical dataset/model status fields move server-side.
- Continue expanding focused UI automation around sequence review and AI prelabels.
- Improve webcam capture diagnostics for intermittent browser/device write failures.
