# Frontend Workflow Refactor Note

## Introduced components
- `ProjectRibbon`
- `ProjectShellProvider`
- `ProjectSectionLayout`
- `AssetBrowser`
- `CanvasToolbar`
- `AssetFilmstrip`
- `ModelTable`
- `modelList` helper utilities for dataset-version/status enrichment

## Pages changed
- `projects/[projectId]/layout.tsx`
- `projects/[projectId]/dataset/page.tsx`
- `projects/[projectId]/models/page.tsx`
- `projects/[projectId]/models/new/page.tsx`
- `projects/[projectId]/models/[modelId]/page.tsx`
- `ProjectAssetsWorkspace`
- `Viewer`
- `LabelPanel`

## Route and CTA changes
- Labeling now routes to Dataset via `Create Dataset`
- Dataset now routes to prefilled model creation via `Train Model`
- Models list now routes to `/models/new` instead of immediately creating a draft
- `/models/new` accepts `taskId` and `datasetVersionId` query params for preselection

## Follow-up work
- Replace the current filmstrip chips with real thumbnails if asset preview loading is worth the added complexity
- Add task-aware filtering to experiments and deploy pages beyond the shared global context
- Expose canonical model status and dataset-version fields from the backend to remove client-side enrichment heuristics
- Add true pan/zoom interactions to the annotation canvas toolbar
