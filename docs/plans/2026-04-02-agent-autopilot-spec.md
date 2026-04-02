# Constrained Agent Autopilot Spec

**Date:** 2026-04-02
**Status:** Proposed

## Summary

Add a constrained, review-first "agent autopilot" layer on top of Pixel Sheriff's existing dataset, model, experiment, variant, deployment, and prediction-review workflows.

The product goal is not autonomous model invention. The goal is to help a user move faster through the existing loop:

1. inspect dataset quality and readiness
2. propose a small set of sensible model and training configurations
3. queue repeatable experiment sweeps
4. compare quality, runtime, and export variants
5. surface relabel candidates from poor predictions
6. recommend a deployment candidate for human approval

This should feel like an expert assistant operating inside the current project/task workflow, not a separate "AI lab" product.

## Why This Fits The Repo

The current repository already has the right primitives:

- family/backbone/task constraints live in `apps/api/src/sheriff_api/ml/registry.py`
- experiment jobs already have typed payloads in `apps/trainer/src/pixel_sheriff_trainer/jobs.py`
- experiment variant orchestration already exists in `apps/trainer/src/pixel_sheriff_trainer/variants.py`
- augmentation controls already exist in `apps/web/src/lib/workspace/augmentationConfig.js`
- experiment CRUD, start, logs, runtime, and variants are already exposed through the web client helpers in `apps/web/src/lib/api/experiments.ts`

Because those primitives already exist, the first useful "agent" can be an orchestrator and recommender over supported choices, rather than a system that invents arbitrary architectures or training pipelines.

## Product Framing

Use "Autopilot" or "Advisor" language rather than "agents train their own models."

Recommended framing:

- the system can recommend
- the system can batch safe actions
- the system must explain why it made a recommendation
- the user stays in control of experiment launch, deployment activation, and label acceptance

## Goals

- Reduce the time from dataset version to first good experiment.
- Make model and training setup less intimidating for non-ML-specialist users.
- Increase the number of experiments that are meaningfully different instead of randomly tweaked.
- Tighten the loop between experiment outcomes and relabel/review work.
- Reuse current task, family, backbone, augmentation, runtime, and variant flows rather than creating parallel systems.

## Non-Goals

- Arbitrary model architecture authoring.
- Writing raw PyTorch model code from prompts.
- Auto-deploying models without explicit user approval.
- Auto-promoting predictions or prelabels into annotations without review.
- Replacing the current model builder or experiment pages with opaque AI-only flows.
- Supporting unsupported task/family combinations outside the existing registry.

## Core Principles

### 1. Constrained

Recommendations must only target supported tasks, families, backbones, augmentations, runtimes, and variant actions already implemented by the repo.

### 2. Explainable

Every recommendation should include short human-readable reasoning such as:

- "dataset is small, start with pretrained classifier and light augmentation"
- "class imbalance is high, compare weighted loss and oversampling-safe augmentation"
- "detection latency target is tight, compare RetinaNet FP32 vs SSD Lite FP16"

### 3. Review-First

The system can prepare plans and queue work, but destructive or user-visible state changes still require confirmation.

### 4. Project-Local

Recommendations should be derived from project-local state: dataset versions, categories, experiment history, evaluation metrics, deployment history, accepted/rejected predictions, and prelabel review outcomes.

## User Stories

### Dataset Readiness

As a user, I can ask Pixel Sheriff to assess a dataset version and tell me:

- class counts and imbalance
- unlabeled or low-signal risks
- likely task fit issues
- whether the dataset is ready for a baseline run

### Config Proposal

As a user, I can ask for a recommended baseline config for a task/dataset and get:

- family
- backbone
- input size
- augmentation profile or steps
- runtime defaults
- recommended experiment variants to compare

### Sweep Planning

As a user, I can ask for a safe sweep plan and receive a small set of named experiments such as:

- baseline
- smaller/faster alternative
- stronger/slower alternative
- augmentation-focused variant
- quantization follow-up candidates

### Result Interpretation

As a user, I can ask why one run is better than another and get a short explanation based on:

- task metrics
- runtime
- variant artifacts
- confusion or prediction quality

### Relabel Loop

As a user, I can ask which assets should be reviewed next and get candidates drawn from:

- misclassified samples
- lowest-confidence correct samples
- high-confidence wrong samples
- rejected deployment predictions
- unresolved prelabel proposals

### Deployment Recommendation

As a user, I can ask which experiment to deploy next and get a recommendation that balances:

- quality
- export readiness
- variant availability
- approximate inference tradeoffs

## Scope By Phase

### Phase 1: Advisor

Deliver a read-only advisor that does not auto-launch work.

Includes:

- dataset health summary
- baseline config recommendation
- short rationale blocks
- "recommended next actions" UI

Does not include:

- experiment queueing
- auto-created experiments
- deployment recommendation writes

### Phase 2: Sweep Runner

Allow the user to accept a recommended sweep plan and create a bounded set of experiments.

Includes:

- generated sweep presets from supported family/backbone/input/augmentation/runtime combinations
- one-click "create recommended sweep"
- optional one-click "start all queued"
- experiment grouping by autopilot run

Guardrails:

- hard cap on number of generated experiments per action
- only supported combinations from registry and existing config contracts
- no mutation of existing experiments

### Phase 3: Variant Intelligence

Use the existing variant pipeline to recommend and compare export paths.

Includes:

- suggest PTQ, QAT, or FP16 when supported for the task
- summarize tradeoffs across `fp32`, `fp16`, `ptq_int8`, and `qat_int8`
- mark a recommended serving candidate and a recommended best-quality candidate

### Phase 4: Curation Loop

Connect experiment outcomes and deployment review back into the labeling workflow.

Includes:

- relabel candidate queue
- folder or dataset slices needing attention
- "train again after review" recommendations
- explanations tied to confusion rows, low-confidence samples, or rejected deployment predictions

### Phase 5: Deployment Advisor

Recommend, but do not automatically activate, a deployment candidate.

Includes:

- deployment candidate ranking
- explicit rationale and caveats
- recommended threshold range where applicable
- reminder when no experiment meets a minimum confidence bar

## UX Proposal

Introduce a project-scoped "Autopilot" surface in stages.

Suggested initial surfaces:

- model builder: "Recommend baseline"
- experiments list/new experiment: "Plan sweep"
- experiment detail: "Explain results" and "Recommend variants"
- labeling workspace or dataset page: "Review hard examples"
- deploy page: "Recommend candidate"

Phase 1 UI should bias toward side panels, cards, or callouts inside existing pages rather than a brand-new top-level workflow.

## Technical Design Direction

### Recommendation Engine First

The first version should be deterministic and repo-native:

- use dataset metadata and experiment history
- use explicit heuristics and scoring rules
- produce structured recommendation payloads
- optionally add LLM-generated explanation text later

This avoids making product value depend on an external model provider before the internal recommendation contracts are stable.

### Suggested API Shape

Add project/task-scoped autopilot endpoints after the recommendation payload is defined.

Likely endpoint families:

- `GET /autopilot/dataset-health`
- `POST /autopilot/recommend-baseline`
- `POST /autopilot/plan-sweep`
- `POST /autopilot/create-sweep`
- `GET /autopilot/relabel-candidates`
- `POST /autopilot/recommend-deployment`

The important part is the contract shape, not the exact paths yet.

### Suggested Record Types

Persist autopilot artifacts as explicit records rather than transient text blobs.

Candidate record types:

- `autopilot_runs`
- `autopilot_recommendations`
- `autopilot_sweep_plans`
- `autopilot_explanations`

Each record should store:

- project/task/dataset context
- inputs used
- outputs produced
- schema version
- created timestamp
- optional accepted/rejected state

That gives the product an audit trail and keeps "why did it do that?" answerable.

## Recommendation Inputs

The advisor should be allowed to read:

- dataset version summary and split membership
- category counts and class imbalance
- annotation coverage
- model family/task constraints
- augmentation settings
- experiment metrics and runtime info
- experiment samples and confusion views
- deployment review outcomes
- prelabel proposal outcomes

The advisor should not need raw image bytes for Phase 1.

## Recommendation Outputs

Phase 1 outputs should be structured and small.

Example recommendation payload categories:

- `dataset_health`
- `baseline_config`
- `sweep_plan`
- `variant_followups`
- `relabel_candidates`
- `deployment_candidate`

Each recommendation should include:

- `kind`
- `title`
- `summary`
- `reasoning`
- `confidence`
- `actions`
- `inputs_snapshot`

## Decision Policy Examples

Examples of repo-native heuristics that fit the current product:

- small classification dataset: pretrained classifier, smaller sweep, light augmentation
- small detection dataset with few labels per class: avoid aggressive augmentation, prefer simpler baseline first
- high class imbalance: recommend class-aware loss or weighted sampling when supported
- poor validation quality but strong train quality: recommend augmentation or regularization changes instead of larger backbone first
- strong FP32 result on supported task: suggest PTQ and optionally QAT follow-up
- weak metrics and many rejected deployment predictions: recommend relabel loop before more sweeps

## Roadmap

### Milestone A: Contracts And Heuristics

- define autopilot payload schemas
- add dataset-health and baseline-recommendation services
- add focused API tests for recommendation contracts
- add a minimal web UI callout in model builder or experiments

### Milestone B: Sweep Planning

- define sweep-plan schema
- generate bounded experiment sets
- support one-click creation of queued experiments
- show autopilot-created experiment groups in the UI

### Milestone C: Variant Recommendation

- score FP32, FP16, PTQ, and QAT follow-ups using current task support
- summarize export/runtime tradeoffs in experiment detail
- add deployment-candidate recommendation payload

### Milestone D: Relabel Intelligence

- define relabel-candidate contract
- aggregate low-confidence and wrong-prediction evidence
- expose review queues back in labeling and dataset workflows

### Milestone E: Optional LLM Layer

- add natural-language explanation generation on top of structured recommendations
- keep structured recommendation generation deterministic underneath
- never make the LLM the source of truth for supported combinations

## Open Questions

- Should Phase 1 ship with purely deterministic copy, or with optional LLM-written explanations?
- Should autopilot artifacts live in PostgreSQL immediately, or start with file-backed persistence like some current model metadata paths?
- Should sweep plans create experiments only, or also auto-start them behind a single confirmation?
- What is the minimum dataset-health signal set needed to feel useful on day one?
- Do we want "Autopilot" to be global branding, or should the first UI simply use labels like "Recommend baseline" and "Plan sweep"?
- Should relabel candidate generation prioritize experiment evaluation samples, deployment review rejects, or prelabel review outcomes first?

## Recommendation

Implement this as a constrained advisor and orchestration layer over the current system.

Do not position it as autonomous model invention.

The best first slice is:

1. dataset health summary
2. baseline config recommendation
3. bounded sweep plan
4. variant recommendation on successful experiments

That slice is useful, aligned with the current architecture, and commercially meaningful without requiring a new model-authoring system.
