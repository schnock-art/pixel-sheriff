Vision Hub
The Home Assistant of Edge Vision
1. Product Vision

Vision Hub is a local-first, opinionated computer vision pipeline framework that enables fast, reproducible iteration from dataset to deployable edge model.

It provides a single place to:

Collect and version datasets

Train and retrain models

Compare runs

Track exactly what changed

Export reproducible ONNX deployment bundles

Promote models through clear lifecycle stages

Vision Hub eliminates ambiguity in computer vision workflows.

At any time, it must answer:

What data trained this model?

What label schema was used?

What hyperparameters were used?

Is this model actually better?

What changed since the previous run?

What is currently deployed?

2. Core Philosophy
2.1 Local-First

Runs fully local.
No cloud dependency.
No SaaS requirement.
Designed for edge environments.

Cloud support is optional, not required.

2.2 No Mysteries

Every model artifact must be reproducible from:

Dataset version ID

Label schema ID (class order frozen)

Training configuration

Code version (git hash)

Transform pipeline

Evaluation dataset ID

If a model cannot be reproduced, it is invalid.

2.3 Immutable Everything

Dataset versions are immutable.

Label schemas are immutable once used in training.

Model artifacts are immutable.

Promotion state is explicit.

Mutation creates new versions — never silent overwrites.

2.4 Opinionated Over Flexible

Vision Hub prioritizes:

Clarity

Reproducibility

Minimal configuration

Clean internal contracts

Over:

Plugin chaos

Endless configuration flags

Supporting every edge case

Research flexibility

This is not a research playground.
It is an iteration engine.

3. Non-Goals (Very Important)

Vision Hub is NOT:

A general-purpose ML framework

A replacement for PyTorch

A model zoo

A SaaS platform

A massive labeling suite

A feature-maximized AutoML tool

It intentionally does NOT:

Support arbitrary custom model graphs (initially)

Support every task type

Handle non-CV modalities

Provide distributed training orchestration

Attempt to abstract away ML fundamentals

The scope is narrow on purpose.

4. MVP Scope

MVP must include only the following:

4.1 Immutable Dataset Versions

Dataset snapshots stored on disk

Deterministic manifest

Frozen class order

Clear lineage

Human-readable structure

Each dataset version must have:

dataset_id
version_id
label_schema_id
sample_ids
creation_timestamp
parent_version_id (optional)
4.2 Reproducible Training Runs

Each training run must store:

run_id
dataset_version_id
model_architecture_id
hyperparameters
transform_pipeline_hash
git_commit_hash
metrics
artifacts

Re-running with identical inputs must reproduce the same result (within determinism constraints).

4.3 Model Registry

A simple registry that tracks:

Candidate models

Validated models

Staged models

Production models

Promotion is explicit and recorded.

No implicit overwrites.

4.4 ONNX Export Bundle

Export must produce:

model.onnx
label_schema.json
dataset_manifest.json
metrics.json
runtime_config.json

The bundle must be self-contained.

4.5 Model Comparison View

Must support:

Metric comparison across runs

Dataset version comparison

Hyperparameter comparison

Clear delta view

The question “Is this better?” must be trivial to answer.

5. Core Entities

Vision Hub revolves around five core entities:

Dataset

DatasetVersion

LabelSchema

TrainingRun

ModelArtifact

If new features do not cleanly map to these entities, they should be questioned.

6. Promotion Lifecycle

Each model artifact has a lifecycle state:

candidate

validated

staged

production

archived

Transitions must be explicit and logged.

No silent promotions.

7. North-Star User Experience

The happy path must feel like:

Add new labeled data

Create dataset version

Click “Train”

Compare against previous model

Promote if better

Export ONNX bundle

Deploy to edge device

No notebooks.
No scattered scripts.
No folder archaeology.

One system.
One source of truth.

8. Educational Angle

Vision Hub should make it easy for:

Students

Universities

Hobbyists

To go from labeled dataset to deployable edge model in a clean, inspectable way.

Every decision should be inspectable.
Every artifact should be understandable.

9. Long-Term Direction (Optional, Not MVP)

Active learning loop

Data drift indicators

Edge feedback ingestion

Lightweight deployment agent

Quantization + optimization helpers

Multi-project management

These are expansions, not requirements.

10. Design Constraint

If a feature increases:

Hidden state

Ambiguity

Magic behavior

Implicit defaults

It should be rejected.

Clarity over cleverness.

11. Current Implementation Snapshot (March 2026)

Implemented now:

Local-first full stack (web + API + trainer + Postgres + Redis)

Task-scoped labeling (`classification`, `bbox`, `segmentation`) with per-task label schemas

Immutable dataset versions with deterministic export bundles (`manifest.json`, `coco_instances.json`, `assets/`)

Project-scoped model builder + schema-validated config editing

Experiment lifecycle with queued training, SSE metrics/events, runtime/log endpoints, and ONNX artifacts

Deploy flow for classification ONNX with active deployment selection, warmup, and single-asset prediction

Model-assisted labeling suggestions in the labeling panel (`Suggest`, top-k, `Apply top-1`)

Known gaps:

Detection/segmentation training execution is not implemented yet

Batch MAL scoring/curation workflows are still pending

Review/QA moderation workflows are still pending
