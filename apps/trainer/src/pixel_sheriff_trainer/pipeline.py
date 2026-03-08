from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


@dataclass
class TaskLoaders:
    train: "DataLoader[Any]"
    val: "DataLoader[Any]"
    num_classes: int
    class_names: list[str]
    class_order: list[str]
    train_count: int
    val_count: int
    skipped_unlabeled: int


@dataclass
class TrainingResult:
    status: str  # "done" | "cancelled" | "error"
    best_epoch: int | None
    best_metric: float | None
    final_evaluation: Any | None  # task-specific evaluation object


@dataclass
class EvaluationResult:
    summary: dict[str, Any]  # e.g. {"accuracy": 0.94} | {"mAP50": 0.72} | {"mIoU": 0.65}
    per_class: dict[str, Any]
    raw: Any | None  # original evaluation object for pipeline-internal use


class TaskPipeline:
    """Protocol-style base for task pipelines. Each task type implements this interface."""

    task_kind: ClassVar[str]

    def build_loaders(self, job: Any, workdir: Path) -> TaskLoaders:
        raise NotImplementedError

    def run_training(
        self,
        loaders: TaskLoaders,
        job: Any,
        workdir: Path,
        on_epoch: Callable[..., None],
        on_checkpoint: Callable[..., None],
        should_cancel: Callable[[], bool],
        device: Any,
        resume_state: dict[str, Any] | None,
    ) -> TrainingResult:
        raise NotImplementedError

    def evaluate(
        self,
        loaders: TaskLoaders,
        job: Any,
        workdir: Path,
        training_result: TrainingResult,
    ) -> EvaluationResult:
        raise NotImplementedError

    def write_evaluation(
        self,
        storage: Any,
        *,
        project_id: str,
        experiment_id: str,
        attempt: int,
        job: Any,
        loaders: TaskLoaders,
        training_result: TrainingResult,
        eval_result: EvaluationResult,
    ) -> None:
        raise NotImplementedError

    def export_onnx(
        self,
        storage: Any,
        *,
        project_id: str,
        experiment_id: str,
        attempt: int,
        job: Any,
        loaders: TaskLoaders,
    ) -> Any:
        raise NotImplementedError


# Populated by each pipeline module on import.
# runner.py imports this registry to dispatch training by job.task.
PIPELINE_REGISTRY: dict[str, TaskPipeline] = {}

# Trigger registration of all known pipelines.
def _register_all() -> None:
    from pixel_sheriff_trainer.classification.pipeline import ClassificationPipeline  # noqa: F401
    from pixel_sheriff_trainer.detection.pipeline import DetectionPipeline  # noqa: F401
    from pixel_sheriff_trainer.segmentation.pipeline import SegmentationPipeline  # noqa: F401


_register_all()
