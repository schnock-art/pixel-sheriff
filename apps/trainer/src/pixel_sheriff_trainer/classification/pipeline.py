from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pixel_sheriff_trainer.classification.dataset import build_classification_loaders
from pixel_sheriff_trainer.classification.eval import ClassifierEvaluation
from pixel_sheriff_trainer.classification.train import EpochMetrics, run_training
from pixel_sheriff_trainer.export_onnx import OnnxExportResult, export_best_classification_onnx
from pixel_sheriff_trainer.io.evaluation import write_classification_evaluation
from pixel_sheriff_trainer.pipeline import (
    EvaluationResult,
    PIPELINE_REGISTRY,
    TaskLoaders,
    TaskPipeline,
    TrainingResult,
)


class ClassificationPipeline(TaskPipeline):
    task_kind = "classification"

    def build_loaders(self, job: Any, workdir: Path) -> TaskLoaders:
        # workdir = {storage_root}/experiments/{project_id}/{experiment_id}/runs/{attempt}/workdir
        # storage_root is at parents[5] (0-indexed from immediate parent)
        from pixel_sheriff_trainer.classification.train import resolve_device
        from pixel_sheriff_trainer.io.storage import ExperimentStorage

        storage_root = workdir.parents[5]
        storage = ExperimentStorage(str(storage_root))

        zip_relpath = str(job.dataset_export.get("zip_relpath") or "")
        if not zip_relpath:
            raise ValueError("dataset_export_missing")
        zip_path = storage.resolve(zip_relpath)
        device = resolve_device(job.training_config)

        loaded = build_classification_loaders(
            export_zip_path=zip_path,
            workdir=workdir,
            model_config=job.model_config,
            training_config=job.training_config,
            device_type=device.type,
        )
        return TaskLoaders(
            train=loaded.train_loader,
            val=loaded.val_loader,
            num_classes=loaded.num_classes,
            class_names=loaded.class_names,
            class_order=loaded.class_order,
            train_count=loaded.train_count,
            val_count=loaded.val_count,
            skipped_unlabeled=loaded.skipped_unlabeled,
        )

    def run_training(
        self,
        loaders: TaskLoaders,
        job: Any,
        workdir: Path,
        on_epoch: Callable[[dict[str, Any]], None],
        on_checkpoint: Callable[[str, int, str | None, float | None, dict[str, Any]], None],
        should_cancel: Callable[[], bool],
        device: Any,
        resume_state: dict[str, Any] | None,
    ) -> TrainingResult:
        def _on_epoch(epoch_metrics: EpochMetrics) -> None:
            row: dict[str, Any] = {
                "epoch": int(epoch_metrics.epoch),
                "train_loss": float(epoch_metrics.train_loss),
                "train_accuracy": float(epoch_metrics.train_accuracy)
                if isinstance(epoch_metrics.train_accuracy, (int, float))
                else None,
                "val_loss": float(epoch_metrics.val_loss)
                if isinstance(epoch_metrics.val_loss, (int, float))
                else None,
                "val_accuracy": float(epoch_metrics.val_accuracy)
                if isinstance(epoch_metrics.val_accuracy, (int, float))
                else None,
                "val_macro_f1": float(epoch_metrics.val_macro_f1)
                if isinstance(epoch_metrics.val_macro_f1, (int, float))
                else None,
                "val_macro_precision": float(epoch_metrics.val_macro_precision)
                if isinstance(epoch_metrics.val_macro_precision, (int, float))
                else None,
                "val_macro_recall": float(epoch_metrics.val_macro_recall)
                if isinstance(epoch_metrics.val_macro_recall, (int, float))
                else None,
                "lr": float(epoch_metrics.lr),
                "epoch_seconds": float(epoch_metrics.epoch_seconds),
                "eta_seconds": float(epoch_metrics.eta_seconds)
                if isinstance(epoch_metrics.eta_seconds, (int, float))
                else None,
                "evaluated": bool(epoch_metrics.evaluated),
            }
            on_epoch(row)

        run_status, final_evaluation = run_training(
            model_config=job.model_config,
            training_config=job.training_config,
            train_loader=loaders.train,
            val_loader=loaders.val,
            num_classes=loaders.num_classes,
            should_cancel=should_cancel,
            on_epoch=_on_epoch,
            on_checkpoint=on_checkpoint,
            device=device,
            resume_state=resume_state,
        )
        return TrainingResult(
            status=run_status,
            best_epoch=None,
            best_metric=None,
            final_evaluation=final_evaluation,
        )

    def evaluate(
        self,
        loaders: TaskLoaders,
        job: Any,
        workdir: Path,
        training_result: TrainingResult,
    ) -> EvaluationResult:
        evaluation: ClassifierEvaluation | None = training_result.final_evaluation
        if evaluation is None:
            return EvaluationResult(summary={}, per_class={}, raw=None)
        return EvaluationResult(
            summary={
                "accuracy": float(evaluation.accuracy),
                "macro_f1": float(evaluation.macro_f1),
                "macro_precision": float(evaluation.macro_precision),
                "macro_recall": float(evaluation.macro_recall),
            },
            per_class={},
            raw=evaluation,
        )

    def write_evaluation(
        self,
        storage: Any,
        *,
        project_id: str,
        experiment_id: str,
        attempt: int,
        loaders: TaskLoaders,
        training_result: TrainingResult,
        eval_result: EvaluationResult,
    ) -> None:
        if eval_result.raw is None:
            return
        write_classification_evaluation(
            storage,
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=attempt,
            class_order=loaders.class_order,
            class_names=loaders.class_names,
            evaluation=eval_result.raw,
        )

    def export_onnx(
        self,
        storage: Any,
        *,
        project_id: str,
        experiment_id: str,
        attempt: int,
        job: Any,
        loaders: TaskLoaders,
    ) -> OnnxExportResult:
        return export_best_classification_onnx(
            storage,
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=attempt,
            model_config=job.model_config,
            num_classes=loaders.num_classes,
            class_names=loaders.class_names,
            class_order=loaders.class_order,
        )


PIPELINE_REGISTRY["classification"] = ClassificationPipeline()
