from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from pixel_sheriff_trainer.detection.dataset import build_detection_loaders
from pixel_sheriff_trainer.detection.eval import DetectionEvaluation, evaluate_detection
from pixel_sheriff_trainer.detection.train import (
    DetectionEpochMetrics,
    _build_detection_model,
    run_detection_training,
)
from pixel_sheriff_trainer.export_onnx import OnnxExportResult, _resolve_best_checkpoint, _as_relative_uri
from pixel_sheriff_trainer.io.storage import ExperimentStorage
from pixel_sheriff_trainer.pipeline import (
    EvaluationResult,
    PIPELINE_REGISTRY,
    TaskLoaders,
    TaskPipeline,
    TrainingResult,
)
from pixel_sheriff_trainer.utils.torchvision_cache import configure_torchvision_cache
from pixel_sheriff_trainer.utils.time import utc_now_iso


class DetectionPipeline(TaskPipeline):
    task_kind = "detection"

    def build_loaders(self, job: Any, workdir: Path) -> TaskLoaders:
        from pixel_sheriff_trainer.io.storage import ExperimentStorage

        storage_root = workdir.parents[5]
        storage = ExperimentStorage(str(storage_root))

        zip_relpath = str(job.dataset_export.get("zip_relpath") or "")
        if not zip_relpath:
            raise ValueError("dataset_export_missing")
        zip_path = storage.resolve(zip_relpath)

        loaded = build_detection_loaders(
            export_zip_path=zip_path,
            workdir=workdir,
            model_config=job.model_config,
            training_config=job.training_config,
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
        def _on_epoch(epoch_metrics: DetectionEpochMetrics) -> None:
            row: dict[str, Any] = {
                "epoch": int(epoch_metrics.epoch),
                "train_loss": float(epoch_metrics.train_loss),
                "val_map": epoch_metrics.mAP50,
                "val_map_50_95": epoch_metrics.mAP50_95,
                "lr": float(epoch_metrics.lr),
                "epoch_seconds": float(epoch_metrics.epoch_seconds),
                "eta_seconds": epoch_metrics.eta_seconds,
                "evaluated": bool(epoch_metrics.evaluated),
            }
            on_epoch(row)

        run_status, final_evaluation = run_detection_training(
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
        evaluation: DetectionEvaluation | None = training_result.final_evaluation
        if evaluation is None:
            return EvaluationResult(summary={}, per_class={}, raw=None)
        return EvaluationResult(
            summary={
                "mAP50": float(evaluation.mAP50),
                "mAP50_95": float(evaluation.mAP50_95),
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
        job: Any,
        loaders: TaskLoaders,
        training_result: TrainingResult,
        eval_result: EvaluationResult,
    ) -> None:
        if eval_result.raw is None:
            return
        evaluation: DetectionEvaluation = eval_result.raw
        payload = {
            "schema_version": "1",
            "task": "detection",
            "computed_at": utc_now_iso(),
            "split": "val",
            "classes": {
                "class_order": loaders.class_order,
                "class_names": loaders.class_names,
            },
            "overall": {
                "mAP50": float(evaluation.mAP50),
                "mAP50_95": float(evaluation.mAP50_95),
            },
            "per_class": evaluation.per_class,
        }
        eval_path = storage.evaluation_path(project_id, experiment_id, attempt)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        latest_eval_path = storage.evaluation_path(project_id, experiment_id, None)
        latest_eval_path.parent.mkdir(parents=True, exist_ok=True)
        latest_eval_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

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
        import torch
        from pixel_sheriff_trainer.export_onnx import (
            OnnxExportResult,
            _input_shape_from_model,
            _preprocess_from_model,
            _resolve_best_checkpoint,
            _as_relative_uri,
            export_model_to_onnx,
        )

        configure_torchvision_cache(str(storage.root))

        checkpoint_kind, checkpoint_path = _resolve_best_checkpoint(
            storage, project_id=project_id, experiment_id=experiment_id, attempt=attempt,
        )
        if checkpoint_path is None:
            from pixel_sheriff_trainer.utils.time import utc_now_iso
            onnx_dir = storage.run_dir(project_id, experiment_id, attempt) / "onnx"
            onnx_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = onnx_dir / "onnx.metadata.json"
            metadata_path.write_text(json.dumps({
                "schema_version": "1", "status": "failed",
                "error": "No checkpoint found for ONNX export",
                "exported_at": utc_now_iso(),
            }, indent=2), encoding="utf-8")
            return OnnxExportResult(
                status="failed", attempt=attempt, model_uri=None,
                metadata_uri=_as_relative_uri(storage, metadata_path),
                error="No checkpoint found for ONNX export", validation=None,
            )

        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = (
            checkpoint_payload.get("model_state_dict")
            if isinstance(checkpoint_payload, dict)
            else checkpoint_payload
        )

        model = _build_detection_model(job.model_config, num_classes=loaders.num_classes)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        input_shape = _input_shape_from_model(job.model_config)
        preprocess = _preprocess_from_model(job.model_config, input_shape=input_shape)

        return export_model_to_onnx(
            model, storage,
            project_id=project_id, experiment_id=experiment_id, attempt=attempt,
            checkpoint_kind=checkpoint_kind,
            checkpoint_uri=str(checkpoint_path.relative_to(storage.root)).replace("\\", "/"),
            input_shape=input_shape,
            input_names=["input"],
            output_names=["output"],
            preprocess=preprocess,
            class_order=loaders.class_order,
            class_names=loaders.class_names,
            extra_metadata={"task": "detection"},
        )


PIPELINE_REGISTRY["detection"] = DetectionPipeline()
