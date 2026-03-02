from __future__ import annotations

from typing import Any

from pixel_sheriff_trainer.classification.dataset import build_classification_loaders
from pixel_sheriff_trainer.classification.train import EpochMetrics, run_training
from pixel_sheriff_trainer.io.checkpoints import save_checkpoint
from pixel_sheriff_trainer.io.evaluation import write_classification_evaluation
from pixel_sheriff_trainer.io.events import EventLog
from pixel_sheriff_trainer.io.metrics import append_metric
from pixel_sheriff_trainer.io.storage import ExperimentStorage
from pixel_sheriff_trainer.jobs import TrainJob
from pixel_sheriff_trainer.utils.seed import seed_everything
from pixel_sheriff_trainer.utils.time import utc_now_iso


class TrainRunner:
    def __init__(self, storage_root: str) -> None:
        self.storage = ExperimentStorage(storage_root)
        self.events = EventLog(self.storage)

    def _status_summary(self, status: str, attempt: int, job_id: str) -> dict[str, Any]:
        return {"type": "status", "status": status, "attempt": attempt, "job_id": job_id, "ts": utc_now_iso()}

    def _done_event(
        self,
        status: str,
        attempt: int,
        job_id: str,
        *,
        error_code: str | None = None,
        message: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "done",
            "status": status,
            "attempt": attempt,
            "job_id": job_id,
            "ts": utc_now_iso(),
        }
        if error_code:
            payload["error_code"] = error_code
        if message:
            payload["message"] = message
        return payload

    @staticmethod
    def _num_workers_from_config(training_config: dict[str, Any]) -> int:
        advanced = training_config.get("advanced")
        if isinstance(advanced, dict) and isinstance(advanced.get("num_workers"), int):
            return max(0, int(advanced["num_workers"]))
        return 0

    @staticmethod
    def _config_with_num_workers(training_config: dict[str, Any], num_workers: int) -> dict[str, Any]:
        next_config = dict(training_config)
        advanced = next_config.get("advanced")
        next_advanced = dict(advanced) if isinstance(advanced, dict) else {}
        next_advanced["num_workers"] = max(0, int(num_workers))
        next_config["advanced"] = next_advanced
        return next_config

    def process(self, job: TrainJob) -> str:
        status_row = self.storage.read_status(job.project_id, job.experiment_id)
        if status_row.get("active_job_id") != job.job_id:
            return "ignored:stale_job_id"
        if int(status_row.get("current_run_attempt") or 0) != int(job.attempt):
            return "ignored:stale_attempt"
        if str(status_row.get("status")) in {"running", "completed"}:
            return "ignored:already_started"
        if str(status_row.get("status")) != "queued":
            return f"ignored:status={status_row.get('status')}"

        if self.storage.is_cancel_requested(job.project_id, job.experiment_id):
            self.storage.set_experiment_status(job.project_id, job.experiment_id, "canceled")
            self.events.append(
                job.project_id,
                job.experiment_id,
                job.attempt,
                self._done_event("canceled", job.attempt, job.job_id),
            )
            return "canceled:before_start"

        self.storage.set_experiment_status(job.project_id, job.experiment_id, "running")
        self.storage.set_run_started(job.project_id, job.experiment_id, job.attempt)
        self.events.append(job.project_id, job.experiment_id, job.attempt, self._status_summary("running", job.attempt, job.job_id))

        summary: dict[str, Any] = {
            "best_metric_name": None,
            "best_metric_value": None,
            "best_epoch": None,
            "last_epoch": None,
        }

        def should_cancel() -> bool:
            return self.storage.is_cancel_requested(job.project_id, job.experiment_id)

        try:
            task = str(job.task or "").lower()
            family = str((job.model_config.get("architecture") or {}).get("family") or "").lower()
            if task != "classification":
                raise ValueError("unsupported_task")
            if family != "resnet_classifier":
                raise ValueError("unsupported_family")

            advanced = job.training_config.get("advanced")
            seed = 1337
            if isinstance(advanced, dict) and isinstance(advanced.get("seed"), int):
                seed = int(advanced["seed"])
            seed_everything(seed)

            zip_relpath = str(job.dataset_export.get("zip_relpath") or "")
            if not zip_relpath:
                raise ValueError("dataset_export_missing")
            zip_path = self.storage.resolve(zip_relpath)
            workdir = self.storage.run_dir(job.project_id, job.experiment_id, job.attempt) / "workdir"
            effective_training_config = job.training_config
            loaded = build_classification_loaders(
                export_zip_path=zip_path,
                workdir=workdir,
                model_config=job.model_config,
                training_config=effective_training_config,
            )

            if loaded.skipped_unlabeled > 0:
                self.events.append(
                    job.project_id,
                    job.experiment_id,
                    job.attempt,
                    {
                        "type": "status",
                        "status": "running",
                        "attempt": job.attempt,
                        "job_id": job.job_id,
                        "ts": utc_now_iso(),
                        "message": f"skipped_unlabeled={loaded.skipped_unlabeled}",
                    },
                )

            def on_epoch(epoch_metrics: EpochMetrics) -> None:
                metric_row: dict[str, Any] = {
                    "attempt": job.attempt,
                    "epoch": int(epoch_metrics.epoch),
                    "train_loss": float(epoch_metrics.train_loss),
                    "val_loss": float(epoch_metrics.val_loss),
                    "val_accuracy": float(epoch_metrics.val_accuracy),
                    "val_macro_f1": float(epoch_metrics.val_macro_f1),
                    "val_macro_precision": float(epoch_metrics.val_macro_precision),
                    "val_macro_recall": float(epoch_metrics.val_macro_recall),
                    "created_at": utc_now_iso(),
                }
                append_metric(
                    self.storage,
                    project_id=job.project_id,
                    experiment_id=job.experiment_id,
                    attempt=job.attempt,
                    metric_row=metric_row,
                )
                self.events.append(
                    job.project_id,
                    job.experiment_id,
                    job.attempt,
                    {"type": "metric", **metric_row},
                )
                summary["last_epoch"] = int(epoch_metrics.epoch)
                self.storage.set_summary(job.project_id, job.experiment_id, summary)

            def on_checkpoint(kind: str, epoch: int, metric_name: str | None, value: float | None, state: dict[str, Any]) -> None:
                row = save_checkpoint(
                    self.storage,
                    project_id=job.project_id,
                    experiment_id=job.experiment_id,
                    attempt=job.attempt,
                    kind=kind,
                    epoch=epoch,
                    metric_name=metric_name,
                    value=value,
                    state_dict=state,
                )
                if kind == "best_metric":
                    summary["best_metric_name"] = metric_name
                    summary["best_metric_value"] = value
                    summary["best_epoch"] = epoch
                    self.storage.set_summary(job.project_id, job.experiment_id, summary)
                self.events.append(job.project_id, job.experiment_id, job.attempt, {"type": "checkpoint", **row})

            final_evaluation = None
            try:
                run_status, final_evaluation = run_training(
                    model_config=job.model_config,
                    training_config=effective_training_config,
                    train_loader=loaded.train_loader,
                    val_loader=loaded.val_loader,
                    num_classes=loaded.num_classes,
                    should_cancel=should_cancel,
                    on_epoch=on_epoch,
                    on_checkpoint=on_checkpoint,
                )
            except RuntimeError as exc:
                message = str(exc)
                if "shared memory" not in message.lower() or self._num_workers_from_config(effective_training_config) <= 0:
                    raise
                effective_training_config = self._config_with_num_workers(effective_training_config, 0)
                self.events.append(
                    job.project_id,
                    job.experiment_id,
                    job.attempt,
                    {
                        "type": "status",
                        "status": "running",
                        "attempt": job.attempt,
                        "job_id": job.job_id,
                        "ts": utc_now_iso(),
                        "message": "shared-memory error detected; retrying with num_workers=0",
                    },
                )
                loaded = build_classification_loaders(
                    export_zip_path=zip_path,
                    workdir=workdir,
                    model_config=job.model_config,
                    training_config=effective_training_config,
                )
                run_status, final_evaluation = run_training(
                    model_config=job.model_config,
                    training_config=effective_training_config,
                    train_loader=loaded.train_loader,
                    val_loader=loaded.val_loader,
                    num_classes=loaded.num_classes,
                    should_cancel=should_cancel,
                    on_epoch=on_epoch,
                    on_checkpoint=on_checkpoint,
                )
            if run_status == "canceled":
                self.storage.set_experiment_status(job.project_id, job.experiment_id, "canceled")
                self.events.append(
                    job.project_id,
                    job.experiment_id,
                    job.attempt,
                    self._done_event("canceled", job.attempt, job.job_id),
                )
                self.storage.set_run_ended(job.project_id, job.experiment_id, job.attempt)
                return "canceled"

            if final_evaluation is not None:
                write_classification_evaluation(
                    self.storage,
                    project_id=job.project_id,
                    experiment_id=job.experiment_id,
                    attempt=job.attempt,
                    class_order=loaded.class_order,
                    class_names=loaded.class_names,
                    evaluation=final_evaluation,
                )

            self.storage.set_experiment_status(job.project_id, job.experiment_id, "completed")
            self.events.append(
                job.project_id,
                job.experiment_id,
                job.attempt,
                self._done_event("completed", job.attempt, job.job_id),
            )
            self.storage.set_run_ended(job.project_id, job.experiment_id, job.attempt)
            return "completed"
        except ValueError as exc:
            code = str(exc)
            if code == "unsupported_task":
                message = "Only classification training is supported in this phase"
            elif code == "unsupported_family":
                message = "Only resnet_classifier models are supported in this phase"
            else:
                message = code
            self.storage.set_experiment_status(job.project_id, job.experiment_id, "failed", error=message)
            self.events.append(
                job.project_id,
                job.experiment_id,
                job.attempt,
                self._done_event("failed", job.attempt, job.job_id, error_code=code, message=message),
            )
            self.storage.set_run_ended(job.project_id, job.experiment_id, job.attempt)
            return f"failed:{code}"
        except Exception as exc:
            self.storage.set_experiment_status(job.project_id, job.experiment_id, "failed", error=str(exc))
            self.events.append(
                job.project_id,
                job.experiment_id,
                job.attempt,
                self._done_event("failed", job.attempt, job.job_id, error_code="trainer_error", message=str(exc)),
            )
            self.storage.set_run_ended(job.project_id, job.experiment_id, job.attempt)
            return f"failed:trainer_error:{exc}"
