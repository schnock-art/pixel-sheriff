from __future__ import annotations

from typing import Any

from pixel_sheriff_trainer.classification.train import (
    resolve_device,
    resolve_runtime_info,
)
from pixel_sheriff_trainer.io.checkpoints import AsyncCheckpointWriter, compact_completed_checkpoints, read_checkpoints
from pixel_sheriff_trainer.io.events import EventLog
from pixel_sheriff_trainer.io.metrics import append_metric
from pixel_sheriff_trainer.io.run_logging import RunLogger
from pixel_sheriff_trainer.io.runtime import write_runtime_info
from pixel_sheriff_trainer.io.storage import ExperimentStorage
from pixel_sheriff_trainer.jobs import TrainJob
from pixel_sheriff_trainer.pipeline import PIPELINE_REGISTRY
from pixel_sheriff_trainer.utils.seed import seed_everything
from pixel_sheriff_trainer.utils.time import utc_now_iso


class TrainRunner:
    def __init__(self, storage_root: str) -> None:
        self.storage = ExperimentStorage(storage_root)
        self.events = EventLog(self.storage)

    def _status_summary(self, status: str, attempt: int, job_id: str, message: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "status",
            "status": status,
            "attempt": attempt,
            "job_id": job_id,
            "ts": utc_now_iso(),
        }
        if message:
            payload["message"] = message
        return payload

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
        runtime = training_config.get("runtime")
        if isinstance(runtime, dict) and runtime.get("num_workers") is not None:
            try:
                return max(0, int(runtime.get("num_workers")))
            except (TypeError, ValueError):
                return 0
        advanced = training_config.get("advanced")
        if isinstance(advanced, dict) and advanced.get("num_workers") is not None:
            try:
                return max(0, int(advanced.get("num_workers")))
            except (TypeError, ValueError):
                return 0
        return 0

    @staticmethod
    def _config_with_num_workers(training_config: dict[str, Any], num_workers: int) -> dict[str, Any]:
        next_config = dict(training_config)

        runtime = next_config.get("runtime")
        next_runtime = dict(runtime) if isinstance(runtime, dict) else {}
        next_runtime["num_workers"] = max(0, int(num_workers))
        next_config["runtime"] = next_runtime

        advanced = next_config.get("advanced")
        next_advanced = dict(advanced) if isinstance(advanced, dict) else {}
        next_advanced["num_workers"] = max(0, int(num_workers))
        next_config["advanced"] = next_advanced
        return next_config

    @staticmethod
    def _checkpoint_settings(training_config: dict[str, Any]) -> tuple[int, int]:
        logging_cfg = training_config.get("logging")
        if not isinstance(logging_cfg, dict):
            logging_cfg = {}
        save_every = logging_cfg.get("save_every_epochs", 1)
        keep_last = logging_cfg.get("keep_last", 1)
        try:
            save_every_int = max(1, int(save_every))
        except (TypeError, ValueError):
            save_every_int = 1
        try:
            keep_last_int = max(1, int(keep_last))
        except (TypeError, ValueError):
            keep_last_int = 1
        return save_every_int, keep_last_int

    def _try_load_resume_state(
        self,
        *,
        project_id: str,
        experiment_id: str,
        attempt: int,
        training_config: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        resume_cfg = training_config.get("resume")
        if not isinstance(resume_cfg, dict):
            return None, None
        if not bool(resume_cfg.get("enabled", False)):
            return None, None

        resume_kind = str(resume_cfg.get("checkpoint_kind", "latest") or "latest").strip().lower()
        if resume_kind not in {"latest", "best_loss", "best_metric"}:
            resume_kind = "latest"

        rows = read_checkpoints(
            self.storage,
            project_id=project_id,
            experiment_id=experiment_id,
            attempt=attempt,
        )
        row = next((item for item in rows if str(item.get("kind")) == resume_kind), None)
        uri = str(row.get("uri") or "") if isinstance(row, dict) else ""
        if not uri:
            return None, "resume requested but checkpoint invalid; starting fresh"

        try:
            import torch

            checkpoint_path = self.storage.resolve(uri)
            payload = torch.load(checkpoint_path, map_location="cpu")
            if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
                return None, "resume requested but checkpoint invalid; starting fresh"
            loaded_epoch = payload.get("epoch")
            epoch_label = int(loaded_epoch) if isinstance(loaded_epoch, int) else "unknown"
            return payload, f"resume applied from {resume_kind} checkpoint (epoch={epoch_label})"
        except Exception:
            return None, "resume requested but checkpoint invalid; starting fresh"

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

        run_logger = RunLogger(self.storage.training_log_path(job.project_id, job.experiment_id, job.attempt))
        run_logger.log(f"run_started project={job.project_id} experiment={job.experiment_id} attempt={job.attempt}")

        summary: dict[str, Any] = {
            "best_metric_name": None,
            "best_metric_value": None,
            "best_epoch": None,
            "last_epoch": None,
        }

        checkpoint_writer: AsyncCheckpointWriter | None = None

        def emit_status(message: str) -> None:
            self.events.append(
                job.project_id,
                job.experiment_id,
                job.attempt,
                self._status_summary("running", job.attempt, job.job_id, message=message),
            )
            run_logger.log(message)

        def drain_checkpoint_results() -> None:
            if checkpoint_writer is None:
                return
            for result in checkpoint_writer.drain_results():
                if result.dropped:
                    emit_status("checkpoint latest dropped due to full writer queue")
                    continue
                if result.ok and isinstance(result.row, dict):
                    self.events.append(job.project_id, job.experiment_id, job.attempt, {"type": "checkpoint", **result.row})
                    run_logger.log(
                        f"checkpoint_written kind={result.kind} epoch={result.epoch} uri={result.row.get('uri')} status={result.row.get('status')}"
                    )
                    continue
                if isinstance(result.row, dict):
                    self.events.append(job.project_id, job.experiment_id, job.attempt, {"type": "checkpoint", **result.row})
                emit_status(f"checkpoint write failed kind={result.kind} epoch={result.epoch}: {result.error}")

        def should_cancel() -> bool:
            return self.storage.is_cancel_requested(job.project_id, job.experiment_id)

        try:
            import dataclasses

            task = str(job.task or "").lower()
            pipeline = PIPELINE_REGISTRY.get(task)
            if pipeline is None:
                raise ValueError(f"unsupported_task:{task}")

            advanced = job.training_config.get("advanced")
            seed = 1337
            if isinstance(advanced, dict) and isinstance(advanced.get("seed"), int):
                seed = int(advanced["seed"])
            seed_everything(seed)

            device = resolve_device(job.training_config)
            effective_training_config = job.training_config
            runtime_info = resolve_runtime_info(effective_training_config, device=device)
            runtime_payload = {
                "device_selected": runtime_info.device_selected,
                "cuda_available": runtime_info.cuda_available,
                "mps_available": runtime_info.mps_available,
                "amp_enabled": runtime_info.amp_enabled,
                "torch_version": runtime_info.torch_version,
                "torchvision_version": runtime_info.torchvision_version,
                "num_workers": runtime_info.num_workers,
                "pin_memory": runtime_info.pin_memory,
                "persistent_workers": runtime_info.persistent_workers,
                "prefetch_factor": runtime_info.prefetch_factor,
                "cache_resized_images": runtime_info.cache_resized_images,
                "max_cached_images": runtime_info.max_cached_images,
            }
            write_runtime_info(
                self.storage,
                project_id=job.project_id,
                experiment_id=job.experiment_id,
                attempt=job.attempt,
                payload=runtime_payload,
            )
            run_logger.log(
                "runtime "
                f"device={runtime_info.device_selected} amp={runtime_info.amp_enabled} "
                f"cuda_available={runtime_info.cuda_available} mps_available={runtime_info.mps_available} "
                f"torch={runtime_info.torch_version} torchvision={runtime_info.torchvision_version}"
            )

            workdir = self.storage.run_dir(job.project_id, job.experiment_id, job.attempt) / "workdir"
            effective_job = dataclasses.replace(job, training_config=effective_training_config)
            loaders = pipeline.build_loaders(effective_job, workdir)
            run_logger.log(f"dataset train_count={loaders.train_count} val_count={loaders.val_count}")

            if loaders.skipped_unlabeled > 0:
                emit_status(f"skipped_unlabeled={loaders.skipped_unlabeled}")

            _save_every, keep_last = self._checkpoint_settings(effective_training_config)
            checkpoint_writer = AsyncCheckpointWriter(
                self.storage,
                project_id=job.project_id,
                experiment_id=job.experiment_id,
                attempt=job.attempt,
                keep_last=keep_last,
                max_queue_size=8,
            )
            checkpoint_writer.start()

            def on_epoch(epoch_row: dict[str, Any]) -> None:
                metric_row: dict[str, Any] = {
                    "attempt": job.attempt,
                    "created_at": utc_now_iso(),
                    **epoch_row,
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
                epoch_val = epoch_row.get("epoch")
                if isinstance(epoch_val, int):
                    summary["last_epoch"] = epoch_val
                self.storage.set_summary(job.project_id, job.experiment_id, summary)
                run_logger.log(
                    f"epoch={epoch_row.get('epoch')} "
                    f"train_loss={epoch_row.get('train_loss')} "
                    f"lr={epoch_row.get('lr')} "
                    f"seconds={epoch_row.get('epoch_seconds')}"
                )
                drain_checkpoint_results()

            def on_checkpoint(kind: str, epoch: int, metric_name: str | None, value: float | None, state: dict[str, Any]) -> None:
                if kind == "best_metric":
                    summary["best_metric_name"] = metric_name
                    summary["best_metric_value"] = value
                    summary["best_epoch"] = epoch
                    self.storage.set_summary(job.project_id, job.experiment_id, summary)

                if checkpoint_writer is None:
                    return
                checkpoint_writer.enqueue(
                    kind=kind,
                    epoch=epoch,
                    metric_name=metric_name,
                    value=value,
                    state_dict=state,
                )

            resume_state, resume_message = self._try_load_resume_state(
                project_id=job.project_id,
                experiment_id=job.experiment_id,
                attempt=job.attempt,
                training_config=effective_training_config,
            )
            if resume_message:
                emit_status(resume_message)

            try:
                training_result = pipeline.run_training(
                    loaders,
                    effective_job,
                    workdir,
                    on_epoch=on_epoch,
                    on_checkpoint=on_checkpoint,
                    should_cancel=should_cancel,
                    device=device,
                    resume_state=resume_state,
                )
            except RuntimeError as exc:
                message = str(exc)
                if "shared memory" not in message.lower() or self._num_workers_from_config(effective_training_config) <= 0:
                    raise
                effective_training_config = self._config_with_num_workers(effective_training_config, 0)
                emit_status("shared-memory error detected; retrying with num_workers=0")

                runtime_info = resolve_runtime_info(effective_training_config, device=device)
                runtime_payload = {
                    "device_selected": runtime_info.device_selected,
                    "cuda_available": runtime_info.cuda_available,
                    "mps_available": runtime_info.mps_available,
                    "amp_enabled": runtime_info.amp_enabled,
                    "torch_version": runtime_info.torch_version,
                    "torchvision_version": runtime_info.torchvision_version,
                    "num_workers": runtime_info.num_workers,
                    "pin_memory": runtime_info.pin_memory,
                    "persistent_workers": runtime_info.persistent_workers,
                    "prefetch_factor": runtime_info.prefetch_factor,
                    "cache_resized_images": runtime_info.cache_resized_images,
                    "max_cached_images": runtime_info.max_cached_images,
                }
                write_runtime_info(
                    self.storage,
                    project_id=job.project_id,
                    experiment_id=job.experiment_id,
                    attempt=job.attempt,
                    payload=runtime_payload,
                )

                effective_job = dataclasses.replace(job, training_config=effective_training_config)
                loaders = pipeline.build_loaders(effective_job, workdir)
                training_result = pipeline.run_training(
                    loaders,
                    effective_job,
                    workdir,
                    on_epoch=on_epoch,
                    on_checkpoint=on_checkpoint,
                    should_cancel=should_cancel,
                    device=device,
                    resume_state=resume_state,
                )

            if checkpoint_writer is not None:
                checkpoint_writer.flush_and_stop()
            drain_checkpoint_results()

            run_status = training_result.status
            if run_status == "canceled":
                self.storage.set_experiment_status(job.project_id, job.experiment_id, "canceled")
                self.events.append(
                    job.project_id,
                    job.experiment_id,
                    job.attempt,
                    self._done_event("canceled", job.attempt, job.job_id),
                )
                self.storage.set_run_ended(job.project_id, job.experiment_id, job.attempt)
                run_logger.log("run_finished status=canceled")
                return "canceled"

            eval_result = pipeline.evaluate(loaders, effective_job, workdir, training_result)
            pipeline.write_evaluation(
                self.storage,
                project_id=job.project_id,
                experiment_id=job.experiment_id,
                attempt=job.attempt,
                job=job,
                loaders=loaders,
                training_result=training_result,
                eval_result=eval_result,
            )

            onnx_result = pipeline.export_onnx(
                self.storage,
                project_id=job.project_id,
                experiment_id=job.experiment_id,
                attempt=job.attempt,
                job=effective_job,
                loaders=loaders,
            )
            onnx_event: dict[str, Any] = {
                "type": "onnx_export",
                "status": onnx_result.status,
                "attempt": job.attempt,
                "metadata_uri": onnx_result.metadata_uri,
                "ts": utc_now_iso(),
            }
            if onnx_result.model_uri:
                onnx_event["model_uri"] = onnx_result.model_uri
            if onnx_result.error:
                onnx_event["error"] = onnx_result.error
            self.events.append(job.project_id, job.experiment_id, job.attempt, onnx_event)
            if onnx_result.status == "exported":
                run_logger.log(
                    f"onnx_export status=exported model_uri={onnx_result.model_uri} metadata_uri={onnx_result.metadata_uri}"
                )
            else:
                run_logger.log(f"onnx_export status=failed error={onnx_result.error}")

            try:
                compacted_kinds = compact_completed_checkpoints(
                    self.storage,
                    project_id=job.project_id,
                    experiment_id=job.experiment_id,
                    attempt=job.attempt,
                )
            except Exception as exc:
                run_logger.log(f"checkpoint_compaction_failed error={exc}")
            else:
                if compacted_kinds:
                    run_logger.log(f"checkpoint_compacted kinds={','.join(compacted_kinds)}")

            self.storage.set_experiment_status(job.project_id, job.experiment_id, "completed")
            self.events.append(
                job.project_id,
                job.experiment_id,
                job.attempt,
                self._done_event("completed", job.attempt, job.job_id),
            )
            self.storage.set_run_ended(job.project_id, job.experiment_id, job.attempt)
            run_logger.log("run_finished status=completed")
            return "completed"
        except ValueError as exc:
            code = str(exc)
            if code.startswith("unsupported_task"):
                message = f"Unsupported task: {code}"
            elif code == "unsupported_family":
                message = "Model family not supported for this task"
            elif code == "batchnorm_small_batch_unsupported":
                message = "BatchNorm training requires effective batch size >= 2. Increase batch size or enable training.drop_last"
            else:
                message = code
            if checkpoint_writer is not None:
                checkpoint_writer.flush_and_stop()
                drain_checkpoint_results()
            self.storage.set_experiment_status(job.project_id, job.experiment_id, "failed", error=message)
            self.events.append(
                job.project_id,
                job.experiment_id,
                job.attempt,
                self._done_event("failed", job.attempt, job.job_id, error_code=code, message=message),
            )
            self.storage.set_run_ended(job.project_id, job.experiment_id, job.attempt)
            run_logger.log(f"run_finished status=failed code={code} message={message}")
            return f"failed:{code}"
        except Exception as exc:
            if checkpoint_writer is not None:
                checkpoint_writer.flush_and_stop()
                drain_checkpoint_results()
            self.storage.set_experiment_status(job.project_id, job.experiment_id, "failed", error=str(exc))
            self.events.append(
                job.project_id,
                job.experiment_id,
                job.attempt,
                self._done_event("failed", job.attempt, job.job_id, error_code="trainer_error", message=str(exc)),
            )
            self.storage.set_run_ended(job.project_id, job.experiment_id, job.attempt)
            run_logger.log(f"run_finished status=failed code=trainer_error message={exc}")
            return f"failed:trainer_error:{exc}"
