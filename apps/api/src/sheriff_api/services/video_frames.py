from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sheriff_api.config import get_settings
from sheriff_api.db.models import Annotation, Asset, AssetSequence, AssetType, Folder, Suggestion
from sheriff_api.db.session import SessionLocal
from sheriff_api.services.asset_ingest import build_asset_record
from sheriff_api.services.storage import LocalStorage

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
DEFAULT_IMPORT_FPS = 2.0
DEFAULT_IMPORT_MAX_FRAMES = 500
MAX_IMPORT_FPS = 10.0
MAX_IMPORT_FRAMES = 5000


class VideoImportValidationError(ValueError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class VideoFrameExtractionError(RuntimeError):
    pass


def validate_video_import_params(
    *,
    filename: str | None,
    fps: float,
    max_frames: int,
    resize_mode: str,
    resize_width: int | None,
    resize_height: int | None,
) -> dict[str, Any]:
    suffix = Path(str(filename or "")).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise VideoImportValidationError(
            code="video_import_type_unsupported",
            message="Unsupported video format",
            details={"filename": filename, "allowed_extensions": sorted(ALLOWED_VIDEO_EXTENSIONS)},
        )
    if fps <= 0 or fps > MAX_IMPORT_FPS:
        raise VideoImportValidationError(
            code="video_import_fps_invalid",
            message=f"FPS must be > 0 and <= {MAX_IMPORT_FPS}",
            details={"fps": fps},
        )
    if max_frames <= 0 or max_frames > MAX_IMPORT_FRAMES:
        raise VideoImportValidationError(
            code="video_import_max_frames_invalid",
            message=f"Max frames must be > 0 and <= {MAX_IMPORT_FRAMES}",
            details={"max_frames": max_frames},
        )

    normalized_resize_mode = str(resize_mode or "original").strip().lower()
    if normalized_resize_mode not in {"original", "width", "height"}:
        raise VideoImportValidationError(
            code="video_import_resize_mode_invalid",
            message="Resize mode must be original, width, or height",
            details={"resize_mode": resize_mode},
        )
    if normalized_resize_mode == "width" and (resize_width is None or resize_width <= 0):
        raise VideoImportValidationError(
            code="video_import_resize_width_invalid",
            message="Resize width must be a positive integer when resize_mode=width",
            details={"resize_width": resize_width},
        )
    if normalized_resize_mode == "height" and (resize_height is None or resize_height <= 0):
        raise VideoImportValidationError(
            code="video_import_resize_height_invalid",
            message="Resize height must be a positive integer when resize_mode=height",
            details={"resize_height": resize_height},
        )

    return {
        "fps": float(fps),
        "max_frames": int(max_frames),
        "resize_mode": normalized_resize_mode,
        "resize_width": int(resize_width) if resize_width is not None else None,
        "resize_height": int(resize_height) if resize_height is not None else None,
        "extension": suffix,
    }


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def _parse_fraction(value: str | None) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def probe_video_metadata(input_path: Path) -> dict[str, Any]:
    result = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,duration:format=duration",
            "-of",
            "json",
            str(input_path),
        ]
    )
    if result.returncode != 0:
        raise VideoFrameExtractionError(result.stderr.strip() or "ffprobe failed")

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise VideoFrameExtractionError("ffprobe returned invalid JSON") from exc

    streams = payload.get("streams")
    stream = streams[0] if isinstance(streams, list) and streams else {}
    fmt = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    duration_value = stream.get("duration") if isinstance(stream, dict) else None
    if duration_value in {None, ""}:
        duration_value = fmt.get("duration")

    fps = _parse_fraction(stream.get("avg_frame_rate") if isinstance(stream, dict) else None)
    duration = None
    try:
        if duration_value not in {None, ""}:
            duration = float(duration_value)
    except (TypeError, ValueError):
        duration = None

    return {
        "width": int(stream.get("width")) if isinstance(stream, dict) and stream.get("width") is not None else None,
        "height": int(stream.get("height")) if isinstance(stream, dict) and stream.get("height") is not None else None,
        "duration_seconds": duration,
        "fps": fps,
    }


def build_ffmpeg_command(
    *,
    input_path: Path,
    output_pattern: str,
    fps: float,
    max_frames: int,
    resize_mode: str,
    resize_width: int | None,
    resize_height: int | None,
) -> list[str]:
    filters = [f"fps={fps:g}"]
    if resize_mode == "width" and resize_width is not None:
        filters.append(f"scale={resize_width}:-1")
    elif resize_mode == "height" and resize_height is not None:
        filters.append(f"scale=-1:{resize_height}")

    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        ",".join(filters),
        "-frames:v",
        str(max_frames),
        output_pattern,
    ]


async def _mark_sequence_failed(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    sequence_id: str,
    error_message: str,
) -> None:
    async with session_factory() as db:
        sequence = await db.get(AssetSequence, sequence_id)
        if sequence is None:
            return
        sequence.status = "failed"
        sequence.error_message = error_message
        sequence.processed_frames = 0
        sequence.frame_count = 0
        await db.execute(delete(Annotation).where(Annotation.asset_id.in_(select(Asset.id).where(Asset.sequence_id == sequence_id))))
        await db.execute(delete(Suggestion).where(Suggestion.asset_id.in_(select(Asset.id).where(Asset.sequence_id == sequence_id))))
        await db.execute(delete(Asset).where(Asset.sequence_id == sequence_id))
        await db.commit()


async def extract_video_sequence_job(
    payload: dict[str, Any],
    *,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    storage: LocalStorage | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    effective_storage = storage or LocalStorage(settings.storage_root)
    effective_session_factory = session_factory or SessionLocal

    project_id = str(payload.get("project_id") or "").strip()
    sequence_id = str(payload.get("sequence_id") or "").strip()
    video_storage_uri = str(payload.get("video_storage_uri") or "").strip()
    if not project_id or not sequence_id or not video_storage_uri:
        raise VideoFrameExtractionError("project_id, sequence_id, and video_storage_uri are required")

    fps = float(payload.get("fps") or DEFAULT_IMPORT_FPS)
    max_frames = int(payload.get("max_frames") or DEFAULT_IMPORT_MAX_FRAMES)
    resize_mode = str(payload.get("resize_mode") or "original")
    resize_width = payload.get("resize_width")
    resize_height = payload.get("resize_height")
    try:
        resize_width = int(resize_width) if resize_width is not None else None
    except (TypeError, ValueError):
        resize_width = None
    try:
        resize_height = int(resize_height) if resize_height is not None else None
    except (TypeError, ValueError):
        resize_height = None

    try:
        validate_video_import_params(
            filename=video_storage_uri,
            fps=fps,
            max_frames=max_frames,
            resize_mode=resize_mode,
            resize_width=resize_width,
            resize_height=resize_height,
        )
    except VideoImportValidationError as exc:
        await _mark_sequence_failed(
            session_factory=effective_session_factory,
            sequence_id=sequence_id,
            error_message=exc.message,
        )
        raise VideoFrameExtractionError(exc.message) from exc

    input_path = effective_storage.resolve(video_storage_uri)
    if not input_path.exists():
        message = f"Video source is missing: {video_storage_uri}"
        await _mark_sequence_failed(session_factory=effective_session_factory, sequence_id=sequence_id, error_message=message)
        raise VideoFrameExtractionError(message)

    written_storage_uris: list[str] = []
    metadata = probe_video_metadata(input_path)

    with TemporaryDirectory(prefix=f"pixel_sheriff_frames_{sequence_id}_") as temp_dir:
        output_pattern = str(Path(temp_dir) / "frame_%06d.jpg")
        command = build_ffmpeg_command(
            input_path=input_path,
            output_pattern=output_pattern,
            fps=fps,
            max_frames=max_frames,
            resize_mode=resize_mode,
            resize_width=resize_width,
            resize_height=resize_height,
        )
        result = _run_command(command)
        if result.returncode != 0:
            message = result.stderr.strip() or "ffmpeg failed"
            await _mark_sequence_failed(
                session_factory=effective_session_factory,
                sequence_id=sequence_id,
                error_message=message,
            )
            effective_storage.delete_file(video_storage_uri)
            raise VideoFrameExtractionError(message)

        extracted_paths = sorted(Path(temp_dir).glob("frame_*.jpg"))
        if not extracted_paths:
            message = "No frames were extracted from the uploaded video"
            await _mark_sequence_failed(
                session_factory=effective_session_factory,
                sequence_id=sequence_id,
                error_message=message,
            )
            effective_storage.delete_file(video_storage_uri)
            raise VideoFrameExtractionError(message)

        async with effective_session_factory() as db:
            sequence = await db.get(AssetSequence, sequence_id)
            if sequence is None or sequence.project_id != project_id:
                raise VideoFrameExtractionError("Target sequence was not found")
            folder = await db.get(Folder, sequence.folder_id) if sequence.folder_id else None

            try:
                for index, frame_path in enumerate(extracted_paths):
                    content = frame_path.read_bytes()
                    timestamp_seconds = round(index / fps, 6) if fps > 0 else None
                    asset, storage_uri = build_asset_record(
                        project_id=project_id,
                        content=content,
                        file_name=frame_path.name,
                        mime_type="image/jpeg",
                        folder=folder,
                        original_filename=frame_path.name,
                        asset_type=AssetType.frame,
                        sequence_id=sequence.id,
                        sequence_name=sequence.name,
                        source_kind="video_frame",
                        frame_index=index,
                        timestamp_seconds=timestamp_seconds,
                    )
                    effective_storage.write_bytes(storage_uri, content)
                    written_storage_uris.append(storage_uri)
                    db.add(asset)

                sequence.status = "ready"
                sequence.error_message = None
                sequence.frame_count = len(extracted_paths)
                sequence.processed_frames = len(extracted_paths)
                sequence.fps = fps
                sequence.duration_seconds = metadata.get("duration_seconds")
                sequence.width = metadata.get("width")
                sequence.height = metadata.get("height")
                await db.commit()
            except Exception as exc:
                await db.rollback()
                for storage_uri in written_storage_uris:
                    try:
                        effective_storage.delete_file(storage_uri)
                    except ValueError:
                        pass
                message = str(exc) or "Failed to persist extracted frames"
                await _mark_sequence_failed(
                    session_factory=effective_session_factory,
                    sequence_id=sequence_id,
                    error_message=message,
                )
                effective_storage.delete_file(video_storage_uri)
                raise

    effective_storage.delete_file(video_storage_uri)
    return {
        "status": "ready",
        "sequence_id": sequence_id,
        "frame_count": len(extracted_paths),
        "duration_seconds": metadata.get("duration_seconds"),
    }
