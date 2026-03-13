from __future__ import annotations

import asyncio
import json
import struct
import sys
import zlib

from sqlalchemy import select

from sheriff_api.config import get_settings
from sheriff_api.db.models import Asset, AssetSequence, AssetType, Category, PrelabelProposal, PrelabelSession
from sheriff_api.db.session import SessionLocal
from sheriff_api.services.asset_ingest import build_asset_record
from sheriff_api.services.folders import ensure_folder_path
from sheriff_api.services.prelabels import utc_now_dt
from sheriff_api.services.sequences import refresh_sequence_counts
from sheriff_api.services.storage import LocalStorage


DEMO_FOLDER_PATH = "prelabels/review-sequence"
DEMO_SEQUENCE_ID = "demo-prelabel-sequence-v1"
DEMO_SESSION_ID = "demo-prelabel-session-v1"
DEMO_SEQUENCE_NAME = "README AI Review Sequence"

FRAME_SPECS = [
    {
        "id": "demo-prelabel-frame-1",
        "file_name": "frame_000001.png",
        "rgba": (44, 116, 184, 255),
        "timestamp_seconds": 0.0,
    },
    {
        "id": "demo-prelabel-frame-2",
        "file_name": "frame_000002.png",
        "rgba": (82, 138, 92, 255),
        "timestamp_seconds": 1.0,
    },
    {
        "id": "demo-prelabel-frame-3",
        "file_name": "frame_000003.png",
        "rgba": (156, 108, 72, 255),
        "timestamp_seconds": 2.0,
    },
]

PROPOSAL_SPECS = [
    {
        "id": "demo-prelabel-proposal-cat-1",
        "asset_id": "demo-prelabel-frame-1",
        "category_name": "Cat",
        "prompt_text": "cat",
        "confidence": 0.92,
        "bbox": [110.0, 90.0, 260.0, 190.0],
    },
    {
        "id": "demo-prelabel-proposal-dog-1",
        "asset_id": "demo-prelabel-frame-1",
        "category_name": "Dog",
        "prompt_text": "dog",
        "confidence": 0.88,
        "bbox": [430.0, 150.0, 190.0, 170.0],
    },
    {
        "id": "demo-prelabel-proposal-cat-2",
        "asset_id": "demo-prelabel-frame-2",
        "category_name": "Cat",
        "prompt_text": "cat",
        "confidence": 0.84,
        "bbox": [230.0, 130.0, 180.0, 240.0],
    },
]


def _png_bytes(width: int = 960, height: int = 540, *, rgba: tuple[int, int, int, int]) -> bytes:
    row = bytes([0] + list(rgba) * width)
    raw = row * height

    def chunk(chunk_type: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + chunk_type
            + payload
            + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )


async def seed_demo_prelabels(project_id: str, task_id: str) -> dict[str, object]:
    settings = get_settings()
    storage = LocalStorage(settings.storage_root)

    async with SessionLocal() as db:
        folder = await ensure_folder_path(db, project_id, DEMO_FOLDER_PATH)
        if folder is None:
            raise RuntimeError("Failed to create demo prelabel folder")

        category_rows = list(
            (
                await db.execute(
                    select(Category).where(Category.project_id == project_id, Category.task_id == task_id)
                )
            ).scalars().all()
        )
        category_by_name = {str(category.name or "").strip().lower(): category for category in category_rows}
        missing_categories = [name for name in {"cat", "dog"} if name not in category_by_name]
        if missing_categories:
            raise RuntimeError(f"Demo categories missing: {', '.join(sorted(missing_categories))}")

        existing_sequence = await db.get(AssetSequence, DEMO_SEQUENCE_ID)
        if existing_sequence is None:
            sequence = AssetSequence(
                id=DEMO_SEQUENCE_ID,
                project_id=project_id,
                task_id=task_id,
                folder_id=folder.id,
                name=DEMO_SEQUENCE_NAME,
                source_type="video_file",
                source_filename="demo-prelabels.mp4",
                status="ready",
                fps=1.0,
            )
            db.add(sequence)
            await db.flush()
        else:
            sequence = existing_sequence

        storage.ensure_project_dirs(project_id)
        created_assets: list[Asset] = []
        for frame_index, frame_spec in enumerate(FRAME_SPECS):
            asset = await db.get(Asset, frame_spec["id"])
            if asset is None:
                content = _png_bytes(rgba=frame_spec["rgba"])
                asset, storage_uri = build_asset_record(
                    project_id=project_id,
                    content=content,
                    file_name=frame_spec["file_name"],
                    mime_type="image/png",
                    folder=folder,
                    original_filename=frame_spec["file_name"],
                    asset_type=AssetType.frame,
                    sequence_id=sequence.id,
                    sequence_name=sequence.name,
                    source_kind="video_frame",
                    frame_index=frame_index,
                    timestamp_seconds=float(frame_spec["timestamp_seconds"]),
                    asset_id=frame_spec["id"],
                )
                storage.write_bytes(storage_uri, content)
                db.add(asset)
                await db.flush()
            created_assets.append(asset)

        await refresh_sequence_counts(db, sequence.id)
        sequence.width = 960
        sequence.height = 540
        sequence.duration_seconds = float(FRAME_SPECS[-1]["timestamp_seconds"])

        session = await db.get(PrelabelSession, DEMO_SESSION_ID)
        if session is None:
            session = PrelabelSession(
                id=DEMO_SESSION_ID,
                project_id=project_id,
                task_id=task_id,
                sequence_id=sequence.id,
                source_type="florence2",
                source_ref="microsoft/Florence-2-base-ft",
                prompts_json=["Cat", "Dog"],
                sampling_mode="every_n_frames",
                sampling_value=1.0,
                confidence_threshold=0.25,
                max_detections_per_frame=5,
                live_mode=False,
                status="completed",
                input_closed_at=utc_now_dt(),
                enqueued_assets=len(FRAME_SPECS),
                processed_assets=len(FRAME_SPECS),
                generated_proposals=len(PROPOSAL_SPECS),
                skipped_unmatched=0,
            )
            db.add(session)
            await db.flush()

        for proposal_spec in PROPOSAL_SPECS:
            proposal = await db.get(PrelabelProposal, proposal_spec["id"])
            if proposal is not None:
                continue
            category = category_by_name[proposal_spec["category_name"].lower()]
            db.add(
                PrelabelProposal(
                    id=proposal_spec["id"],
                    session_id=session.id,
                    asset_id=proposal_spec["asset_id"],
                    project_id=project_id,
                    task_id=task_id,
                    category_id=category.id,
                    label_text=category.name,
                    prompt_text=str(proposal_spec["prompt_text"]),
                    confidence=float(proposal_spec["confidence"]),
                    bbox_json=list(proposal_spec["bbox"]),
                    status="pending",
                )
            )

        await db.commit()

        return {
            "folderPath": DEMO_FOLDER_PATH,
            "sequenceId": sequence.id,
            "sequenceName": sequence.name,
            "sessionId": session.id,
            "frameAssetIds": [asset.id for asset in created_assets],
            "frameAssetPaths": [
                f"{DEMO_FOLDER_PATH}/{frame_spec['file_name']}"
                for frame_spec in FRAME_SPECS
            ],
            "proposalIds": [proposal_spec["id"] for proposal_spec in PROPOSAL_SPECS],
        }


async def _async_main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: python -m sheriff_api.demo_prelabel_seed <project_id> <task_id>", file=sys.stderr)
        return 1
    metadata = await seed_demo_prelabels(argv[1], argv[2])
    print(json.dumps(metadata, indent=2))
    return 0


def main() -> int:
    return asyncio.run(_async_main(sys.argv))


if __name__ == "__main__":
    raise SystemExit(main())
