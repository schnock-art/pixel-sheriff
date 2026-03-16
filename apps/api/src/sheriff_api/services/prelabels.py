from __future__ import annotations

import sheriff_api.services.prelabel_sources as prelabel_sources
from sheriff_api.services.prelabel_adapters import DetectionResult, PRELABEL_ADAPTER_REGISTRY, PrelabelAdapter
from sheriff_api.services.prelabel_common import (
    close_prelabel_session_input,
    deployment_store,
    get_latest_sequence_prelabel_session,
    inference_client,
    list_sequence_prelabel_sessions,
    list_task_categories,
    normalize_prompts,
    pending_prelabel_counts_for_assets,
    prelabel_proposal_to_read,
    prelabel_session_to_read,
    sequence_pending_total_from_counts,
    utc_now_dt,
)
from sheriff_api.services.prelabel_jobs import (
    enqueue_existing_sequence_assets_for_session,
    enqueue_live_prelabel_jobs_for_asset,
    mark_prelabel_session_failed,
    process_prelabel_asset_job,
)
from sheriff_api.services.prelabel_matching import (
    append_debug_detection as _append_debug_detection,
    bbox_xyxy_to_xywh as _bbox_xyxy_to_xywh,
    category_alias_keys as _category_alias_keys,
    category_exact_keys as _category_exact_keys,
    category_match_maps as _category_match_maps,
    inflection_alias_keys as _inflection_alias_keys,
    match_detection_category as _match_detection_category,
    normalize_prelabel_label_key as _normalize_prelabel_label_key,
    normalize_xywh_bbox as _normalize_xywh_bbox,
    normalized_debug_detections as _normalized_debug_detections,
)
from sheriff_api.services.prelabel_queue import PrelabelQueue
from sheriff_api.services.prelabel_reviews import (
    accept_prelabel_proposals,
    list_session_proposals,
    reject_prelabel_proposals,
    sync_annotation_prelabel_proposals,
)
from sheriff_api.services.prelabel_sources import create_prelabel_session, resolve_active_deployment, resolve_prelabel_source_config, warmup_prelabel_source


asyncio = prelabel_sources.asyncio
