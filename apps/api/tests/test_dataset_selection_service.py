from __future__ import annotations

from types import SimpleNamespace

from sheriff_api.services.dataset_selection import (
    AssetRow,
    apply_selection_filters,
    build_split_plan,
    split_counts,
    to_selection_payload,
    validate_split_ratios,
)


def _row(
    asset_id: str,
    *,
    status: str = "unlabeled",
    category_ids: list[str] | None = None,
    primary_category_id: str | None = None,
    has_objects: bool = False,
    relative_path: str = "",
    filename: str = "",
) -> AssetRow:
    return AssetRow(
        asset=SimpleNamespace(id=asset_id),
        annotation=None,
        status=status,
        category_ids=category_ids or [],
        primary_category_id=primary_category_id,
        has_objects=has_objects,
        relative_path=relative_path or f"{asset_id}.jpg",
        filename=filename or f"{asset_id}.jpg",
    )


def test_apply_selection_filters_respects_exclude_precedence_for_status_and_folder() -> None:
    rows = [
        _row("a", status="labeled", relative_path="animals/cats/a.jpg"),
        _row("b", status="approved", relative_path="animals/cats/b.jpg"),
        _row("c", status="needs_review", relative_path="animals/dogs/c.jpg"),
        _row("d", status="skipped", relative_path="misc/d.jpg"),
    ]

    selected = apply_selection_filters(
        rows,
        mode="filter_snapshot",
        explicit_asset_ids=[],
        filters={
            "include_statuses": ["labeled", "approved", "needs_review"],
            "exclude_statuses": ["approved"],
            "include_folder_paths": ["animals"],
            "exclude_folder_paths": ["animals/cats"],
        },
        task_kind="classification",
    )

    assert [row.asset.id for row in selected] == ["c"]


def test_apply_selection_filters_respects_negative_image_flag_for_geometry_tasks() -> None:
    rows = [
        _row("positive", has_objects=True, category_ids=["boat"], primary_category_id="boat"),
        _row("negative", has_objects=False, category_ids=[]),
    ]

    selected = apply_selection_filters(
        rows,
        mode="filter_snapshot",
        explicit_asset_ids=[],
        filters={"include_negative_images": False},
        task_kind="bbox",
    )

    assert [row.asset.id for row in selected] == ["positive"]


def test_build_split_plan_falls_back_with_warning_when_stratify_is_impossible() -> None:
    rows = [
        _row("a", primary_category_id="rare"),
        _row("b", primary_category_id="common"),
    ]

    split_by_asset, warnings = build_split_plan(
        rows,
        task_kind="classification",
        seed=7,
        ratios=(0.5, 0.5, 0.0),
        stratify_enabled=True,
        strict_stratify=False,
    )

    assert set(split_by_asset) == {"a", "b"}
    assert len(warnings) == 1
    assert "Stratified split is impossible" in warnings[0]


def test_build_split_plan_raises_when_stratify_is_impossible_and_strict() -> None:
    rows = [
        _row("a", primary_category_id="rare"),
        _row("b", primary_category_id="common"),
    ]

    try:
        build_split_plan(
            rows,
            task_kind="classification",
            seed=7,
            ratios=(0.5, 0.5, 0.0),
            stratify_enabled=True,
            strict_stratify=True,
        )
    except RuntimeError as exc:
        assert str(exc) == "dataset_stratify_impossible"
    else:
        raise AssertionError("expected strict stratify to raise")


def test_validate_split_ratios_and_split_counts() -> None:
    ratios = validate_split_ratios({"train": 0.6, "val": 0.2, "test": 0.2})
    assert ratios == (0.6, 0.2, 0.2)
    assert split_counts({"a": "train", "b": "val", "c": "test", "d": "train"}) == {
        "train": 2,
        "val": 1,
        "test": 1,
    }


def test_to_selection_payload_deduplicates_explicit_asset_ids() -> None:
    payload = to_selection_payload("explicit_asset_ids", {}, ["a", "b", "a"])
    assert payload == {"mode": "explicit_asset_ids", "explicit": {"asset_ids": ["a", "b"]}}
