from __future__ import annotations

from sheriff_api.db.models import Category
from sheriff_api.services.prelabel_matching import bbox_xyxy_to_xywh, category_match_maps, match_detection_category


def _category(*, category_id: str, name: str) -> Category:
    return Category(id=category_id, project_id="project-1", task_id="task-1", name=name, display_order=0, is_active=True)


def test_category_match_maps_only_promote_unambiguous_aliases() -> None:
    categories = [
        _category(category_id="cat-person", name="Human"),
        _category(category_id="cat-glass", name="Glass"),
        _category(category_id="cat-eyewear", name="Eyewear"),
    ]

    exact_mapping, alias_mapping = category_match_maps(categories)

    assert match_detection_category(label_text="person", exact_mapping=exact_mapping, alias_mapping=alias_mapping) == categories[0]
    assert match_detection_category(label_text="people", exact_mapping=exact_mapping, alias_mapping=alias_mapping) == categories[0]
    assert match_detection_category(label_text="glasses", exact_mapping=exact_mapping, alias_mapping=alias_mapping) is None


def test_bbox_xyxy_to_xywh_clamps_to_asset_bounds() -> None:
    assert bbox_xyxy_to_xywh((110.0, -5.0, 20.0, 90.0), width=100, height=80) == [20.0, 0.0, 80.0, 80.0]
    assert bbox_xyxy_to_xywh((12.0, 12.0, 12.0, 40.0), width=100, height=80) is None
