from __future__ import annotations

import json
from pathlib import Path

from sheriff_api.ml.metadata.generate_families_json import build_families_payload
from sheriff_api.ml.metadata.generate_registry_json import build_registry_payload


REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_generated_metadata_matches_canonical_contract_files() -> None:
    assert build_registry_payload() == _load_json(REPO_ROOT / "packages" / "contracts" / "metadata" / "backbones.v1.json")
    assert build_families_payload() == _load_json(REPO_ROOT / "packages" / "contracts" / "metadata" / "families.v1.json")


def test_app_local_contract_copies_match_canonical_contract_files() -> None:
    pairs = [
        (
            REPO_ROOT / "packages" / "contracts" / "schemas" / "dataset_version_v2.schema.json",
            REPO_ROOT / "apps" / "api" / "src" / "sheriff_api" / "schemas_json" / "dataset_version_v2.schema.json",
        ),
        (
            REPO_ROOT / "packages" / "contracts" / "schemas" / "dataset_version_v2.schema.json",
            REPO_ROOT / "apps" / "web" / "src" / "lib" / "schemas" / "dataset_version_v2.schema.json",
        ),
        (
            REPO_ROOT / "packages" / "contracts" / "schemas" / "model-config-1.0.schema.json",
            REPO_ROOT / "apps" / "api" / "src" / "sheriff_api" / "schemas" / "model_config_schema.json",
        ),
        (
            REPO_ROOT / "packages" / "contracts" / "schemas" / "model-config-1.0.schema.json",
            REPO_ROOT / "apps" / "web" / "src" / "schemas" / "model-config-1.0.schema.json",
        ),
        (
            REPO_ROOT / "packages" / "contracts" / "metadata" / "backbones.v1.json",
            REPO_ROOT / "apps" / "web" / "src" / "lib" / "metadata" / "backbones.v1.json",
        ),
        (
            REPO_ROOT / "packages" / "contracts" / "metadata" / "families.v1.json",
            REPO_ROOT / "apps" / "web" / "src" / "lib" / "metadata" / "families.v1.json",
        ),
    ]

    for canonical, target in pairs:
        assert target.read_text(encoding="utf-8") == canonical.read_text(encoding="utf-8")
