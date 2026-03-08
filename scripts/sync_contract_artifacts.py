from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_SRC = REPO_ROOT / "apps" / "api" / "src"
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

from sheriff_api.ml.metadata.generate_families_json import build_families_payload  # noqa: E402
from sheriff_api.ml.metadata.generate_registry_json import build_registry_payload  # noqa: E402


CONTRACT_TARGETS: dict[Path, list[Path]] = {
    REPO_ROOT / "packages" / "contracts" / "schemas" / "dataset_version_v2.schema.json": [
        REPO_ROOT / "apps" / "api" / "src" / "sheriff_api" / "schemas_json" / "dataset_version_v2.schema.json",
        REPO_ROOT / "apps" / "web" / "src" / "lib" / "schemas" / "dataset_version_v2.schema.json",
    ],
    REPO_ROOT / "packages" / "contracts" / "schemas" / "model-config-1.0.schema.json": [
        REPO_ROOT / "apps" / "api" / "src" / "sheriff_api" / "schemas" / "model_config_schema.json",
        REPO_ROOT / "apps" / "web" / "src" / "schemas" / "model-config-1.0.schema.json",
    ],
    REPO_ROOT / "packages" / "contracts" / "metadata" / "backbones.v1.json": [
        REPO_ROOT / "apps" / "web" / "src" / "lib" / "metadata" / "backbones.v1.json",
    ],
    REPO_ROOT / "packages" / "contracts" / "metadata" / "families.v1.json": [
        REPO_ROOT / "apps" / "web" / "src" / "lib" / "metadata" / "families.v1.json",
    ],
}


def _render_json(payload: dict, *, sort_keys: bool) -> str:
    return json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n"


def _generated_contracts() -> dict[Path, str]:
    return {
        REPO_ROOT / "packages" / "contracts" / "metadata" / "backbones.v1.json": _render_json(
            build_registry_payload(),
            sort_keys=True,
        ),
        REPO_ROOT / "packages" / "contracts" / "metadata" / "families.v1.json": _render_json(
            build_families_payload(),
            sort_keys=False,
        ),
    }


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def sync_contracts() -> int:
    generated = _generated_contracts()
    for path, body in generated.items():
        _write_text(path, body)
        print(f"generated {path.relative_to(REPO_ROOT)}")

    for canonical, targets in CONTRACT_TARGETS.items():
        body = _read_text(canonical)
        if body is None:
            raise FileNotFoundError(f"Canonical contract is missing: {canonical}")
        for target in targets:
            _write_text(target, body)
            print(f"synced {target.relative_to(REPO_ROOT)}")
    return 0


def check_contracts() -> int:
    generated = _generated_contracts()
    failures: list[str] = []

    for path, expected in generated.items():
        actual = _read_text(path)
        if actual != expected:
            failures.append(f"generated metadata drift: {path.relative_to(REPO_ROOT)}")

    for canonical, targets in CONTRACT_TARGETS.items():
        canonical_body = _read_text(canonical)
        if canonical_body is None:
            failures.append(f"missing canonical contract: {canonical.relative_to(REPO_ROOT)}")
            continue
        for target in targets:
            actual = _read_text(target)
            if actual != canonical_body:
                failures.append(
                    f"out-of-sync target: {target.relative_to(REPO_ROOT)} != {canonical.relative_to(REPO_ROOT)}"
                )

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print("contract artifacts are in sync")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and synchronize shared contract artifacts")
    parser.add_argument("--check", action="store_true", help="Verify contracts are in sync without writing files")
    args = parser.parse_args()
    if args.check:
        return check_contracts()
    return sync_contracts()


if __name__ == "__main__":
    raise SystemExit(main())
