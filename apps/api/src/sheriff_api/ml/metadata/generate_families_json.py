from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sheriff_api.ml.registry import FAMILY_BACKBONES, FAMILY_INPUT_SIZE_RULES, FAMILY_TASK_MAP


def _default_output_path() -> Path:
    return Path(__file__).resolve().parents[6] / "packages" / "contracts" / "metadata" / "families.v1.json"


def build_families_payload() -> dict[str, Any]:
    families_payload: list[dict[str, Any]] = []
    for family_name in sorted(FAMILY_TASK_MAP.keys()):
        families_payload.append(
            {
                "name": family_name,
                "task": FAMILY_TASK_MAP[family_name],
                "allowed_backbones": FAMILY_BACKBONES[family_name],
                "input_size": FAMILY_INPUT_SIZE_RULES[family_name],
            }
        )
    return {"schema_version": "1", "families": families_payload}


def write_families_json(path: str | Path, payload: dict[str, Any] | None = None) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    body = payload if payload is not None else build_families_payload()
    target_path.write_text(json.dumps(body, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pixel Sheriff model families JSON")
    parser.add_argument("--out", default=str(_default_output_path()), help="Output path for the families JSON file")
    args = parser.parse_args()
    written = write_families_json(args.out)
    print(str(written))


if __name__ == "__main__":
    main()
