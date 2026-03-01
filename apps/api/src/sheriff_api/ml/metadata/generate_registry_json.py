from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sheriff_api.ml.metadata.backbones import BACKBONES


def _default_output_path() -> Path:
    return Path(__file__).resolve().parents[6] / "apps" / "web" / "src" / "lib" / "metadata" / "backbones.v1.json"


def build_registry_payload() -> dict[str, Any]:
    backbones_payload: list[dict[str, Any]] = []
    for backbone_name in sorted(BACKBONES.keys()):
        backbone = BACKBONES[backbone_name]
        taps_payload = [
            {
                "name": tap_name,
                "kind": tap_meta.kind,
                "channels": tap_meta.channels,
                "stride": tap_meta.stride,
            }
            for tap_name, tap_meta in sorted(backbone.taps.items(), key=lambda row: row[0])
        ]
        backbones_payload.append(
            {
                "name": backbone.name,
                "family": backbone.family,
                "embedding_dim": backbone.embedding_dim,
                "default_out_strides": list(backbone.default_out_strides),
                "taps": taps_payload,
            }
        )
    return {"schema_version": "1", "backbones": backbones_payload}


def write_registry_json(path: str | Path, payload: dict[str, Any] | None = None) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    body = payload if payload is not None else build_registry_payload()
    target_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pixel Sheriff backbone registry JSON")
    parser.add_argument("--out", default=str(_default_output_path()), help="Output path for the registry JSON file")
    args = parser.parse_args()
    written = write_registry_json(args.out)
    print(str(written))


if __name__ == "__main__":
    main()
