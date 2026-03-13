import json

from sheriff_api.ml.metadata.generate_registry_json import build_registry_payload, write_registry_json


def test_registry_json_generation_snapshotish(tmp_path) -> None:  # type: ignore[no-untyped-def]
    payload = build_registry_payload()
    assert payload["schema_version"] == "1"
    assert isinstance(payload["backbones"], list)

    resnet50 = next(row for row in payload["backbones"] if row["name"] == "resnet50")
    taps = {tap["name"]: tap for tap in resnet50["taps"]}
    assert "backbone.global_pool" in taps
    assert taps["backbone.c4"]["channels"] == 1024
    assert taps["backbone.c4"]["stride"] == 16

    efficientnet_v2_s = next(row for row in payload["backbones"] if row["name"] == "efficientnet_v2_s")
    efficientnet_taps = {tap["name"]: tap for tap in efficientnet_v2_s["taps"]}
    assert efficientnet_v2_s["family"] == "efficientnet_v2"
    assert efficientnet_v2_s["embedding_dim"] == 1280
    assert efficientnet_taps["backbone.c5"]["stride"] == 32
    assert efficientnet_taps["backbone.global_pool"]["channels"] == 1280

    output_path = tmp_path / "backbones.v1.json"
    write_registry_json(output_path, payload)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["schema_version"] == "1"
    assert any(backbone["name"] == "resnet18" for backbone in written["backbones"])
