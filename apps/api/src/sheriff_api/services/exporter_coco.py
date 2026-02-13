from sheriff_api.services.hashing import stable_hash


def build_manifest(project_id: str, categories: list[dict], assets: list[dict], annotations: list[dict]) -> dict:
    return {
        "project_id": project_id,
        "schema_version": "1.0.0",
        "categories": categories,
        "assets": assets,
        "annotations": annotations,
    }


def build_export_result(project_id: str, categories: list[dict], assets: list[dict], annotations: list[dict]) -> tuple[dict, str]:
    manifest = build_manifest(project_id, categories, assets, annotations)
    return manifest, stable_hash(manifest)
