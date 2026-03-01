from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sheriff_api.ml.metadata.backbones import get_tap_meta, normalize_tap_name
from sheriff_api.ml.metadata.verify import verify_backbone_meta
from sheriff_api.ml.outputs.composer import AuxOutputSpec, OutputComposer
from sheriff_api.ml.outputs.projections import normalize_projection_spec
from sheriff_api.ml.registry import build_family_adapter
from sheriff_api.ml.utils.schema_validation import validate_model_config_schema

if TYPE_CHECKING:
    from torch import nn


class ModelFactoryValidationError(ValueError):
    pass


@dataclass(frozen=True)
class BuiltModel:
    model: "nn.Module"
    output_names: list[str]
    label_map: dict[str, Any]


def _as_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelFactoryValidationError(f"{field_name} must be an object")
    return value


def _as_list(value: Any, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ModelFactoryValidationError(f"{field_name} must be an array")
    return value


def _as_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int):
        raise ModelFactoryValidationError(f"{field_name} must be an integer")
    return value


def _as_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelFactoryValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _collect_aux_specs(model_config: dict[str, Any], *, backbone_name: str, adapter_supported_taps: list[str]) -> list[AuxOutputSpec]:
    outputs = _as_dict(model_config.get("outputs"), field_name="outputs")
    aux_rows = _as_list(outputs.get("aux"), field_name="outputs.aux")
    aux_specs: list[AuxOutputSpec] = []
    seen_names: set[str] = set()

    for index, row in enumerate(aux_rows):
        aux = _as_dict(row, field_name=f"outputs.aux[{index}]")
        aux_name = _as_str(aux.get("name"), field_name=f"outputs.aux[{index}].name")
        if aux_name in seen_names:
            raise ModelFactoryValidationError(f"Duplicate aux output name: {aux_name}")
        seen_names.add(aux_name)

        source = _as_dict(aux.get("source"), field_name=f"outputs.aux[{index}].source")
        block = _as_str(source.get("block"), field_name=f"outputs.aux[{index}].source.block")
        tap = _as_str(source.get("tap"), field_name=f"outputs.aux[{index}].source.tap")
        canonical_tap = normalize_tap_name(f"{block}.{tap}")

        try:
            tap_meta = get_tap_meta(backbone_name, canonical_tap)
        except KeyError as exc:
            raise ModelFactoryValidationError(str(exc)) from exc
        if canonical_tap not in adapter_supported_taps:
            raise ModelFactoryValidationError(
                f"Tap '{canonical_tap}' is not supported by adapter family for backbone '{backbone_name}'"
            )

        projection = _as_dict(aux.get("projection"), field_name=f"outputs.aux[{index}].projection")
        normalized_projection = normalize_projection_spec(projection)
        projection_type = _as_str(normalized_projection.get("type"), field_name=f"outputs.aux[{index}].projection.type").lower()
        if projection_type not in {"none", "pool_linear", "mlp"}:
            raise ModelFactoryValidationError(
                f"Projection type '{projection_type}' is not supported in v0 (expected one of: none, pool_linear, mlp, linear)"
            )

        aux_type = _as_str(aux.get("type"), field_name=f"outputs.aux[{index}].type").lower()
        if aux_type == "embedding" and projection_type == "none" and tap_meta.kind != "embedding":
            raise ModelFactoryValidationError(
                f"Aux output '{aux_name}' has type=embedding but source tap '{canonical_tap}' is feature_map. "
                "Use pool_linear/mlp projection or choose an embedding tap."
            )
        if aux_type == "feature_map" and (tap_meta.kind != "feature_map" or projection_type != "none"):
            raise ModelFactoryValidationError(
                f"Aux output '{aux_name}' has type=feature_map but requires feature_map tap with projection.type=none"
            )

        normalize_mode = str(normalized_projection.get("normalize", "none")).lower()
        if normalize_mode not in {"none", "l2"}:
            raise ModelFactoryValidationError(f"Unsupported normalize mode '{normalize_mode}' for aux output '{aux_name}'")

        aux_specs.append(
            AuxOutputSpec(
                name=aux_name,
                tap_name=canonical_tap,
                projection_spec=normalized_projection,
                normalize=normalize_mode,
            )
        )

    return aux_specs


def _validate_outputs_declared(model_config: dict[str, Any], *, aux_specs: list[AuxOutputSpec]) -> tuple[str, list[str]]:
    outputs = _as_dict(model_config.get("outputs"), field_name="outputs")
    export = _as_dict(model_config.get("export"), field_name="export")
    onnx = _as_dict(export.get("onnx"), field_name="export.onnx")
    output_names_raw = _as_list(onnx.get("output_names"), field_name="export.onnx.output_names")
    output_names = [_as_str(name, field_name="export.onnx.output_names[]") for name in output_names_raw]

    primary = _as_dict(outputs.get("primary"), field_name="outputs.primary")
    primary_name = _as_str(primary.get("name"), field_name="outputs.primary.name")
    if primary_name not in output_names:
        raise ModelFactoryValidationError(
            f"outputs.primary.name '{primary_name}' must be included in export.onnx.output_names"
        )

    for aux in aux_specs:
        if aux.name not in output_names:
            raise ModelFactoryValidationError(
                f"Aux output '{aux.name}' must be included in export.onnx.output_names"
            )
    return primary_name, output_names


def _validate_head_consistency(model_config: dict[str, Any]) -> None:
    source_dataset = _as_dict(model_config.get("source_dataset"), field_name="source_dataset")
    architecture = _as_dict(model_config.get("architecture"), field_name="architecture")
    head = _as_dict(architecture.get("head"), field_name="architecture.head")

    expected_num_classes = _as_int(source_dataset.get("num_classes"), field_name="source_dataset.num_classes")
    head_num_classes = _as_int(head.get("num_classes"), field_name="architecture.head.num_classes")
    if expected_num_classes != head_num_classes:
        raise ModelFactoryValidationError(
            "architecture.head.num_classes must match source_dataset.num_classes "
            f"(expected {expected_num_classes}, got {head_num_classes})"
        )


def _build_label_map(model_config: dict[str, Any]) -> dict[str, Any]:
    source_dataset = _as_dict(model_config.get("source_dataset"), field_name="source_dataset")
    class_order = _as_list(source_dataset.get("class_order"), field_name="source_dataset.class_order")
    class_names = source_dataset.get("class_names")
    if not isinstance(class_names, list):
        class_names = []
    task = _as_str(source_dataset.get("task"), field_name="source_dataset.task")
    return {
        "index_to_class_id": class_order,
        "class_names": class_names,
        "task": task,
    }


def _run_metadata_verification(backbone_name: str) -> None:
    issues = verify_backbone_meta(backbone_name)
    if not issues:
        return
    rendered = "; ".join(str(issue.get("message", issue)) for issue in issues)
    raise ModelFactoryValidationError(f"Backbone metadata verification failed for '{backbone_name}': {rendered}")


def build_model(model_config: dict[str, Any], verify_metadata: bool = False) -> BuiltModel:
    try:
        validate_model_config_schema(model_config)
    except ValueError as exc:
        raise ModelFactoryValidationError(f"Model config schema validation failed: {exc}") from exc

    _validate_head_consistency(model_config)
    adapter = build_family_adapter(model_config)
    aux_specs = _collect_aux_specs(
        model_config,
        backbone_name=adapter.backbone_name,
        adapter_supported_taps=adapter.supported_taps,
    )
    primary_output_name, output_names = _validate_outputs_declared(model_config, aux_specs=aux_specs)

    if verify_metadata:
        _run_metadata_verification(adapter.backbone_name)

    composed = OutputComposer(
        adapter=adapter,
        primary_output_name=primary_output_name,
        output_names=output_names,
        aux_specs=aux_specs,
    )
    return BuiltModel(
        model=composed,
        output_names=output_names,
        label_map=_build_label_map(model_config),
    )
