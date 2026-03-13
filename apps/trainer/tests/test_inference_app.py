from __future__ import annotations
from pathlib import Path
import math
import sys
import types

from PIL import Image
import pytest

import pixel_sheriff_trainer.inference.app as inference_app_module
import pixel_sheriff_trainer.inference.session_cache as session_cache_module
from pixel_sheriff_trainer.inference.schemas import InferDetectionWarmupRequest


@pytest.mark.asyncio
async def test_detection_warmup_endpoint_returns_selected_device(tmp_path: Path, monkeypatch) -> None:
    storage_root = tmp_path / "storage"
    onnx_path = storage_root / "models" / "demo.onnx"
    metadata_path = storage_root / "models" / "demo.metadata.json"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(b"fake-onnx")
    metadata_path.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))
    monkeypatch.setattr(inference_app_module, "sha256_file", lambda _path: "model-key")
    monkeypatch.setattr(
        session_cache_module,
        "ort",
        type(
            "_DummyOrt",
            (),
            {
                "get_available_providers": staticmethod(lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]),
                "InferenceSession": object,
            },
        ),
    )

    async def _acquire_session(self, *, model_key: str, onnx_path: Path, device_preference: str):
        assert model_key == "model-key"
        assert onnx_path.name == "demo.onnx"
        assert device_preference == "auto"
        return object(), "cuda"

    async def _release(self, model_key: str, device_selected: str) -> None:
        assert model_key == "model-key"
        assert device_selected == "cuda"

    monkeypatch.setattr(inference_app_module.SessionCache, "acquire_session", _acquire_session)
    monkeypatch.setattr(inference_app_module.SessionCache, "release", _release)

    app = inference_app_module.create_app()
    route = next(route for route in app.routes if getattr(route, "path", None) == "/infer/detection/warmup")
    payload = InferDetectionWarmupRequest(
        onnx_relpath="models/demo.onnx",
        metadata_relpath="models/demo.metadata.json",
        device_preference="auto",
        model_key="model-key",
    )

    response = await route.endpoint(payload)
    assert response.model_dump() == {"device_selected": "cuda", "warmed": True}


def test_parse_florence_generation_skips_malformed_entries() -> None:
    class _FakeProcessor:
        def post_process_generation(self, _generated_text: str, *, task: str, image_size: tuple[int, int]):
            assert task == "<OPEN_VOCABULARY_DETECTION>"
            assert image_size == (100, 80)
            return {
                task: {
                    "bboxes": [
                        [120.0, -10.0, 20.0, 70.0],
                        [10.0, 12.0, 10.0, 40.0],
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, float("nan"), 9.0],
                        [5.0, 6.0, 20.0, 30.0],
                    ],
                    "labels": ["Cat", "Dog", "Cat", "Dog", ""],
                    "scores": [0.91, 0.88, 0.95, math.inf, 0.77],
                }
            }

    detections = inference_app_module._parse_florence_generation(
        _FakeProcessor(),
        generated_text="ignored",
        task="<OPEN_VOCABULARY_DETECTION>",
        image_size=(100, 80),
        score_threshold=0.25,
        max_detections=10,
    )

    assert [item.model_dump() for item in detections] == [
        {
            "label_text": "Cat",
            "score": 0.91,
            "bbox": [120.0, -10.0, 20.0, 70.0],
        }
    ]


def test_run_florence_detection_disables_generation_cache(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 24), color=(12, 34, 56)).save(image_path)

    class _FakeInputs(dict):
        def to(self, _device: str):
            return self

    class _FakeProcessor:
        def __call__(self, *, text: str, images, return_tensors: str):
            assert text == "<OPEN_VOCABULARY_DETECTION> human"
            assert images.size == (32, 24)
            assert return_tensors == "pt"
            return _FakeInputs({"input_ids": [[1, 2, 3]], "pixel_values": [[0.0]]})

        def batch_decode(self, generated_ids, *, skip_special_tokens: bool):
            assert generated_ids == [[101, 102]]
            assert skip_special_tokens is False
            return ["ignored"]

        def post_process_generation(self, _generated_text: str, *, task: str, image_size: tuple[int, int]):
            assert task == "<OPEN_VOCABULARY_DETECTION>"
            assert image_size == (32, 24)
            return {
                task: {
                    "bboxes": [[1.0, 2.0, 10.0, 20.0]],
                    "labels": ["human"],
                    "scores": [0.9],
                }
            }

    class _FakeModel:
        def generate(self, **kwargs):
            assert kwargs["use_cache"] is False
            assert kwargs["num_beams"] == 3
            assert kwargs["do_sample"] is False
            return [[101, 102]]

    detections = inference_app_module._run_florence_detection(
        _FakeModel(),
        _FakeProcessor(),
        "cpu",
        image_path,
        ["human"],
        0.25,
        20,
    )

    assert [item.model_dump() for item in detections] == [
        {
            "label_text": "human",
            "score": 0.9,
            "bbox": [1.0, 2.0, 10.0, 20.0],
        }
    ]


def test_load_florence_runtime_retries_meta_tensor_failure(monkeypatch) -> None:
    inference_app_module._FLORENCE_CACHE.clear()

    calls: list[dict[str, object]] = []

    class _FakeModel:
        def __init__(self, *, fail_on_to: bool) -> None:
            self.fail_on_to = fail_on_to
            self.eval_called = False

        def to(self, _device: str):
            if self.fail_on_to:
                raise RuntimeError(
                    "Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead."
                )
            return self

        def eval(self) -> None:
            self.eval_called = True

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name: str, **kwargs):
            calls.append(dict(kwargs))
            return _FakeModel(fail_on_to=len(calls) == 1)

    class _FakeProcessor:
        pass

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(_model_name: str, **_kwargs):
            return _FakeProcessor()

    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=_FakeAutoModel,
        AutoProcessor=_FakeAutoProcessor,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(inference_app_module.torch.cuda, "is_available", lambda: False)

    model, processor, device = inference_app_module._load_florence_runtime("microsoft/Florence-2-base-ft")

    assert device == "cpu"
    assert isinstance(model, _FakeModel)
    assert isinstance(processor, _FakeProcessor)
    assert model.eval_called is True
    assert calls == [
        {"trust_remote_code": True, "attn_implementation": "eager"},
        {"trust_remote_code": True, "attn_implementation": "eager", "low_cpu_mem_usage": False},
    ]

    cached_model, cached_processor, cached_device = inference_app_module._load_florence_runtime("microsoft/Florence-2-base-ft")
    assert cached_model is model
    assert cached_processor is processor
    assert cached_device == device
    assert len(calls) == 2

    inference_app_module._FLORENCE_CACHE.clear()
