from sheriff_api.ml.metadata.verify import verify_backbone_meta


def test_resnet18_meta_matches_model() -> None:
    issues = verify_backbone_meta("resnet18")
    assert issues == []


def test_resnet50_meta_matches_model() -> None:
    issues = verify_backbone_meta("resnet50")
    assert issues == []


def test_efficientnet_v2_s_meta_matches_model() -> None:
    issues = verify_backbone_meta("efficientnet_v2_s")
    assert issues == []
