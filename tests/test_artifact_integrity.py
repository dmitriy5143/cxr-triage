import hashlib
import json
import os
from pathlib import Path

import pytest

from fluoro_mvp_backend.inference import ImageModelScoreProvider


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_declares_required_delivery_roles():
    manifest = json.loads((ROOT / "model_bundle" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["primary_candidate"] == "ensemble_chexfound_head_plus_eva_x_base_last1_router"
    assert "eva_x_base_partial_unfreeze_last1" in manifest["single_model_candidates"]
    assert "chexfound_frozen_tuned_head_h512_do20_lr8e4_wd1e4" in manifest["single_model_candidates"]


def test_manifest_artifact_files_exist_and_small_checksums_match():
    manifest = json.loads((ROOT / "model_bundle" / "manifest.json").read_text(encoding="utf-8"))
    checks = manifest["artifact_checksums"]
    assert len(checks) >= 10
    check_large = os.environ.get("CHECK_LARGE_ARTIFACTS") == "1"
    binary_suffixes = {".pt", ".pth", ".pkl", ".safetensors", ".onnx"}
    for item in checks:
        path = ROOT / item["path"]
        is_binary_artifact = path.suffix in binary_suffixes
        if is_binary_artifact and not check_large:
            continue
        assert path.exists(), item["path"]
        assert path.stat().st_size == item["bytes"], item["path"]

        if item["bytes"] > 20_000_000 and not check_large:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == item["sha256"], item["path"]


def test_chexfound_safetensors_header_is_valid():
    provider = ImageModelScoreProvider(ROOT / "model_bundle")
    status = provider.artifact_status()
    assert status["chexfound_hf_config"]["exists"] is True
    assert status["chexfound_external_code"]["exists"] is True
    assert status["eva_x_external_code"]["exists"] is True
    if os.environ.get("CHECK_LARGE_ARTIFACTS") != "1":
        pytest.skip("Large artifacts are checked only with CHECK_LARGE_ARTIFACTS=1.")
    assert status["chexfound_hf_model_safetensors"]["exists"] is True
    assert status["eva_x_base_frozen_ood_weights"]["exists"] is True
    assert status["full_image_adapter_wired"] is True
    header = provider.validate_chexfound_safetensors_header()
    assert header["tensor_count"] > 100
    assert any(key.startswith("model.blocks.") for key in header["sample_keys"])
