import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_research_files_match_transfer_manifest():
    manifest = json.loads((ROOT / "research" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["created_for"] == "research_transfer_integrity"
    assert len(manifest["files"]) >= 20
    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert path.exists(), item["path"]
        assert path.stat().st_size == item["bytes"], item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["path"]
