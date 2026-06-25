import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cli(scores_file: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        sys.executable,
        "-m",
        "fluoro_mvp_backend.cli",
        "--bundle",
        str(ROOT / "model_bundle"),
        "--scores-json",
        str(scores_file),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    return json.loads(completed.stdout)


def test_cli_auto_negative_demo():
    out = run_cli(ROOT / "examples" / "demo_scores_auto_negative.json")
    assert out["route"] == "no_attention_required"
    assert out["selected_by_rule"] is True


def test_cli_manual_review_demo():
    out = run_cli(ROOT / "examples" / "demo_scores_manual_review.json")
    assert out["route"] in {"N/A", "requires_attention"}
    assert out["selected_by_rule"] is False
