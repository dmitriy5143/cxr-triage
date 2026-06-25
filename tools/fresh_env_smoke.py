from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluoro_mvp_backend.inference import predict_from_scores  # noqa: E402
from fluoro_mvp_backend.router import load_router_config, route_dataframe, summarize_routes  # noqa: E402


def main() -> int:
    bundle = ROOT / "model_bundle"
    config = load_router_config(bundle)
    scores = pd.read_csv(bundle / "reports" / "input_scores_final_test.csv")
    routed = route_dataframe(scores, config)
    summary = summarize_routes(routed)

    demo = json.loads((ROOT / "examples" / "demo_scores_auto_negative.json").read_text(encoding="utf-8"))
    decision = predict_from_scores(demo, bundle)

    print("Fresh-environment smoke summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("Demo route:", decision["route"], decision["reason"])

    assert summary["selected_count"] == 125
    assert summary["FN_count"] == 0
    assert abs(summary["auto_negative_coverage"] - 0.09952229299363058) < 1e-12
    assert decision["route"] == "no_attention_required"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
