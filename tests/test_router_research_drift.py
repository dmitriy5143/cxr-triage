from pathlib import Path

import pandas as pd

from fluoro_mvp_backend.router import load_router_config, route_dataframe, summarize_routes


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "model_bundle"


def test_router_reproduces_research_final_auto_negative_mask():
    config = load_router_config(BUNDLE)
    scores = pd.read_csv(BUNDLE / "reports" / "input_scores_final_test.csv")
    research_routes = pd.read_csv(BUNDLE / "reports" / "selected_routes_final_test.csv")

    routed = route_dataframe(scores, config)
    backend_auto = routed["route"].eq("no_attention_required")
    research_auto = research_routes["route"].eq("no_attention_required")

    assert backend_auto.sum() == 125
    assert backend_auto.equals(research_auto)


def test_router_reproduces_research_final_metrics():
    config = load_router_config(BUNDLE)
    scores = pd.read_csv(BUNDLE / "reports" / "input_scores_final_test.csv")
    routed = route_dataframe(scores, config)
    summary = summarize_routes(routed)

    expected = config["selected_fixed_final_metrics"]
    assert summary["n"] == 1256
    assert summary["selected_count"] == int(expected["selected_count"])
    assert summary["FN_count"] == 0
    assert summary["NPV"] == 1.0
    assert abs(summary["auto_negative_coverage"] - expected["auto_negative_coverage"]) < 1e-12
    assert abs(summary["NPV_ci95_low"] - expected["NPV_ci95_low"]) < 1e-4
