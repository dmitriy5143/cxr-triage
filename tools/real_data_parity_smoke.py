from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluoro_mvp_backend.inference import ImageModelScoreProvider, predict_from_scores  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-data parity smoke against locked research final-test scores.")
    parser.add_argument("--bundle", default=str(ROOT / "model_bundle"), help="Path to model bundle.")
    parser.add_argument(
        "--data-root",
        default=str(ROOT.parent / "data" / "incxr_png" / "IN-CXR (pre-processed)"),
        help="Root with Normal/ and AbNormal/ IN-CXR PNG folders.",
    )
    parser.add_argument("--auto-n", type=int, default=10, help="Number of locked auto-negative final-test cases.")
    parser.add_argument("--nonauto-n", type=int, default=10, help="Number of locked non-auto final-test cases.")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "reports" / "real_data_parity"),
        help="Where to write CSV and summary JSON.",
    )
    parser.add_argument("--min-route-match", type=float, default=1.0, help="Minimum expected auto-decision match rate.")
    return parser


def image_path_for(data_root: Path, image_file: str) -> Path:
    subdir = "Normal" if image_file.endswith("_Normal.png") else "AbNormal"
    return data_root / subdir / image_file


def main() -> int:
    args = build_parser().parse_args()
    bundle = Path(args.bundle)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    routes = pd.read_csv(bundle / "reports" / "selected_routes_final_test.csv")
    auto = routes[routes["route"].eq("no_attention_required")].head(args.auto_n)
    nonauto = routes[routes["reason"].eq("not_auto_negative")].head(args.nonauto_n)
    sample = pd.concat([auto, nonauto], ignore_index=True)
    if sample.empty:
        raise RuntimeError("No final-test rows were selected for parity smoke.")

    provider = ImageModelScoreProvider(bundle)
    rows: list[dict[str, object]] = []
    for _, row in sample.iterrows():
        image_file = str(row["image_file"])
        image_path = image_path_for(data_root, image_file)
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        scored = provider.score_image_with_metadata(image_path)["scores"]
        decision = predict_from_scores(scored, bundle)
        expected_auto = str(row.get("route")) == "no_attention_required"
        actual_auto = decision["route"] == "no_attention_required"
        rows.append(
            {
                "study_id": row["study_id"],
                "image_file": image_file,
                "y_attention": int(row["y_attention"]),
                "expected_auto_negative": expected_auto,
                "actual_auto_negative": actual_auto,
                "actual_route": decision["route"],
                "actual_reason": decision["reason"],
                "research_p_chex_head": float(row["p_chex_head"]),
                "backend_p_chex_head": float(scored["p_chex_head"]),
                "absdiff_p_chex_head": abs(float(row["p_chex_head"]) - float(scored["p_chex_head"])),
                "research_p_last1": float(row["p_last1"]),
                "backend_p_last1": float(scored["p_last1"]),
                "absdiff_p_last1": abs(float(row["p_last1"]) - float(scored["p_last1"])),
                "research_ood_chex": float(row["ood_score_chex"]),
                "backend_ood_chex": float(scored["ood_score_chex"]),
                "absdiff_ood_chex": abs(float(row["ood_score_chex"]) - float(scored["ood_score_chex"])),
                "research_ood_eva": float(row["ood_score_eva"]),
                "backend_ood_eva": float(scored["ood_score_eva"]),
                "absdiff_ood_eva": abs(float(row["ood_score_eva"]) - float(scored["ood_score_eva"])),
                "quality_score": float(scored["quality_score"]),
            }
        )

    report = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "real_data_parity_sample.csv", index=False)
    summary = {
        "n": int(len(report)),
        "auto_decision_matches": int((report["expected_auto_negative"] == report["actual_auto_negative"]).sum()),
        "auto_decision_match_rate": float((report["expected_auto_negative"] == report["actual_auto_negative"]).mean()),
        "max_absdiff_p_chex_head": float(report["absdiff_p_chex_head"].max()),
        "median_absdiff_p_chex_head": float(report["absdiff_p_chex_head"].median()),
        "max_absdiff_p_last1": float(report["absdiff_p_last1"].max()),
        "median_absdiff_p_last1": float(report["absdiff_p_last1"].median()),
        "max_absdiff_ood_chex": float(report["absdiff_ood_chex"].max()),
        "median_absdiff_ood_chex": float(report["absdiff_ood_chex"].median()),
        "max_absdiff_ood_eva": float(report["absdiff_ood_eva"].max()),
        "median_absdiff_ood_eva": float(report["absdiff_ood_eva"].median()),
    }
    (out_dir / "real_data_parity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["auto_decision_match_rate"] < args.min_route_match:
        raise AssertionError(f"Auto-decision match rate below target: {summary['auto_decision_match_rate']}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
