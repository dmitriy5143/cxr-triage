from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .inference import ImageModelScoreProvider, PrecomputedScoreProvider, predict_from_scores


def load_scores_arg(value: str) -> dict[str, Any]:
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FLG MVP router on model scores.")
    parser.add_argument("--bundle", default="model_bundle", help="Path to backend model bundle.")
    parser.add_argument("--scores-json", help="JSON file or inline JSON with model scores.")
    parser.add_argument("--scores-csv", help="CSV with precomputed scores.")
    parser.add_argument("--study-id", help="Study id to read from --scores-csv.")
    parser.add_argument("--image", help="Image path for full EVA/CheXFound image inference.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.image:
        scored = ImageModelScoreProvider(args.bundle).score_image_with_metadata(args.image)
        decision = predict_from_scores(scored["scores"], args.bundle)
        decision["scores"] = scored["scores"]
        decision["preprocessing"] = scored["preprocessing"]
        text = json.dumps(decision, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True)
        print(text)
        return 0
    if args.scores_json:
        scores = load_scores_arg(args.scores_json)
    elif args.scores_csv and args.study_id:
        scores = PrecomputedScoreProvider.from_csv(args.scores_csv).get(args.study_id)
    else:
        raise SystemExit("Pass --image, --scores-json, or --scores-csv with --study-id.")

    decision = predict_from_scores(scores, args.bundle)
    text = json.dumps(decision, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
