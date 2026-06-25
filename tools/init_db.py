from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluoro_mvp_backend.storage import FeedbackStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize the MVP SQLite database.")
    parser.add_argument(
        "--db",
        default=str(ROOT / "runtime" / "fluoro_mvp.sqlite"),
        help="SQLite database path.",
    )
    args = parser.parse_args()
    store = FeedbackStore(args.db)
    print(json.dumps({"db_path": str(store.db_path), "counts": store.counts()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
