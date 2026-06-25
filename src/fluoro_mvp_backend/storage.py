from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  study_id TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  route TEXT NOT NULL,
  reason TEXT NOT NULL,
  p_requires_attention REAL,
  payload_json TEXT NOT NULL,
  decision_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  prediction_id INTEGER,
  study_id TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  reviewer_id TEXT,
  label_attention INTEGER,
  comment TEXT,
  metadata_json TEXT,
  FOREIGN KEY(prediction_id) REFERENCES predictions(id)
);

CREATE TABLE IF NOT EXISTS training_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  status TEXT NOT NULL,
  run_name TEXT NOT NULL,
  config_json TEXT NOT NULL,
  metrics_json TEXT,
  artifact_uri TEXT
);
"""


class FeedbackStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def log_prediction(self, payload: dict[str, Any], decision: dict[str, Any]) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO predictions
                  (study_id, route, reason, p_requires_attention, payload_json, decision_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.get("study_id"),
                    decision["route"],
                    decision["reason"],
                    decision.get("p_requires_attention"),
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    json.dumps(decision, ensure_ascii=False, sort_keys=True),
                ),
            )
            return int(cur.lastrowid)

    def add_review_feedback(
        self,
        study_id: str,
        label_attention: int,
        prediction_id: int | None = None,
        reviewer_id: str | None = None,
        comment: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO review_feedback
                  (prediction_id, study_id, reviewer_id, label_attention, comment, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    study_id,
                    reviewer_id,
                    int(label_attention),
                    comment,
                    json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                ),
            )
            return int(cur.lastrowid)

    def create_training_run(self, run_name: str, config: dict[str, Any], status: str = "planned") -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO training_runs (status, run_name, config_json)
                VALUES (?, ?, ?)
                """,
                (status, run_name, json.dumps(config, ensure_ascii=False, sort_keys=True)),
            )
            return int(cur.lastrowid)

    def update_training_run(
        self,
        run_id: int,
        status: str,
        metrics: dict[str, Any] | None = None,
        artifact_uri: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE training_runs
                SET status = ?, metrics_json = ?, artifact_uri = ?
                WHERE id = ?
                """,
                (
                    status,
                    json.dumps(metrics or {}, ensure_ascii=False, sort_keys=True),
                    artifact_uri,
                    run_id,
                ),
            )

    def counts(self) -> dict[str, int]:
        with self.connect() as conn:
            return {
                "predictions": int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]),
                "review_feedback": int(conn.execute("SELECT COUNT(*) FROM review_feedback").fetchone()[0]),
                "training_runs": int(conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]),
            }

    def list_predictions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, study_id, created_at, route, reason, p_requires_attention
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_training_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, status, run_name, config_json, metrics_json, artifact_uri
                FROM training_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            out = []
            for row in rows:
                record = dict(row)
                record["config"] = json.loads(record.pop("config_json") or "{}")
                metrics_json = record.pop("metrics_json")
                record["metrics"] = json.loads(metrics_json) if metrics_json else None
                out.append(record)
            return out
