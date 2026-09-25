"""SQLite persistence for uploaded tiles and their predictions."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    image_sha256 TEXT NOT NULL,
    image_path TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('classified', 'uncertain')),
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (image_sha256, model_version)
);
CREATE INDEX IF NOT EXISTS idx_predictions_class_created
    ON predictions(predicted_class, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_status_created
    ON predictions(status, created_at DESC);
"""


class PredictionStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._migrate_hash_only_key(connection)
            connection.executescript(SCHEMA)

    @staticmethod
    def _migrate_hash_only_key(connection: sqlite3.Connection) -> None:
        """Replace the original hash-only unique constraint without losing predictions."""
        if connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'predictions'"
        ).fetchone() is None:
            return

        for index in connection.execute("PRAGMA index_list(predictions)"):
            if not index[2]:
                continue
            index_name = index[1].replace('"', '""')
            columns = [row[2] for row in connection.execute(f'PRAGMA index_info("{index_name}")')]
            if columns != ["image_sha256"]:
                continue

            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("""
                    CREATE TABLE predictions_v2 (
                        id TEXT PRIMARY KEY,
                        image_sha256 TEXT NOT NULL,
                        image_path TEXT NOT NULL,
                        predicted_class TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        status TEXT NOT NULL CHECK (status IN ('classified', 'uncertain')),
                        model_version TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE (image_sha256, model_version)
                    )
                """)
                connection.execute("""
                    INSERT INTO predictions_v2
                    SELECT id, image_sha256, image_path, predicted_class,
                           confidence, status, model_version, created_at
                    FROM predictions
                """)
                connection.execute("DROP TABLE predictions")
                connection.execute("ALTER TABLE predictions_v2 RENAME TO predictions")
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()
            return

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def get_by_hash_and_model(self, image_sha256: str, model_version: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM predictions WHERE image_sha256 = ? AND model_version = ?",
                (image_sha256, model_version),
            ).fetchone()
        return dict(row) if row else None

    def create(self, prediction: dict[str, Any]) -> dict[str, Any]:
        fields = (
            "id",
            "image_sha256",
            "image_path",
            "predicted_class",
            "confidence",
            "status",
            "model_version",
            "created_at",
        )
        values = tuple(prediction[field] for field in fields)
        placeholders = ", ".join("?" for _ in fields)
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO predictions ({', '.join(fields)}) VALUES ({placeholders})",
                values,
            )
        return prediction

    def list(
        self, *, class_name: str | None, status: str | None,
        image_sha256: str | None, limit: int
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if class_name is not None:
            clauses.append("predicted_class = ?")
            parameters.append(class_name)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        if image_sha256 is not None:
            clauses.append("image_sha256 = ?")
            parameters.append(image_sha256)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM predictions {where} ORDER BY created_at DESC LIMIT ?",
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, prediction_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
            ).fetchone()
        return dict(row) if row else None
