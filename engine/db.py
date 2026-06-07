from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "results.db"

SCHEMA_VERSION = 1

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_metadata (
    run_id          TEXT PRIMARY KEY,
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    total_jobs      INTEGER,
    completed_jobs  INTEGER DEFAULT 0,
    skipped_jobs    INTEGER DEFAULT 0,
    failed_jobs     INTEGER DEFAULT 0,
    catalog_version TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    strategy_id         TEXT NOT NULL,
    pair                TEXT NOT NULL,
    timeframe           TEXT NOT NULL,
    in_sample_start     TEXT NOT NULL,
    in_sample_end       TEXT NOT NULL,
    oos_start           TEXT NOT NULL,
    oos_end             TEXT NOT NULL,
    is_total_trades     INTEGER,
    is_win_rate         REAL,
    is_win_loss_ratio   REAL,
    is_total_return_pct REAL,
    is_max_drawdown_pct REAL,
    is_sharpe_ratio     REAL,
    oos_total_trades    INTEGER,
    oos_win_rate        REAL,
    oos_win_loss_ratio  REAL,
    oos_total_return_pct REAL,
    oos_max_drawdown_pct REAL,
    classification      TEXT NOT NULL,
    classification_reason TEXT,
    status              TEXT NOT NULL,
    skip_reason         TEXT,
    error_message       TEXT,
    duration_seconds    REAL,
    created_at          TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES run_metadata(run_id)
);

CREATE INDEX IF NOT EXISTS idx_results_run_id
    ON backtest_results(run_id);

CREATE INDEX IF NOT EXISTS idx_results_strategy_id
    ON backtest_results(strategy_id);

CREATE INDEX IF NOT EXISTS idx_results_classification
    ON backtest_results(classification);
"""


def _utc_now_iso() -> str:
    """
    Return the current UTC timestamp as an ISO-8601 string.

    Args:
        None.

    Returns:
        ISO-formatted UTC timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    Create the database and all tables if they do not exist.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Open SQLite connection with WAL mode and row factory configured.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_CREATE_TABLES_SQL)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row

    row = conn.execute("SELECT COUNT(*) AS count FROM schema_version").fetchone()
    if row is not None and row["count"] == 0:
        conn.execute(
            "INSERT INTO schema_version (version, created_at) VALUES (?, ?)",
            (SCHEMA_VERSION, _utc_now_iso()),
        )
        conn.commit()
        logger.info("Initialized database schema version %s at %s", SCHEMA_VERSION, db_path)

    return conn


def record_run_start(
    conn: sqlite3.Connection,
    run_id: str,
    total_jobs: int,
    catalog_version: str,
    notes: str,
) -> None:
    """
    Insert a new run metadata row when a backtest run starts.

    Args:
        conn: Open SQLite connection.
        run_id: Unique run identifier.
        total_jobs: Total number of jobs scheduled for the run.
        catalog_version: Strategy catalog version string.
        notes: Optional free-form notes for the run.

    Returns:
        None.
    """
    conn.execute(
        """
        INSERT INTO run_metadata (
            run_id, started_at, total_jobs, catalog_version, notes
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, _utc_now_iso(), total_jobs, catalog_version, notes),
    )


def record_run_complete(
    conn: sqlite3.Connection,
    run_id: str,
    completed: int,
    skipped: int,
    failed: int,
) -> None:
    """
    Update run metadata when a backtest run finishes.

    Args:
        conn: Open SQLite connection.
        run_id: Unique run identifier.
        completed: Number of completed jobs.
        skipped: Number of skipped jobs.
        failed: Number of failed jobs.

    Returns:
        None.
    """
    conn.execute(
        """
        UPDATE run_metadata
        SET completed_at = ?,
            completed_jobs = ?,
            skipped_jobs = ?,
            failed_jobs = ?
        WHERE run_id = ?
        """,
        (_utc_now_iso(), completed, skipped, failed, run_id),
    )


def insert_result(conn: sqlite3.Connection, result: dict[str, Any]) -> None:
    """
    Insert one backtest result row.

    Args:
        conn: Open SQLite connection.
        result: Result dictionary whose keys match `backtest_results` columns.

    Returns:
        None.
    """
    created_at = result.get("created_at") or _utc_now_iso()
    conn.execute(
        """
        INSERT INTO backtest_results (
            run_id,
            strategy_id,
            pair,
            timeframe,
            in_sample_start,
            in_sample_end,
            oos_start,
            oos_end,
            is_total_trades,
            is_win_rate,
            is_win_loss_ratio,
            is_total_return_pct,
            is_max_drawdown_pct,
            is_sharpe_ratio,
            oos_total_trades,
            oos_win_rate,
            oos_win_loss_ratio,
            oos_total_return_pct,
            oos_max_drawdown_pct,
            classification,
            classification_reason,
            status,
            skip_reason,
            error_message,
            duration_seconds,
            created_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            result["run_id"],
            result["strategy_id"],
            result["pair"],
            result["timeframe"],
            result["in_sample_start"],
            result["in_sample_end"],
            result["oos_start"],
            result["oos_end"],
            result["is_total_trades"],
            result["is_win_rate"],
            result["is_win_loss_ratio"],
            result["is_total_return_pct"],
            result["is_max_drawdown_pct"],
            result["is_sharpe_ratio"],
            result["oos_total_trades"],
            result["oos_win_rate"],
            result["oos_win_loss_ratio"],
            result["oos_total_return_pct"],
            result["oos_max_drawdown_pct"],
            result["classification"],
            result.get("classification_reason"),
            result["status"],
            result.get("skip_reason"),
            result.get("error_message"),
            result.get("duration_seconds"),
            created_at,
        ),
    )


def get_results_for_run(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    """
    Fetch all backtest results for a run.

    Args:
        conn: Open SQLite connection.
        run_id: Unique run identifier.

    Returns:
        Result rows ordered by strategy_id, pair, and timeframe.
    """
    cursor = conn.execute(
        """
        SELECT *
        FROM backtest_results
        WHERE run_id = ?
        ORDER BY strategy_id, pair, timeframe
        """,
        (run_id,),
    )
    return list(cursor.fetchall())


def get_promoted_strategies(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    """
    Fetch promoted strategy results for a run.

    Args:
        conn: Open SQLite connection.
        run_id: Unique run identifier.

    Returns:
        Rows where classification is PROMOTE for the given run.
    """
    cursor = conn.execute(
        """
        SELECT *
        FROM backtest_results
        WHERE run_id = ? AND classification = 'PROMOTE'
        ORDER BY strategy_id, pair, timeframe
        """,
        (run_id,),
    )
    return list(cursor.fetchall())


def get_run_summary(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
    """
    Fetch run metadata for a run.

    Args:
        conn: Open SQLite connection.
        run_id: Unique run identifier.

    Returns:
        The matching run_metadata row, or None if not found.
    """
    cursor = conn.execute(
        "SELECT * FROM run_metadata WHERE run_id = ?",
        (run_id,),
    )
    return cursor.fetchone()
