from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from engine.db import (
    get_run_summary,
    init_db,
    insert_result,
    record_run_complete,
    record_run_start,
)
from engine.runner import (
    BacktestJob,
    BacktestMetrics,
    build_job_queue,
    classify_result,
    load_ohlcv,
    run_single_job,
    split_data,
    timeframe_to_freq,
)
from strategies import get_strategy


class TestDatabase:
    def test_init_db_creates_tables(self, tmp_path: Path) -> None:
        conn = init_db(tmp_path / "test.db")
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "run_metadata" in tables
        assert "backtest_results" in tables
        assert "schema_version" in tables
        conn.close()

    def test_record_run_lifecycle(self, tmp_path: Path) -> None:
        conn = init_db(tmp_path / "test.db")
        record_run_start(
            conn,
            "test_run_001",
            total_jobs=100,
            catalog_version="v1.0.0",
            notes="test",
        )
        record_run_complete(conn, "test_run_001", completed=80, skipped=15, failed=5)
        summary = get_run_summary(conn, "test_run_001")
        assert summary is not None
        assert summary["completed_jobs"] == 80
        assert summary["skipped_jobs"] == 15
        assert summary["failed_jobs"] == 5
        conn.close()

    def test_insert_result(self, tmp_path: Path) -> None:
        conn = init_db(tmp_path / "test.db")
        record_run_start(conn, "r001", 1, "v1", "")
        insert_result(
            conn,
            {
                "run_id": "r001",
                "strategy_id": "T11_SUPERTREND",
                "pair": "EURUSD",
                "timeframe": "1d",
                "in_sample_start": "2023-01-01",
                "in_sample_end": "2024-06-30",
                "oos_start": "2024-07-01",
                "oos_end": "2024-12-31",
                "is_total_trades": 45,
                "is_win_rate": 0.51,
                "is_win_loss_ratio": 2.1,
                "is_total_return_pct": 12.5,
                "is_max_drawdown_pct": 8.2,
                "is_sharpe_ratio": 1.3,
                "oos_total_trades": 12,
                "oos_win_rate": 0.50,
                "oos_win_loss_ratio": 1.9,
                "oos_total_return_pct": 3.2,
                "oos_max_drawdown_pct": 4.1,
                "classification": "PROMOTE",
                "classification_reason": "All criteria met",
                "status": "COMPLETED",
                "skip_reason": None,
                "error_message": None,
                "duration_seconds": 0.42,
            },
        )
        conn.commit()
        rows = conn.execute(
            "SELECT * FROM backtest_results WHERE strategy_id='T11_SUPERTREND'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["classification"] == "PROMOTE"
        conn.close()


class TestDataLoading:
    def test_load_ohlcv_eurusd_1d(self) -> None:
        df = load_ohlcv("EURUSD", "1d")
        assert df is not None
        assert len(df) > 100
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.is_monotonic_increasing

    def test_load_ohlcv_missing_pair(self) -> None:
        df = load_ohlcv("XYZABC", "1d")
        assert df is None

    def test_split_data_correct_proportions(self) -> None:
        df = load_ohlcv("EURUSD", "1d")
        assert df is not None
        result = split_data(df)
        assert result is not None
        is_df, oos_df = result
        assert len(is_df) >= 30
        assert len(oos_df) >= 30
        assert is_df.index[-1] < oos_df.index[0]

    def test_split_data_too_short(self) -> None:
        df = load_ohlcv("EURUSD", "1d")
        assert df is not None
        tiny_df = df.iloc[:40]
        result = split_data(tiny_df)
        assert result is None


class TestClassification:
    def test_promote_criteria(self) -> None:
        is_m = BacktestMetrics(50, 0.50, 2.0, 15.0, 5.0, 1.5)
        oos_m = BacktestMetrics(10, 0.45, 1.9, 5.0, 3.0, 1.2)
        classification, reason = classify_result(is_m, oos_m)
        assert classification == "PROMOTE"

    def test_insufficient_trades(self) -> None:
        is_m = BacktestMetrics(10, 0.60, 3.0, 20.0, 2.0, 2.0)
        oos_m = BacktestMetrics(3, 0.60, 3.0, 5.0, 1.0, 2.0)
        classification, _ = classify_result(is_m, oos_m)
        assert classification == "INSUFFICIENT"

    def test_archive_poor_win_rate(self) -> None:
        is_m = BacktestMetrics(50, 0.30, 1.2, -5.0, 20.0, -0.5)
        oos_m = BacktestMetrics(15, 0.25, 1.1, -3.0, 10.0, None)
        classification, _ = classify_result(is_m, oos_m)
        assert classification == "ARCHIVE"

    def test_monitor_borderline(self) -> None:
        is_m = BacktestMetrics(30, 0.42, 1.6, 8.0, 10.0, 0.8)
        oos_m = BacktestMetrics(8, 0.37, 1.5, 2.0, 5.0, None)
        classification, _ = classify_result(is_m, oos_m)
        assert classification == "MONITOR"


class TestJobQueue:
    def test_job_queue_builds(self) -> None:
        jobs = build_job_queue()
        assert len(jobs) > 0
        assert len(jobs) < 12000

    def test_all_jobs_have_cached_data(self) -> None:
        jobs = build_job_queue()
        cache_dir = Path("data/cache")
        for job in jobs[:50]:
            path = cache_dir / f"{job.pair}_{job.timeframe}.parquet"
            assert path.exists(), f"Missing cache: {path}"

    def test_m02_has_no_1h_jobs(self) -> None:
        jobs = build_job_queue()
        m02_jobs = [job for job in jobs if job.strategy.id == "M02_MACD_ZERO_CROSS"]
        assert all(job.timeframe != "1h" for job in m02_jobs)


class TestSingleJob:
    def test_stub_strategy_returns_skipped(self) -> None:
        strategy = get_strategy("T05_HIGHER_TIMEFRAME_TREND")
        job = BacktestJob(strategy=strategy, pair="EURUSD", timeframe="1d")
        result = run_single_job(job, "test_run")
        assert result["status"] == "SKIPPED"
        assert (
            "stub" in result["skip_reason"].lower()
            or "not yet implemented" in result["skip_reason"].lower()
        )

    def test_implemented_strategy_completes(self) -> None:
        strategy = get_strategy("T11_SUPERTREND")
        job = BacktestJob(strategy=strategy, pair="EURUSD", timeframe="1d")
        result = run_single_job(job, "test_run")
        assert result["status"] in ("COMPLETED", "SKIPPED")
        assert result["strategy_id"] == "T11_SUPERTREND"
        assert result["pair"] == "EURUSD"

    def test_invalid_pair_returns_skipped(self) -> None:
        strategy = get_strategy("T11_SUPERTREND")
        job = BacktestJob(strategy=strategy, pair="XYZABC", timeframe="1d")
        result = run_single_job(job, "test_run")
        assert result["status"] == "SKIPPED"

    def test_result_dict_has_all_required_keys(self) -> None:
        required_keys = {
            "run_id",
            "strategy_id",
            "pair",
            "timeframe",
            "status",
            "classification",
            "in_sample_start",
            "in_sample_end",
            "oos_start",
            "oos_end",
            "is_total_trades",
            "is_win_rate",
            "duration_seconds",
        }
        strategy = get_strategy("T11_SUPERTREND")
        job = BacktestJob(strategy=strategy, pair="EURUSD", timeframe="1d")
        result = run_single_job(job, "test_run")
        assert required_keys.issubset(result.keys())


class TestTimeframeFreq:
    def test_known_timeframes(self) -> None:
        assert timeframe_to_freq("1h") == "1h"
        assert timeframe_to_freq("1d") == "1d"
        assert timeframe_to_freq("1wk") == "1W"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            timeframe_to_freq("5m")
