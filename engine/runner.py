from __future__ import annotations

import sys
from pathlib import Path as _Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from engine.db import (
    DB_PATH,
    init_db,
    insert_result,
    record_run_complete,
    record_run_start,
)
from strategies import CATALOG, SIGNAL_REGISTRY, list_testable
from strategies.schema import Strategy

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

ALL_PAIRS: list[str] = [
    "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY",
    "GBPUSD", "EURGBP", "GBPAUD", "GBPCAD", "GBPNZD", "GBPCHF",
    "EURUSD", "EURAUD", "EURCAD", "EURNZD", "EURCHF", "EURNOK",
    "AUDUSD", "AUDCAD", "AUDNZD", "AUDCHF",
    "NZDUSD", "NZDCAD", "NZDCHF",
    "USDCAD", "USDCHF", "USDNOK", "USDSEK", "USDZAR",
    "CADCHF",
]

IN_SAMPLE_MONTHS = 18
OOS_MONTHS = 6

MIN_TRADES_IS = 30
MIN_WIN_RATE = 0.45
MIN_WIN_LOSS_RATIO = 1.8

MONITOR_WIN_RATE = 0.40
MONITOR_WIN_LOSS_RATIO = 1.5
MONITOR_MIN_TRADES = 20


@dataclass(frozen=True)
class BacktestJob:
    """
    A single unit of work: one strategy × one pair × one timeframe.

    Fields:
        strategy: The Strategy instance from the catalog.
        pair: Currency pair string e.g. "EURUSD".
        timeframe: One of "1h", "1d", "1wk".
        job_id: Auto-generated unique identifier for this job.
    """

    strategy: Strategy
    pair: str
    timeframe: str
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class BacktestMetrics:
    """
    Computed metrics for one backtest window (in-sample or out-of-sample).

    Fields:
        total_trades: Number of completed trades.
        win_rate: Fraction of winning trades (0.0–1.0).
        win_loss_ratio: Average win / average loss (absolute values).
        total_return_pct: Total percentage return over the window.
        max_drawdown_pct: Maximum drawdown percentage (positive number).
        sharpe_ratio: Annualized Sharpe ratio. None if insufficient data.
    """

    total_trades: int
    win_rate: float
    win_loss_ratio: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float | None = None


def load_ohlcv(
    pair: str,
    timeframe: str,
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame | None:
    """
    Load cached OHLCV data for a pair and timeframe.

    Args:
        pair: Currency pair without =X suffix e.g. "EURUSD".
        timeframe: "1h", "1d", or "1wk".
        cache_dir: Directory containing Parquet cache files.

    Returns:
        DataFrame with columns [open, high, low, close, volume] and DatetimeIndex,
        sorted ascending. Returns None if the file does not exist or fails to load.
    """
    path = cache_dir / f"{pair}_{timeframe}.parquet"
    if not path.exists():
        logger.warning("OHLCV cache file not found: %s", path)
        return None

    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.dropna(subset=["close"])
        return df
    except Exception as exc:
        logger.warning("Failed to load OHLCV from %s: %s: %s", path, type(exc).__name__, exc)
        return None


def split_data(
    df: pd.DataFrame,
    is_months: int = IN_SAMPLE_MONTHS,
    oos_months: int = OOS_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Split OHLCV data into in-sample and out-of-sample windows.

    Args:
        df: Full OHLCV DataFrame sorted ascending.
        is_months: Number of months for the in-sample window.
        oos_months: Number of months for the out-of-sample window.

    Returns:
        Tuple of (in_sample_df, oos_df). Returns None if df is too short to
        produce both windows with at least 30 rows each.
    """
    if df.empty:
        return None

    last_date = df.index.max()
    oos_start = last_date - pd.DateOffset(months=oos_months)
    oos_df = df[df.index >= oos_start]
    if oos_df.empty:
        return None

    oos_first = oos_df.index.min()
    is_end = oos_first - pd.Timedelta(days=1)
    is_start = is_end - pd.DateOffset(months=is_months) + pd.Timedelta(days=1)
    is_df = df[(df.index >= is_start) & (df.index <= is_end)]

    if len(is_df) < 30 or len(oos_df) < 30:
        return None

    return is_df, oos_df


def compute_metrics(portfolio: vbt.Portfolio) -> BacktestMetrics:
    """
    Compute backtest metrics from a vectorbt portfolio.

    Args:
        portfolio: A vectorbt Portfolio object after calling .from_signals().

    Returns:
        BacktestMetrics computed from the portfolio's trade records.
    """
    total_trades = int(portfolio.trades.count())
    if total_trades == 0:
        return BacktestMetrics(0, 0.0, 0.0, 0.0, 0.0, None)

    pnl = np.asarray(portfolio.trades.pnl.values, dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = round(float(len(wins) / len(pnl)), 4)
    if len(losses) == 0:
        win_loss_ratio = 999.0
    else:
        win_loss_ratio = round(float(wins.mean() / abs(losses.mean())), 4)

    total_return_pct = round(float(portfolio.total_return() * 100), 4)
    max_drawdown_pct = round(float(abs(portfolio.max_drawdown()) * 100), 4)

    sharpe_ratio: float | None
    try:
        sharpe_ratio = round(float(portfolio.sharpe_ratio()), 4)
    except Exception:
        sharpe_ratio = None

    return BacktestMetrics(
        total_trades=total_trades,
        win_rate=win_rate,
        win_loss_ratio=win_loss_ratio,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
    )


def classify_result(
    is_metrics: BacktestMetrics,
    oos_metrics: BacktestMetrics,
) -> tuple[str, str]:
    """
    Classify a strategy result using in-sample and out-of-sample metrics.

    Args:
        is_metrics: In-sample BacktestMetrics.
        oos_metrics: Out-of-sample BacktestMetrics.

    Returns:
        Tuple of (classification, reason). Classification is one of
        PROMOTE, MONITOR, ARCHIVE, or INSUFFICIENT.
    """
    if is_metrics.total_trades < MIN_TRADES_IS:
        return (
            "INSUFFICIENT",
            f"Only {is_metrics.total_trades} in-sample trades (minimum {MIN_TRADES_IS})",
        )

    promote_ok = (
        is_metrics.win_rate >= MIN_WIN_RATE
        and is_metrics.win_loss_ratio >= MIN_WIN_LOSS_RATIO
        and is_metrics.total_trades >= MIN_TRADES_IS
        and oos_metrics.total_trades >= 5
        and oos_metrics.win_rate >= (MIN_WIN_RATE - 0.10)
    )
    if promote_ok:
        return (
            "PROMOTE",
            (
                f"WR={is_metrics.win_rate:.1%} WL={is_metrics.win_loss_ratio:.2f} "
                f"trades={is_metrics.total_trades} OOS_WR={oos_metrics.win_rate:.1%}"
            ),
        )

    monitor_checks = [
        is_metrics.win_rate >= MONITOR_WIN_RATE,
        is_metrics.win_loss_ratio >= MONITOR_WIN_LOSS_RATIO,
        is_metrics.total_trades >= MONITOR_MIN_TRADES,
        oos_metrics.win_rate >= MONITOR_WIN_RATE - 0.05,
    ]
    if sum(monitor_checks) >= 3:
        return (
            "MONITOR",
            (
                f"Borderline: WR={is_metrics.win_rate:.1%} "
                f"WL={is_metrics.win_loss_ratio:.2f} trades={is_metrics.total_trades}"
            ),
        )

    return (
        "ARCHIVE",
        (
            f"Failed: WR={is_metrics.win_rate:.1%} "
            f"WL={is_metrics.win_loss_ratio:.2f} trades={is_metrics.total_trades}"
        ),
    )


def build_job_queue(catalog: Any | None = None) -> list[BacktestJob]:
    """
    Build the full queue of backtest jobs.

    Args:
        catalog: StrategyCatalog to use. Defaults to the global CATALOG.

    Returns:
        List of BacktestJob instances covering all valid
        strategy × pair × timeframe combinations.
    """
    if catalog is None:
        catalog = CATALOG

    strategies = list_testable()
    jobs: list[BacktestJob] = []

    for strategy in strategies:
        for timeframe in strategy.timeframes:
            for pair in ALL_PAIRS:
                if strategy.allowed_pairs is not None and pair not in strategy.allowed_pairs:
                    continue
                if pair in strategy.excluded_pairs:
                    continue
                cache_path = CACHE_DIR / f"{pair}_{timeframe}.parquet"
                if not cache_path.exists():
                    continue
                jobs.append(
                    BacktestJob(strategy=strategy, pair=pair, timeframe=timeframe)
                )

    logger.info("Built job queue with %s jobs", len(jobs))
    return jobs


def timeframe_to_freq(timeframe: str) -> str:
    """
    Convert a strategy timeframe to a vectorbt frequency string.

    Args:
        timeframe: One of "1h", "1d", "1wk".

    Returns:
        vectorbt-compatible frequency string.

    Raises:
        ValueError: If the timeframe is unknown.
    """
    mapping = {"1h": "1h", "1d": "1d", "1wk": "1W"}
    if timeframe not in mapping:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    return mapping[timeframe]


def _base_result(job: BacktestJob, run_id: str) -> dict[str, Any]:
    """
    Create the default result dictionary for a job.

    Args:
        job: Backtest job being executed.
        run_id: Run identifier for database recording.

    Returns:
        Base result dictionary with default values.
    """
    return {
        "run_id": run_id,
        "strategy_id": job.strategy.id,
        "pair": job.pair,
        "timeframe": job.timeframe,
        "status": "FAILED",
        "classification": "ARCHIVE",
        "in_sample_start": "",
        "in_sample_end": "",
        "oos_start": "",
        "oos_end": "",
        "is_total_trades": 0,
        "is_win_rate": 0.0,
        "is_win_loss_ratio": 0.0,
        "is_total_return_pct": 0.0,
        "is_max_drawdown_pct": 0.0,
        "is_sharpe_ratio": None,
        "oos_total_trades": 0,
        "oos_win_rate": 0.0,
        "oos_win_loss_ratio": 0.0,
        "oos_total_return_pct": 0.0,
        "oos_max_drawdown_pct": 0.0,
        "classification_reason": "",
        "skip_reason": None,
        "error_message": None,
        "duration_seconds": 0.0,
    }


def _run_portfolio(
    ohlcv_df: pd.DataFrame,
    signals: Any,
    timeframe: str,
) -> vbt.Portfolio:
    """
    Run a vectorbt backtest from signal output.

    Args:
        ohlcv_df: OHLCV DataFrame for the window.
        signals: SignalOutput instance.
        timeframe: Timeframe string for frequency mapping.

    Returns:
        vectorbt Portfolio object.
    """
    return vbt.Portfolio.from_signals(
        close=ohlcv_df["close"],
        entries=signals.long_entries,
        exits=signals.long_exits,
        short_entries=signals.short_entries,
        short_exits=signals.short_exits,
        init_cash=10000.0,
        fees=0.0001,
        slippage=0.0001,
        freq=timeframe_to_freq(timeframe),
    )


def run_single_job(job: BacktestJob, run_id: str) -> dict[str, Any]:
    """
    Execute a single backtest job.

    Args:
        job: The BacktestJob to execute.
        run_id: The run identifier for database recording.

    Returns:
        A result dict ready to be passed to insert_result(). Always returns a dict.
    """
    start_time = time.monotonic()
    result = _base_result(job, run_id)

    try:
        fn = SIGNAL_REGISTRY.get(job.strategy.signal_fn_name)
        if fn is None:
            result["status"] = "SKIPPED"
            result["skip_reason"] = (
                f"signal_fn_name '{job.strategy.signal_fn_name}' not in registry"
            )
            result["duration_seconds"] = round(time.monotonic() - start_time, 3)
            return result

        try:
            df = load_ohlcv(job.pair, job.timeframe)
            if df is None or len(df) < 60:
                result["status"] = "SKIPPED"
                result["skip_reason"] = "Insufficient data"
                result["duration_seconds"] = round(time.monotonic() - start_time, 3)
                return result
        except Exception as exc:
            logger.warning(
                "Data load failed for %s %s %s: %s: %s",
                job.strategy.id,
                job.pair,
                job.timeframe,
                type(exc).__name__,
                exc,
            )
            result["status"] = "FAILED"
            result["error_message"] = f"Data load failed: {type(exc).__name__}: {exc}"
            result["duration_seconds"] = round(time.monotonic() - start_time, 3)
            return result

        try:
            split = split_data(df)
            if split is None:
                result["status"] = "SKIPPED"
                result["skip_reason"] = "Data too short for IS/OOS split"
                result["duration_seconds"] = round(time.monotonic() - start_time, 3)
                return result
            is_df, oos_df = split
        except Exception as exc:
            logger.warning(
                "Data split failed for %s %s %s: %s: %s",
                job.strategy.id,
                job.pair,
                job.timeframe,
                type(exc).__name__,
                exc,
            )
            result["status"] = "FAILED"
            result["error_message"] = f"Data split failed: {type(exc).__name__}: {exc}"
            result["duration_seconds"] = round(time.monotonic() - start_time, 3)
            return result

        result["in_sample_start"] = is_df.index[0].isoformat()
        result["in_sample_end"] = is_df.index[-1].isoformat()
        result["oos_start"] = oos_df.index[0].isoformat()
        result["oos_end"] = oos_df.index[-1].isoformat()

        try:
            is_signals = fn(is_df, job.strategy.params)
            oos_signals = fn(oos_df, job.strategy.params)
        except NotImplementedError:
            result["status"] = "SKIPPED"
            result["skip_reason"] = "Signal function not yet implemented (stub)"
            result["duration_seconds"] = round(time.monotonic() - start_time, 3)
            return result
        except Exception as exc:
            logger.warning(
                "Signal generation failed for %s %s %s: %s: %s",
                job.strategy.id,
                job.pair,
                job.timeframe,
                type(exc).__name__,
                exc,
            )
            result["status"] = "FAILED"
            result["error_message"] = f"Signal generation failed: {type(exc).__name__}: {exc}"
            result["duration_seconds"] = round(time.monotonic() - start_time, 3)
            return result

        try:
            is_portfolio = _run_portfolio(is_df, is_signals, job.timeframe)
            oos_portfolio = _run_portfolio(oos_df, oos_signals, job.timeframe)
        except Exception as exc:
            logger.warning(
                "Backtest failed for %s %s %s: %s: %s",
                job.strategy.id,
                job.pair,
                job.timeframe,
                type(exc).__name__,
                exc,
            )
            result["status"] = "FAILED"
            result["error_message"] = f"Backtest failed: {type(exc).__name__}: {exc}"
            result["duration_seconds"] = round(time.monotonic() - start_time, 3)
            return result

        try:
            is_metrics = compute_metrics(is_portfolio)
            oos_metrics = compute_metrics(oos_portfolio)
            classification, reason = classify_result(is_metrics, oos_metrics)
        except Exception as exc:
            logger.warning(
                "Metrics/classification failed for %s %s %s: %s: %s",
                job.strategy.id,
                job.pair,
                job.timeframe,
                type(exc).__name__,
                exc,
            )
            result["status"] = "FAILED"
            result["error_message"] = (
                f"Metrics/classification failed: {type(exc).__name__}: {exc}"
            )
            result["duration_seconds"] = round(time.monotonic() - start_time, 3)
            return result

        result["status"] = "COMPLETED"
        result["classification"] = classification
        result["classification_reason"] = reason
        result["is_total_trades"] = is_metrics.total_trades
        result["is_win_rate"] = is_metrics.win_rate
        result["is_win_loss_ratio"] = is_metrics.win_loss_ratio
        result["is_total_return_pct"] = is_metrics.total_return_pct
        result["is_max_drawdown_pct"] = is_metrics.max_drawdown_pct
        result["is_sharpe_ratio"] = is_metrics.sharpe_ratio
        result["oos_total_trades"] = oos_metrics.total_trades
        result["oos_win_rate"] = oos_metrics.win_rate
        result["oos_win_loss_ratio"] = oos_metrics.win_loss_ratio
        result["oos_total_return_pct"] = oos_metrics.total_return_pct
        result["oos_max_drawdown_pct"] = oos_metrics.max_drawdown_pct

    except Exception as exc:
        logger.warning(
            "Unexpected failure for %s %s %s: %s: %s",
            job.strategy.id,
            job.pair,
            job.timeframe,
            type(exc).__name__,
            exc,
        )
        result["status"] = "FAILED"
        result["error_message"] = f"Unexpected failure: {type(exc).__name__}: {exc}"

    result["duration_seconds"] = round(time.monotonic() - start_time, 3)
    return result


def run_all(
    run_id: str | None = None,
    db_path: Path = DB_PATH,
    notes: str = "",
) -> str:
    """
    Execute all jobs in the built queue and persist results to SQLite.

    Args:
        run_id: Optional run identifier. Auto-generated if not provided.
        db_path: Path to SQLite database.
        notes: Optional notes to store with this run.

    Returns:
        The run_id string.
    """
    if run_id is None:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / f"run_{run_id}.log"),
        ],
        force=True,
    )

    conn = init_db(db_path)
    jobs = build_job_queue()
    record_run_start(
        conn,
        run_id,
        total_jobs=len(jobs),
        catalog_version=CATALOG.version,
        notes=notes,
    )
    conn.commit()

    completed = 0
    skipped = 0
    failed = 0

    for index, job in enumerate(jobs):
        if (index + 1) % 100 == 0 or index == 0:
            logger.info(
                "Progress: %s/%s | completed=%s skipped=%s failed=%s | current: %s %s %s",
                index + 1,
                len(jobs),
                completed,
                skipped,
                failed,
                job.strategy.id,
                job.pair,
                job.timeframe,
            )

        try:
            result = run_single_job(job, run_id)
            insert_result(conn, result)
            conn.commit()
        except Exception as exc:
            logger.warning(
                "Failed to persist result for %s %s %s: %s: %s",
                job.strategy.id,
                job.pair,
                job.timeframe,
                type(exc).__name__,
                exc,
            )
            failed += 1
            continue

        if result["status"] == "COMPLETED":
            completed += 1
        elif result["status"] == "SKIPPED":
            skipped += 1
        else:
            failed += 1

    record_run_complete(conn, run_id, completed, skipped, failed)
    conn.commit()

    logger.info(
        "Run %s complete. Total=%s Completed=%s Skipped=%s Failed=%s",
        run_id,
        len(jobs),
        completed,
        skipped,
        failed,
    )

    conn.close()
    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run APEX backtesting engine")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run ID (auto-generated if not provided)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help=f"SQLite database path (default: {DB_PATH})",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional notes for this run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build job queue and print count without running",
    )
    args = parser.parse_args()

    if args.dry_run:
        jobs = build_job_queue()
        print(f"Dry run: {len(jobs)} jobs would be executed")
        for strategy_id in sorted({job.strategy.id for job in jobs}):
            count = sum(1 for job in jobs if job.strategy.id == strategy_id)
            print(f"  {strategy_id}: {count} jobs")
    else:
        run_id = run_all(
            run_id=args.run_id,
            db_path=args.db_path,
            notes=args.notes,
        )
        print(f"Run complete: {run_id}")
