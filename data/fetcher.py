"""Download and cache historical forex OHLCV data via yfinance."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf

PAIRS = [
    # JPY pairs
    "USDJPY=X",
    "EURJPY=X",
    "GBPJPY=X",
    "AUDJPY=X",
    "CADJPY=X",
    "CHFJPY=X",
    "NZDJPY=X",
    # GBP pairs
    "GBPUSD=X",
    "EURGBP=X",
    "GBPAUD=X",
    "GBPCAD=X",
    "GBPNZD=X",
    "GBPCHF=X",
    # EUR pairs
    "EURUSD=X",
    "EURAUD=X",
    "EURCAD=X",
    "EURNZD=X",
    "EURCHF=X",
    "EURNOK=X",
    # AUD pairs
    "AUDUSD=X",
    "AUDCAD=X",
    "AUDNZD=X",
    "AUDCHF=X",
    # NZD pairs
    "NZDUSD=X",
    "NZDCAD=X",
    "NZDCHF=X",
    # USD pairs
    "USDCAD=X",
    "USDCHF=X",
    "USDNOK=X",
    "USDSEK=X",
    "USDZAR=X",
    # CAD pairs
    "CADCHF=X",
]

TIMEFRAMES: dict[str, str] = {
    "1d": "1d",
    "1wk": "1wk",
    "1h": "1h",
}

# Forex trades ~252 weekdays per year; hourly assumes 24h x 5 days/week.
YEARS = 3
PERIOD = f"{YEARS}y"
# yfinance only serves ~730 days of hourly data; use that as the 1h/4h proxy window.
HOURLY_PERIOD = "730d"
HOURLY_DAYS = 730

TIMEFRAME_PERIODS: dict[str, str] = {
    "1d": PERIOD,
    "1wk": PERIOD,
    "1h": HOURLY_PERIOD,
}

EXPECTED_CANDLES: dict[str, int] = {
    "1d": 252 * YEARS,
    "1wk": 52 * YEARS,
    "1h": int(HOURLY_DAYS * 5 / 7 * 24),
}
CACHE_DIR = Path(__file__).resolve().parent / "cache"
MIN_COVERAGE_RATIO = 0.80

FetchStatus = Literal["ok", "warned", "failed", "skipped"]


@dataclass
class FetchSummary:
    """Aggregate results from a fetch run."""

    ok: int = 0
    warned: int = 0
    failed: int = 0
    skipped: int = 0
    details: list[str] = field(default_factory=list)

    def record(self, status: FetchStatus, message: str) -> None:
        setattr(self, status, getattr(self, status) + 1)
        self.details.append(message)

    def print_report(self) -> None:
        total = self.ok + self.warned + self.failed + self.skipped
        print("\n" + "=" * 60)
        print("FETCH SUMMARY")
        print("=" * 60)
        print(f"Total jobs:  {total}")
        print(f"  OK:        {self.ok}")
        print(f"  Warned:    {self.warned}  (sparse data, <{MIN_COVERAGE_RATIO:.0%} coverage)")
        print(f"  Failed:    {self.failed}")
        print(f"  Skipped:   {self.skipped}  (cache still fresh)")
        if self.details:
            print("-" * 60)
            for line in self.details:
                print(line)
        print("=" * 60)


def ticker_to_filename(ticker: str) -> str:
    """Convert yfinance ticker to cache filename stem (e.g. EURUSD=X -> EURUSD)."""
    return ticker.removesuffix("=X")


def cache_path(ticker: str, timeframe: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{ticker_to_filename(ticker)}_{timeframe}.parquet"


def is_cache_fresh(path: Path, refresh_if_older_than_days: int | None) -> bool:
    if refresh_if_older_than_days is None or not path.exists():
        return False
    age_days = (time.time() - path.stat().st_mtime) / 86_400
    return age_days < refresh_if_older_than_days


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output to a standard OHLCV DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def download_pair(
    ticker: str,
    timeframe: str,
    *,
    period: str = PERIOD,
    cache_dir: Path = CACHE_DIR,
    refresh_if_older_than_days: int | None = None,
    summary: FetchSummary | None = None,
) -> pd.DataFrame | None:
    """
    Download (or load cached) OHLCV data for one pair and timeframe.

    Returns the DataFrame on success, None on failure. When *summary* is provided,
    status is recorded automatically.
    """
    interval = TIMEFRAMES[timeframe]
    fetch_period = TIMEFRAME_PERIODS.get(timeframe, period)
    out_path = cache_path(ticker, timeframe, cache_dir)
    label = f"{ticker_to_filename(ticker)}_{timeframe}"

    if is_cache_fresh(out_path, refresh_if_older_than_days):
        msg = f"[SKIPPED] {label}: cache fresh (< {refresh_if_older_than_days} days old)"
        print(msg)
        if summary is not None:
            summary.record("skipped", msg)
        return pd.read_parquet(out_path)

    try:
        raw = yf.download(
            ticker,
            period=fetch_period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            actions=False,
        )
        df = normalize_ohlcv(raw)
        if df.empty:
            raise ValueError("yfinance returned no rows")

        expected = EXPECTED_CANDLES[timeframe]
        coverage = len(df) / expected
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)

        if coverage < MIN_COVERAGE_RATIO:
            msg = (
                f"[WARN] {label}: only {len(df)}/{expected} candles "
                f"({coverage:.1%}) — potentially sparse (exotic pair issue?)"
            )
            print(msg)
            if summary is not None:
                summary.record("warned", msg)
        else:
            msg = f"[OK] {label}: {len(df)} candles saved -> {out_path}"
            print(msg)
            if summary is not None:
                summary.record("ok", msg)

        return df

    except Exception as exc:
        msg = f"[FAILED] {label}: {type(exc).__name__}: {exc}"
        print(msg)
        if summary is not None:
            summary.record("failed", msg)
        return None


def fetch_all(
    pairs: list[str] | None = None,
    timeframes: list[str] | None = None,
    *,
    period: str = PERIOD,
    cache_dir: Path = CACHE_DIR,
    refresh_if_older_than_days: int | None = 7,
) -> FetchSummary:
    """
    Download historical OHLCV for all configured pairs and timeframes.

    Parameters
    ----------
    pairs:
        yfinance tickers to fetch. Defaults to :data:`PAIRS`.
    timeframes:
        Keys from :data:`TIMEFRAMES` (``1d``, ``1wk``, ``1h``).
    period:
        yfinance lookback period (default ``3y``).
    cache_dir:
        Directory for Parquet cache files.
    refresh_if_older_than_days:
        Skip re-download when the cache file is newer than this many days.
        Set to ``None`` to always re-download.
    """
    pairs = pairs or PAIRS
    timeframes = timeframes or list(TIMEFRAMES)
    summary = FetchSummary()

    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"Forex data fetch started at {started}")
    print(f"Pairs: {len(pairs)} | Timeframes: {', '.join(timeframes)} | Period: {period}")
    if refresh_if_older_than_days is not None:
        print(f"Cache refresh threshold: {refresh_if_older_than_days} days")
    print()

    for ticker in pairs:
        for timeframe in timeframes:
            download_pair(
                ticker,
                timeframe,
                period=period,
                cache_dir=cache_dir,
                refresh_if_older_than_days=refresh_if_older_than_days,
                summary=summary,
            )

    summary.print_report()
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch forex OHLCV data via yfinance")
    parser.add_argument(
        "--refresh-if-older-than-days",
        type=int,
        default=7,
        help="Skip download when cache is newer than N days (default: 7). Use -1 to always refresh.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help=f"Parquet cache directory (default: {CACHE_DIR})",
    )
    args = parser.parse_args()

    refresh_days = None if args.refresh_if_older_than_days < 0 else args.refresh_if_older_than_days
    fetch_all(
        cache_dir=args.cache_dir,
        refresh_if_older_than_days=refresh_days,
    )


if __name__ == "__main__":
    main()
