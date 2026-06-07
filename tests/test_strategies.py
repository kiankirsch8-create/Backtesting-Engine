from __future__ import annotations

import pandas as pd
import pytest

from strategies import CATALOG, SIGNAL_REGISTRY, get_strategy
from strategies.schema import SignalOutput, Strategy


class TestCatalog:
    def test_catalog_has_100_strategies(self) -> None:
        assert len(CATALOG) == 100

    def test_all_ids_unique(self) -> None:
        ids = [s.id for s in CATALOG]
        assert len(ids) == len(set(ids))

    def test_no_blocked_strategies(self) -> None:
        for s in CATALOG:
            assert s.status != "BLOCKED"

    def test_all_signal_fns_registered(self) -> None:
        for s in CATALOG:
            assert s.signal_fn_name in SIGNAL_REGISTRY, (
                f"{s.id} references unregistered signal: {s.signal_fn_name}"
            )

    def test_all_timeframes_valid(self) -> None:
        valid = {"1h", "1d", "1wk"}
        for s in CATALOG:
            assert len(s.timeframes) > 0
            for tf in s.timeframes:
                assert tf in valid

    def test_fragile_and_all_weather_mutually_exclusive(self) -> None:
        for s in CATALOG:
            assert not (s.fragile and s.all_weather)

    def test_m02_excludes_1h(self) -> None:
        m02 = get_strategy("M02_MACD_ZERO_CROSS")
        assert "1h" not in m02.timeframes


class TestSignalContract:
    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Load a real cached Parquet to test against."""
        return pd.read_parquet("data/cache/EURUSD_1d.parquet")

    @pytest.mark.parametrize(
        "strategy_id",
        [
            "T11_SUPERTREND",
            "T12_HULL_MA_CROSS",
            "T13_ICHIMOKU_CLOUD_BREAK",
            "T15_TRIPLE_EMA",
            "T16_TK_CROSS",
            "T18_HH_HL_TREND_COUNT",
            "M07_CCI_MOMENTUM",
            "M09_ROC_THRESHOLD",
            "M10_STOCHASTIC_CROSS_TREND",
            "M12_MOMENTUM_BURST",
            "B15_N_BAR_HIGH_BREAK",
            "R07_CCI_EXTREME_REVERSAL",
            "R08_WILLIAMS_R_REVERSION",
            "V04_BOLLINGER_WALK",
            "V06_SQUEEZE_BREAK",
            "Q07_ZSCORE_REVERSION",
            "PA01_PIN_BAR_AT_LEVEL",
            "PA02_ENGULFING_AT_LEVEL",
            "T02_EMA_CROSSOVER",
            "T07_MA_RIBBON_ALIGNMENT",
            "M02_MACD_ZERO_CROSS",
            "M05_STOCHASTIC_MOMENTUM",
            "R02_BOLLINGER_REVERSION",
            "V01_VOLATILITY_BREAKOUT",
            "V03_SQUEEZE_MOMENTUM",
        ],
    )
    def test_signal_fn_returns_valid_output(
        self, sample_df: pd.DataFrame, strategy_id: str
    ) -> None:
        """Every implemented signal fn must return a valid SignalOutput."""
        strategy = get_strategy(strategy_id)
        fn = SIGNAL_REGISTRY[strategy.signal_fn_name]
        result = fn(sample_df, strategy.params)
        assert isinstance(result, SignalOutput)
        for series_name in ("long_entries", "long_exits", "short_entries", "short_exits"):
            s = getattr(result, series_name)
            assert s.dtype == bool, f"{strategy_id}.{series_name} not bool"
            assert len(s) == len(sample_df), f"{strategy_id}.{series_name} length mismatch"
            assert not s.isna().any(), f"{strategy_id}.{series_name} contains NaN"

    def test_stub_signals_raise_not_implemented(self, sample_df: pd.DataFrame) -> None:
        """Stub signal functions should raise NotImplementedError when called."""
        fn = SIGNAL_REGISTRY["signal_apex_native_placeholder"]
        with pytest.raises(NotImplementedError):
            fn(sample_df, {})
