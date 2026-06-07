from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import pandas as pd

StrategyType = Literal[
    "TREND", "MOMENTUM", "BREAKOUT", "REVERSION",
    "VOLATILITY", "SMC", "QUANT", "SESSION", "MTF", "PRICE_ACTION"
]

StrategyStatus = Literal["LOCKED", "TESTING", "UNTESTED", "BLOCKED", "NEW_CANDIDATE"]

Timeframe = Literal["1h", "1d", "1wk"]

Regime = Literal["STRONG_TAILWIND", "TAILWIND", "NEUTRAL", "HEADWIND"]

Direction = Literal["LONG", "SHORT", "BOTH"]

_VALID_TIMEFRAMES = frozenset({"1h", "1d", "1wk"})
_ID_PATTERN = re.compile(r"^[A-Z]+\d+_[A-Z0-9_]+$")
_UNTESTED_PATTERN = re.compile(r"^UNTESTED_\d+$")


@dataclass(frozen=True)
class SignalOutput:
    """
    Output of every signal function. All four Series MUST be:
    - boolean dtype
    - same length as input DataFrame
    - same index as input DataFrame
    - NaN-free (fillna(False) before returning)
    """

    long_entries: pd.Series
    long_exits: pd.Series
    short_entries: pd.Series
    short_exits: pd.Series

    def validate(self) -> None:
        """
        Raise ValueError if any Series is invalid. Called by runner before use.

        Args:
            None.

        Returns:
            None. Raises ValueError on invalid output.
        """
        series_map = {
            "long_entries": self.long_entries,
            "long_exits": self.long_exits,
            "short_entries": self.short_entries,
            "short_exits": self.short_exits,
        }
        lengths = {name: len(series) for name, series in series_map.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"SignalOutput series length mismatch: {lengths}")

        for name, series in series_map.items():
            if series.dtype != bool:
                raise ValueError(f"{name} dtype must be bool, got {series.dtype}")
            if series.isna().any():
                raise ValueError(f"{name} contains NaN values")


@dataclass(frozen=True)
class Strategy:
    """
    A single strategy definition. Immutable — to change params, create a new instance.

    Fields:
        id: Unique uppercase identifier. Pattern: {TYPE_PREFIX}{NUMBER}_{DESCRIPTION}.
            Examples: "T11_SUPERTREND", "M07_CCI_MOMENTUM".
        name: Human-readable name.
        strategy_type: Category from StrategyType literal.
        status: Lifecycle status from StrategyStatus literal.
        signal_fn_name: Name of function in strategies/signals.py to call.
            Must be a key in signals.SIGNAL_REGISTRY.
        params: Dict of parameters passed to the signal function.
            All values must be JSON-serializable (no callables, no DataFrames).
        timeframes: List of timeframes this strategy is valid on.
            Subset of ["1h", "1d", "1wk"]. Cannot be empty.
        allowed_pairs: If not None, ONLY these pairs are valid. Pair format: "EURUSD"
            (no =X suffix). If None, all pairs are allowed (subject to excluded_pairs).
        excluded_pairs: Pairs explicitly forbidden for this strategy. Same format.
            Applied AFTER allowed_pairs filter.
        allowed_regimes: Macro regimes during which this strategy is permitted.
            Default: all four regimes.
        direction: LONG, SHORT, or BOTH. Default BOTH.
        min_trades: Minimum trades required before drawing conclusions. Default 30.
        fragile: If True, skip during BAD periods (rolling 30-day WR < 45%). Default False.
        all_weather: If True, always run regardless of regime. Default False.
            Cannot be True if fragile=True. Validation must enforce this.
        recipe_boost_pairs: Pairs that get extra position size when paired with
            STRONG_TAILWIND. From APEX rule 5. Default empty.
        monday_boost: If True, 1.25x position size on Mondays. Default True (system-wide).
        description: 1-2 sentence plain-English explanation.
        source_rationale: Where this strategy comes from. One of:
            "APEX_NATIVE" — already exists in APEX strategies_v5_data.py
            "ACADEMIC" — from forex trading literature
            "TECHNICAL" — well-known technical analysis pattern
            "QUANT_RESEARCH" — quantitative finance approach
            "CALENDAR" — session/time-based
    """

    id: str
    name: str
    strategy_type: StrategyType
    status: StrategyStatus
    signal_fn_name: str
    params: dict[str, Any]
    timeframes: list[Timeframe]
    description: str
    source_rationale: str
    allowed_pairs: list[str] | None = None
    excluded_pairs: list[str] = field(default_factory=list)
    allowed_regimes: list[Regime] = field(
        default_factory=lambda: ["STRONG_TAILWIND", "TAILWIND", "NEUTRAL", "HEADWIND"]
    )
    direction: Direction = "BOTH"
    min_trades: int = 30
    fragile: bool = False
    all_weather: bool = False
    recipe_boost_pairs: list[str] = field(default_factory=list)
    monday_boost: bool = True

    def __post_init__(self) -> None:
        """
        Validation that runs on every Strategy instantiation.

        Args:
            None.

        Returns:
            None. Raises ValueError on invalid configuration.
        """
        if not (_ID_PATTERN.match(self.id) or _UNTESTED_PATTERN.match(self.id)):
            raise ValueError(
                f"Strategy id '{self.id}' must match "
                r"^[A-Z]+\d+_[A-Z0-9_]+$ or UNTESTED_NNN"
            )
        if not self.timeframes:
            raise ValueError(f"Strategy '{self.id}' must have at least one timeframe")
        invalid_tf = set(self.timeframes) - _VALID_TIMEFRAMES
        if invalid_tf:
            raise ValueError(
                f"Strategy '{self.id}' has invalid timeframes: {sorted(invalid_tf)}"
            )
        if not self.allowed_regimes:
            raise ValueError(f"Strategy '{self.id}' must have at least one allowed regime")
        if self.fragile and self.all_weather:
            raise ValueError(
                f"Strategy '{self.id}' cannot have both fragile=True and all_weather=True"
            )
        if self.min_trades < 1:
            raise ValueError(
                f"Strategy '{self.id}' min_trades must be >= 1, got {self.min_trades}"
            )


@dataclass
class StrategyCatalog:
    """
    Container for all strategies. Provides filter and lookup methods.

    Fields:
        strategies: List of Strategy instances. Order is preserved.
        version: Catalog version string, e.g. "v1.0.0-2026-06-07".
    """

    strategies: list[Strategy]
    version: str

    def __post_init__(self) -> None:
        """
        Validate the catalog as a whole.

        Args:
            None.

        Returns:
            None. Raises ValueError on invalid catalog state.
        """
        ids = [s.id for s in self.strategies]
        seen: dict[str, int] = {}
        duplicates: list[str] = []
        for strategy_id in ids:
            seen[strategy_id] = seen.get(strategy_id, 0) + 1
            if seen[strategy_id] == 2:
                duplicates.append(strategy_id)
        if duplicates:
            raise ValueError(f"Duplicate strategy IDs found: {sorted(duplicates)}")

        blocked = [s.id for s in self.strategies if s.status == "BLOCKED"]
        if blocked:
            raise ValueError(f"BLOCKED strategies must not be in catalog: {blocked}")

        from strategies.signals import SIGNAL_REGISTRY

        missing = [
            s.id for s in self.strategies if s.signal_fn_name not in SIGNAL_REGISTRY
        ]
        if missing:
            names = {s.id: s.signal_fn_name for s in self.strategies if s.id in missing}
            raise ValueError(f"Unregistered signal functions in catalog: {names}")

    def get_by_id(self, strategy_id: str) -> Strategy:
        """
        Return the Strategy with the given id.

        Args:
            strategy_id: Unique strategy identifier.

        Returns:
            Matching Strategy instance.

        Raises:
            KeyError: If no strategy with the given id exists.
        """
        for strategy in self.strategies:
            if strategy.id == strategy_id:
                return strategy
        raise KeyError(f"Strategy not found: {strategy_id}")

    def filter(
        self,
        *,
        strategy_type: StrategyType | None = None,
        status: StrategyStatus | None = None,
        timeframe: Timeframe | None = None,
        pair: str | None = None,
        regime: Regime | None = None,
    ) -> list[Strategy]:
        """
        Return strategies matching ALL provided filters. None means no filter on that field.

        For pair filter:
            - If strategy.allowed_pairs is None and pair not in excluded_pairs: include.
            - If strategy.allowed_pairs is set and pair in allowed_pairs and not in
              excluded_pairs: include.

        Args:
            strategy_type: Optional strategy category filter.
            status: Optional lifecycle status filter.
            timeframe: Optional timeframe filter.
            pair: Optional pair filter (e.g. "EURUSD").
            regime: Optional macro regime filter.

        Returns:
            List of Strategy instances matching all provided filters.
        """
        results: list[Strategy] = []
        for strategy in self.strategies:
            if strategy_type is not None and strategy.strategy_type != strategy_type:
                continue
            if status is not None and strategy.status != status:
                continue
            if timeframe is not None and timeframe not in strategy.timeframes:
                continue
            if regime is not None and regime not in strategy.allowed_regimes:
                continue
            if pair is not None:
                if pair in strategy.excluded_pairs:
                    continue
                if strategy.allowed_pairs is not None and pair not in strategy.allowed_pairs:
                    continue
            results.append(strategy)
        return results

    def __len__(self) -> int:
        """
        Return the number of strategies in the catalog.

        Args:
            None.

        Returns:
            Strategy count.
        """
        return len(self.strategies)

    def __iter__(self):
        """
        Iterate over strategies in catalog order.

        Args:
            None.

        Returns:
            Iterator over Strategy instances.
        """
        return iter(self.strategies)
