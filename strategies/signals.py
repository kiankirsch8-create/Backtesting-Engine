from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import ta

from strategies.schema import SignalOutput


def _safe_bool_series(s: pd.Series, index: pd.Index) -> pd.Series:
    """
    Convert any Series to a boolean Series aligned to index, NaN→False.

    Args:
        s: Input series to convert.
        index: Target index for alignment.

    Returns:
        Boolean series aligned to index with NaNs filled as False.
    """
    aligned = s.reindex(index)
    return aligned.fillna(False).astype(bool)


def _crosses_above(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    True on bars where a crosses from <=b to >b. Look-ahead safe.

    Args:
        a: First series.
        b: Second series.

    Returns:
        Boolean series marking upward crosses.
    """
    prev_a = a.shift(1)
    prev_b = b.shift(1)
    return (a > b) & (prev_a <= prev_b)


def _crosses_below(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    True on bars where a crosses from >=b to <b. Look-ahead safe.

    Args:
        a: First series.
        b: Second series.

    Returns:
        Boolean series marking downward crosses.
    """
    prev_a = a.shift(1)
    prev_b = b.shift(1)
    return (a < b) & (prev_a >= prev_b)


def _rolling_high(s: pd.Series, window: int) -> pd.Series:
    """
    Rolling high of previous N bars (excludes current bar to avoid look-ahead).

    Args:
        s: Price series.
        window: Lookback window.

    Returns:
        Rolling high of prior bars.
    """
    return s.shift(1).rolling(window).max()


def _rolling_low(s: pd.Series, window: int) -> pd.Series:
    """
    Rolling low of previous N bars (excludes current bar to avoid look-ahead).

    Args:
        s: Price series.
        window: Lookback window.

    Returns:
        Rolling low of prior bars.
    """
    return s.shift(1).rolling(window).min()


def _make_output(
    index: pd.Index,
    long_entries: pd.Series,
    long_exits: pd.Series,
    short_entries: pd.Series,
    short_exits: pd.Series,
) -> SignalOutput:
    """
    Build a validated SignalOutput with NaN-safe boolean series.

    Args:
        index: Target index.
        long_entries: Long entry signals.
        long_exits: Long exit signals.
        short_entries: Short entry signals.
        short_exits: Short exit signals.

    Returns:
        SignalOutput with boolean, NaN-free series.
    """
    return SignalOutput(
        long_entries=_safe_bool_series(long_entries, index),
        long_exits=_safe_bool_series(long_exits, index),
        short_entries=_safe_bool_series(short_entries, index),
        short_exits=_safe_bool_series(short_exits, index),
    )


def _empty_output(df: pd.DataFrame) -> SignalOutput:
    """
    Return an all-False SignalOutput for the given DataFrame.

    Args:
        df: OHLCV DataFrame.

    Returns:
        Empty SignalOutput.
    """
    false = pd.Series(False, index=df.index)
    return _make_output(df.index, false, false, false, false)


def _consecutive_true(condition: pd.Series, bars: int) -> pd.Series:
    """
    True when condition has held for exactly bars consecutive bars starting this bar.

    Args:
        condition: Boolean condition series.
        bars: Required consecutive bar count.

    Returns:
        Boolean series marking the start of a confirmed streak.
    """
    streak = condition.rolling(bars).min().fillna(False).astype(bool)
    return streak & ~streak.shift(1).fillna(False).astype(bool)


def _ema(series: pd.Series, period: int) -> pd.Series:
    """
    Compute EMA using pandas-ta.

    Args:
        series: Input price series.
        period: EMA period.

    Returns:
        EMA series.
    """
    return ta.trend.ema_indicator(series, window=period)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute ATR using pandas-ta.

    Args:
        df: OHLCV DataFrame.
        period: ATR period.

    Returns:
        ATR series.
    """
    return ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=period
    )


def _hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average via EMA composition."""
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    ema_half = ta.trend.ema_indicator(series, window=half)
    ema_full = ta.trend.ema_indicator(series, window=period)
    raw = 2 * ema_half - ema_full
    return ta.trend.ema_indicator(raw, window=sqrt_n)


def _adx(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ADX, +DI, -DI from the ta library."""
    from ta.trend import ADXIndicator

    indicator = ADXIndicator(df["high"], df["low"], df["close"], window=period)
    return indicator.adx(), indicator.adx_pos(), indicator.adx_neg()


def _supertrend(df: pd.DataFrame, period: int, multiplier: float) -> pd.Series:
    """Supertrend direction series (+1 bullish, -1 bearish)."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=period
    ).to_numpy(dtype=float)
    n = len(close)
    direction = np.ones(n, dtype=int)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    for i in range(1, n):
        final_upper[i] = (
            basic_upper[i]
            if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]
            else final_upper[i - 1]
        )
        final_lower[i] = (
            basic_lower[i]
            if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]
            else final_lower[i - 1]
        )
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if close[i] > final_upper[i] else -1
    return pd.Series(direction, index=df.index)


def _ichimoku(
    df: pd.DataFrame, tenkan: int, kijun: int, senkou: int
) -> dict[str, pd.Series]:
    """Ichimoku components computed from rolling extrema."""
    high = df["high"]
    low = df["low"]
    tenkan_line = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_line = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = (tenkan_line + kijun_line) / 2
    senkou_b = (high.rolling(senkou).max() + low.rolling(senkou).min()) / 2
    return {
        "tenkan": tenkan_line,
        "kijun": kijun_line,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
    }


def _psar(
    df: pd.DataFrame, af0: float, max_af: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Parabolic SAR long/short levels and reversal flags."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(high)
    psar_l = np.full(n, np.nan)
    psar_s = np.full(n, np.nan)
    reversal = np.zeros(n, dtype=bool)
    bull = True
    af = af0
    ep = high[0]
    psar = low[0]
    psar_l[0] = psar
    for i in range(1, n):
        prev_psar = psar
        psar = prev_psar + af * (ep - prev_psar)
        if bull:
            psar = min(psar, low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < psar:
                bull = False
                reversal[i] = True
                psar = ep
                ep = low[i]
                af = af0
            elif high[i] > ep:
                ep = high[i]
                af = min(af + af0, max_af)
        else:
            psar = max(psar, high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > psar:
                bull = True
                reversal[i] = True
                psar = ep
                ep = high[i]
                af = af0
            elif low[i] < ep:
                ep = low[i]
                af = min(af + af0, max_af)
        if bull:
            psar_l[i] = psar
        else:
            psar_s[i] = psar
    idx = df.index
    return (
        pd.Series(psar_l, index=idx),
        pd.Series(psar_s, index=idx),
        pd.Series(reversal, index=idx),
    )


def _bbands(
    close: pd.Series, length: int, std: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger lower, upper, and middle bands."""
    bands = ta.volatility.BollingerBands(close, window=length, window_dev=std)
    return bands.bollinger_lband(), bands.bollinger_hband(), bands.bollinger_mavg()


def _keltner(
    df: pd.DataFrame, length: int, multiplier: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner lower, middle, and upper channels."""
    middle = ta.trend.ema_indicator(df["close"], window=length)
    atr = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=length
    )
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr
    return lower, middle, upper


def _crosses_above_level(series: pd.Series, level: float) -> pd.Series:
    """True when series crosses above a constant level."""
    return (series > level) & (series.shift(1) <= level)


def _crosses_below_level(series: pd.Series, level: float) -> pd.Series:
    """True when series crosses below a constant level."""
    return (series < level) & (series.shift(1) >= level)


def _col_starting(columns: pd.Index, prefix: str) -> str:
    """Return the first column name that starts with prefix."""
    matches = [c for c in columns if c.startswith(prefix)]
    if not matches:
        raise ValueError(f"No column starting with '{prefix}' in {list(columns)}")
    return matches[0]


def signal_ema_cross(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Dual EMA crossover trend signal.

    Args:
        df: OHLCV DataFrame with open, high, low, close columns.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        fast_period (int): Fast EMA length. Default 12.
        slow_period (int): Slow EMA length. Default 26.

    Long entry:
        Fast EMA crosses above slow EMA.

    Short entry:
        Fast EMA crosses below slow EMA.

    Exit:
        Opposite EMA cross closes the position.

    Calibration target:
        fast_period, slow_period.
    """
    fast_period = params.get("fast_period", 12)
    slow_period = params.get("slow_period", 26)
    ema_fast = _ema(df["close"], fast_period)
    ema_slow = _ema(df["close"], slow_period)
    long_entries = _crosses_above(ema_fast, ema_slow)
    long_exits = _crosses_below(ema_fast, ema_slow)
    short_entries = _crosses_below(ema_fast, ema_slow)
    short_exits = _crosses_above(ema_fast, ema_slow)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_ema_pullback(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Trend pullback to a faster EMA within a broader trend.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        ema_period (int): Pullback EMA length. Default 21.
        trend_period (int): Trend filter EMA length. Default 50.
        touch_atr_mult (float): ATR tolerance for EMA touch. Default 0.5.
        atr_period (int): ATR length. Default 14.

    Long entry:
        Price above trend EMA, low touches fast EMA, bullish close above fast EMA.

    Short entry:
        Price below trend EMA, high touches fast EMA, bearish close below fast EMA.

    Exit:
        Close crosses the fast EMA against the position.

    Calibration target:
        ema_period, trend_period, touch_atr_mult.
    """
    ema_period = params.get("ema_period", 21)
    trend_period = params.get("trend_period", 50)
    touch_atr_mult = params.get("touch_atr_mult", 0.5)
    atr_period = params.get("atr_period", 14)
    ema_fast = _ema(df["close"], ema_period)
    ema_trend = _ema(df["close"], trend_period)
    atr = _atr(df, atr_period)
    prev_close = df["close"].shift(1)
    uptrend = prev_close > ema_trend.shift(1)
    downtrend = prev_close < ema_trend.shift(1)
    touch_long = df["low"] <= (ema_fast + touch_atr_mult * atr)
    touch_short = df["high"] >= (ema_fast - touch_atr_mult * atr)
    bounce_long = (df["close"] > ema_fast) & (df["close"] > prev_close)
    bounce_short = (df["close"] < ema_fast) & (df["close"] < prev_close)
    long_entries = uptrend & touch_long & bounce_long
    short_entries = downtrend & touch_short & bounce_short
    long_exits = _crosses_below(df["close"], ema_fast)
    short_exits = _crosses_above(df["close"], ema_fast)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_ma_ribbon(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Stacked EMA ribbon alignment with price resumption.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        periods (list[int]): EMA periods fastest-to-slowest. Default [8, 13, 21, 34, 55].
        confirm_bars (int): Bars ribbon must stay aligned. Default 3.

    Long entry:
        Bullish ribbon confirmed and close crosses above fastest EMA.

    Short entry:
        Bearish ribbon confirmed and close crosses below fastest EMA.

    Exit:
        Ribbon alignment breaks (fast EMA crosses slow EMA).

    Calibration target:
        periods, confirm_bars.
    """
    periods = params.get("periods", [8, 13, 21, 34, 55])
    confirm_bars = params.get("confirm_bars", 3)
    emas = [_ema(df["close"], p) for p in periods]
    bullish = pd.Series(True, index=df.index)
    bearish = pd.Series(True, index=df.index)
    for i in range(len(emas) - 1):
        bullish = bullish & (emas[i] > emas[i + 1])
        bearish = bearish & (emas[i] < emas[i + 1])
    bull_confirmed = _consecutive_true(bullish, confirm_bars)
    bear_confirmed = _consecutive_true(bearish, confirm_bars)
    long_entries = bull_confirmed & _crosses_above(df["close"], emas[0])
    short_entries = bear_confirmed & _crosses_below(df["close"], emas[0])
    long_exits = _crosses_below(emas[0], emas[-1])
    short_exits = _crosses_above(emas[0], emas[-1])
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_adx_trend(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    ADX trend strength with directional index crossover.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        adx_period (int): ADX lookback. Default 14.
        adx_threshold (float): Minimum ADX for trend. Default 25.

    Long entry:
        ADX above threshold and +DI crosses above -DI.

    Short entry:
        ADX above threshold and -DI crosses above +DI.

    Exit:
        Opposite DI cross or ADX falls below threshold.

    Calibration target:
        adx_period, adx_threshold.
    """
    adx_period = params.get("adx_period", 14)
    adx_threshold = params.get("adx_threshold", 25)
    adx, dmp, dmn = _adx(df, adx_period)
    strong = adx > adx_threshold
    long_entries = strong & _crosses_above(dmp, dmn)
    short_entries = strong & _crosses_below(dmp, dmn)
    long_exits = _crosses_below(dmp, dmn) | (adx < adx_threshold)
    short_exits = _crosses_above(dmp, dmn) | (adx < adx_threshold)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_supertrend(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Supertrend direction flip signal.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        period (int): ATR period for Supertrend. Default 10.
        multiplier (float): ATR multiplier. Default 3.0.

    Long entry:
        Supertrend direction flips to bullish (+1).

    Short entry:
        Supertrend direction flips to bearish (-1).

    Exit:
        Opposite direction flip.

    Calibration target:
        period, multiplier.
    """
    period = params.get("period", 10)
    multiplier = params.get("multiplier", 3.0)
    direction = _supertrend(df, period, multiplier)
    prev_dir = direction.shift(1)
    long_entries = (direction == 1) & (prev_dir == -1)
    short_entries = (direction == -1) & (prev_dir == 1)
    long_exits = (direction == -1) & (prev_dir == 1)
    short_exits = (direction == 1) & (prev_dir == -1)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_hull_ma_cross(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Hull moving average crossover.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        fast_period (int): Fast HMA length. Default 9.
        slow_period (int): Slow HMA length. Default 21.

    Long entry:
        Fast HMA crosses above slow HMA.

    Short entry:
        Fast HMA crosses below slow HMA.

    Exit:
        Opposite HMA cross.

    Calibration target:
        fast_period, slow_period.
    """
    fast_period = params.get("fast_period", 9)
    slow_period = params.get("slow_period", 21)
    hma_fast = _hma(df["close"], fast_period)
    hma_slow = _hma(df["close"], slow_period)
    long_entries = _crosses_above(hma_fast, hma_slow)
    long_exits = _crosses_below(hma_fast, hma_slow)
    short_entries = _crosses_below(hma_fast, hma_slow)
    short_exits = _crosses_above(hma_fast, hma_slow)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_ichimoku_cloud_break(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Ichimoku cloud breakout (price vs Senkou span envelope).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        tenkan (int): Tenkan period. Default 9.
        kijun (int): Kijun period. Default 26.
        senkou (int): Senkou B period. Default 52.

    Long entry:
        Close crosses above the cloud top (max of Senkou A/B).

    Short entry:
        Close crosses below the cloud bottom (min of Senkou A/B).

    Exit:
        Opposite cloud-side cross.

    Calibration target:
        tenkan, kijun, senkou.
    """
    tenkan = params.get("tenkan", 9)
    kijun = params.get("kijun", 26)
    senkou = params.get("senkou", 52)
    ichi = _ichimoku(df, tenkan, kijun, senkou)
    isa = ichi["senkou_a"]
    isb = ichi["senkou_b"]
    cloud_top = isa.combine(isb, max)
    cloud_bottom = isa.combine(isb, min)
    long_entries = _crosses_above(df["close"], cloud_top)
    short_entries = _crosses_below(df["close"], cloud_bottom)
    long_exits = _crosses_below(df["close"], cloud_bottom)
    short_exits = _crosses_above(df["close"], cloud_top)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_ichimoku_tk_cross(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Ichimoku Tenkan/Kijun crossover.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        tenkan (int): Tenkan period. Default 9.
        kijun (int): Kijun period. Default 26.
        senkou (int): Senkou B period. Default 52.

    Long entry:
        Tenkan crosses above Kijun.

    Short entry:
        Tenkan crosses below Kijun.

    Exit:
        Opposite TK cross.

    Calibration target:
        tenkan, kijun.
    """
    tenkan_p = params.get("tenkan", 9)
    kijun_p = params.get("kijun", 26)
    senkou_p = params.get("senkou", 52)
    ichi = _ichimoku(df, tenkan_p, kijun_p, senkou_p)
    tenkan = ichi["tenkan"]
    kijun = ichi["kijun"]
    long_entries = _crosses_above(tenkan, kijun)
    long_exits = _crosses_below(tenkan, kijun)
    short_entries = _crosses_below(tenkan, kijun)
    short_exits = _crosses_above(tenkan, kijun)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_parabolic_sar_trend(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Parabolic SAR reversal trend signal.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        af0 (float): Initial acceleration factor. Default 0.02.
        af (float): Max acceleration factor. Default 0.2.

    Long entry:
        PSAR reversal to long (PSARr=1 with active PSARl).

    Short entry:
        PSAR reversal to short (PSARr=1 with active PSARs).

    Exit:
        Opposite PSAR reversal.

    Calibration target:
        af0, af.
    """
    af0 = params.get("af0", 0.02)
    af = params.get("af", 0.2)
    psar_long, psar_short, reversal = _psar(df, af0, af)
    long_active = psar_long.notna()
    short_active = psar_short.notna()
    long_entries = reversal & long_active
    short_entries = reversal & short_active
    long_exits = reversal & short_active
    short_exits = reversal & long_active
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_donchian_break(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Donchian channel breakout using prior-bar extremes.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        period (int): Donchian lookback. Default 20.

    Long entry:
        Close breaks above the prior N-bar high.

    Short entry:
        Close breaks below the prior N-bar low.

    Exit:
        Opposite channel break.

    Calibration target:
        period.
    """
    period = params.get("period", 20)
    upper = _rolling_high(df["high"], period)
    lower = _rolling_low(df["low"], period)
    long_entries = _crosses_above(df["close"], upper)
    short_entries = _crosses_below(df["close"], lower)
    long_exits = _crosses_below(df["close"], lower)
    short_exits = _crosses_above(df["close"], upper)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_ema_200_bounce(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Long-term EMA bounce continuation.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        ema_period (int): Slow EMA length. Default 200.
        tolerance_atr (float): ATR band for touch detection. Default 0.3.
        atr_period (int): ATR length. Default 14.

    Long entry:
        Uptrend above EMA, low tags EMA zone, bullish close above EMA.

    Short entry:
        Downtrend below EMA, high tags EMA zone, bearish close below EMA.

    Exit:
        Close crosses EMA against position.

    Calibration target:
        ema_period, tolerance_atr.
    """
    ema_period = params.get("ema_period", 200)
    tolerance_atr = params.get("tolerance_atr", 0.3)
    atr_period = params.get("atr_period", 14)
    ema = _ema(df["close"], ema_period)
    atr = _atr(df, atr_period)
    band = tolerance_atr * atr
    prev_close = df["close"].shift(1)
    uptrend = prev_close > ema.shift(1)
    downtrend = prev_close < ema.shift(1)
    touch_long = (df["low"] <= ema + band) & (df["low"] >= ema - band)
    touch_short = (df["high"] >= ema - band) & (df["high"] <= ema + band)
    bullish = df["close"] > df["open"]
    bearish = df["close"] < df["open"]
    long_entries = uptrend & touch_long & bullish & (df["close"] > ema)
    short_entries = downtrend & touch_short & bearish & (df["close"] < ema)
    long_exits = _crosses_below(df["close"], ema)
    short_exits = _crosses_above(df["close"], ema)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_hh_hl_continuation(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Higher-high / higher-low (or lower-low / lower-high) structure continuation.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        lookback (int): Swing comparison window. Default 5.

    Long entry:
        Recent swing high and low both exceed prior window (HH+HL) with bullish close.

    Short entry:
        Recent swing high and low both below prior window (LH+LL) with bearish close.

    Exit:
        Structure breaks (loss of HL for longs, loss of LH for shorts).

    Calibration target:
        lookback.
    """
    lookback = params.get("lookback", 5)
    recent_high = df["high"].shift(1).rolling(lookback).max()
    recent_low = df["low"].shift(1).rolling(lookback).min()
    prior_high = df["high"].shift(lookback + 1).rolling(lookback).max()
    prior_low = df["low"].shift(lookback + 1).rolling(lookback).min()
    hh = recent_high > prior_high
    hl = recent_low > prior_low
    lh = recent_high < prior_high
    ll = recent_low < prior_low
    long_entries = hh & hl & (df["close"] > df["close"].shift(1))
    short_entries = lh & ll & (df["close"] < df["close"].shift(1))
    long_exits = ~hl
    short_exits = ~lh
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_n_bar_break(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    N-bar high/low breakout.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        n_bars (int): Lookback for prior extreme. Default 20.

    Long entry:
        Close crosses above prior N-bar high.

    Short entry:
        Close crosses below prior N-bar low.

    Exit:
        Opposite N-bar break.

    Calibration target:
        n_bars.
    """
    n_bars = params.get("n_bars", 20)
    upper = _rolling_high(df["high"], n_bars)
    lower = _rolling_low(df["low"], n_bars)
    long_entries = _crosses_above(df["close"], upper)
    short_entries = _crosses_below(df["close"], lower)
    long_exits = _crosses_below(df["close"], lower)
    short_exits = _crosses_above(df["close"], upper)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_rsi_threshold(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    RSI threshold crossover mean-reversion entries.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        rsi_period (int): RSI length. Default 14.
        oversold (float): Long entry threshold. Default 30.
        overbought (float): Short entry threshold. Default 70.

    Long entry:
        RSI crosses above oversold.

    Short entry:
        RSI crosses below overbought.

    Exit:
        Long exits at overbought cross; short exits at oversold cross.

    Calibration target:
        rsi_period, oversold, overbought.
    """
    rsi_period = params.get("rsi_period", 14)
    oversold = params.get("oversold", 30)
    overbought = params.get("overbought", 70)
    rsi = ta.momentum.rsi(df["close"], window=rsi_period)
    long_entries = _crosses_above_level(rsi, oversold)
    short_entries = _crosses_below_level(rsi, overbought)
    long_exits = _crosses_above_level(rsi, overbought)
    short_exits = _crosses_below_level(rsi, oversold)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_rsi_momentum_break(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    RSI midline momentum breakout.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        rsi_period (int): RSI length. Default 14.
        midline (float): Momentum threshold. Default 50.

    Long entry:
        RSI crosses above midline.

    Short entry:
        RSI crosses below midline.

    Exit:
        Opposite midline cross.

    Calibration target:
        rsi_period, midline.
    """
    rsi_period = params.get("rsi_period", 14)
    midline = params.get("midline", 50)
    rsi = ta.momentum.rsi(df["close"], window=rsi_period)
    long_entries = _crosses_above_level(rsi, midline)
    long_exits = _crosses_below_level(rsi, midline)
    short_entries = _crosses_below_level(rsi, midline)
    short_exits = _crosses_above_level(rsi, midline)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_macd_cross(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    MACD line / signal line crossover.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        fast (int): MACD fast EMA. Default 12.
        slow (int): MACD slow EMA. Default 26.
        signal (int): Signal EMA. Default 9.

    Long entry:
        MACD crosses above signal line.

    Short entry:
        MACD crosses below signal line.

    Exit:
        Opposite MACD cross.

    Calibration target:
        fast, slow, signal.
    """
    fast = params.get("fast", 12)
    slow = params.get("slow", 26)
    signal_len = params.get("signal", 9)
    macd_line = ta.trend.macd(df["close"], window_slow=slow, window_fast=fast)
    signal_line = ta.trend.macd_signal(
        df["close"], window_slow=slow, window_fast=fast, window_sign=signal_len
    )
    long_entries = _crosses_above(macd_line, signal_line)
    long_exits = _crosses_below(macd_line, signal_line)
    short_entries = _crosses_below(macd_line, signal_line)
    short_exits = _crosses_above(macd_line, signal_line)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_cci_momentum(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    CCI zero-line momentum crossover.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        cci_period (int): CCI length. Default 20.

    Long entry:
        CCI crosses above zero.

    Short entry:
        CCI crosses below zero.

    Exit:
        Opposite zero-line cross.

    Calibration target:
        cci_period.
    """
    cci_period = params.get("cci_period", 20)
    cci = ta.trend.cci(df["high"], df["low"], df["close"], window=cci_period)
    long_entries = _crosses_above_level(cci, 0.0)
    long_exits = _crosses_below_level(cci, 0.0)
    short_entries = _crosses_below_level(cci, 0.0)
    short_exits = _crosses_above_level(cci, 0.0)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_cci_extreme(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    CCI extreme zone mean reversion.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        cci_period (int): CCI length. Default 20.
        upper (float): Overbought level. Default 100.
        lower (float): Oversold level. Default -100.

    Long entry:
        CCI crosses back above lower extreme.

    Short entry:
        CCI crosses back below upper extreme.

    Exit:
        CCI reaches opposite extreme.

    Calibration target:
        cci_period, upper, lower.
    """
    cci_period = params.get("cci_period", 20)
    upper = params.get("upper", 100)
    lower = params.get("lower", -100)
    cci = ta.trend.cci(df["high"], df["low"], df["close"], window=cci_period)
    long_entries = _crosses_above_level(cci, lower)
    short_entries = _crosses_below_level(cci, upper)
    long_exits = _crosses_above_level(cci, upper)
    short_exits = _crosses_below_level(cci, lower)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_williams_r(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Williams %R threshold crossover.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        period (int): Williams %R length. Default 14.
        oversold (float): Long entry level. Default -80.
        overbought (float): Short entry level. Default -20.

    Long entry:
        %R crosses above oversold (leaving oversold).

    Short entry:
        %R crosses below overbought (leaving overbought).

    Exit:
        Long exits at overbought cross; short exits at oversold cross.

    Calibration target:
        period, oversold, overbought.
    """
    period = params.get("period", 14)
    oversold = params.get("oversold", -80)
    overbought = params.get("overbought", -20)
    wr = ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp=period)
    long_entries = _crosses_above_level(wr, oversold)
    short_entries = _crosses_below_level(wr, overbought)
    long_exits = _crosses_above_level(wr, overbought)
    short_exits = _crosses_below_level(wr, oversold)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_roc_threshold(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Rate-of-change threshold crossover.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        roc_period (int): ROC length. Default 10.
        threshold (float): Momentum threshold. Default 0.0.

    Long entry:
        ROC crosses above threshold.

    Short entry:
        ROC crosses below negative threshold (or below threshold).

    Exit:
        Opposite threshold cross.

    Calibration target:
        roc_period, threshold.
    """
    roc_period = params.get("roc_period", 10)
    threshold = params.get("threshold", 0.0)
    roc = ((df["close"] - df["close"].shift(roc_period)) / df["close"].shift(roc_period)) * 100
    long_entries = _crosses_above_level(roc, threshold)
    long_exits = _crosses_below_level(roc, threshold)
    short_entries = _crosses_below_level(roc, -threshold if threshold != 0 else threshold)
    short_exits = _crosses_above_level(roc, -threshold if threshold != 0 else threshold)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_stochastic_cross(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Stochastic %K / %D crossover in extreme zones.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        k_period (int): %K length. Default 14.
        d_period (int): %D smoothing. Default 3.
        smooth_k (int): %K smoothing. Default 3.
        oversold (float): Long zone ceiling. Default 20.
        overbought (float): Short zone floor. Default 80.

    Long entry:
        %K crosses above %D while prior %K was oversold.

    Short entry:
        %K crosses below %D while prior %K was overbought.

    Exit:
        Opposite stochastic cross.

    Calibration target:
        k_period, d_period, oversold, overbought.
    """
    k_period = params.get("k_period", 14)
    d_period = params.get("d_period", 3)
    smooth_k = params.get("smooth_k", 3)
    oversold = params.get("oversold", 20)
    overbought = params.get("overbought", 80)
    from ta.momentum import StochasticOscillator

    stoch = StochasticOscillator(
        df["high"], df["low"], df["close"], window=k_period, smooth_window=smooth_k
    )
    k_line = stoch.stoch()
    d_line = k_line.rolling(d_period).mean()
    long_entries = _crosses_above(k_line, d_line) & (k_line.shift(1) < oversold)
    short_entries = _crosses_below(k_line, d_line) & (k_line.shift(1) > overbought)
    long_exits = _crosses_below(k_line, d_line)
    short_exits = _crosses_above(k_line, d_line)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_bollinger_reversion(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Bollinger Band mean reversion from extremes.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        length (int): BB period. Default 20.
        std (float): Standard deviation multiplier. Default 2.0.

    Long entry:
        Prior bar tagged lower band; close reclaims above lower band.

    Short entry:
        Prior bar tagged upper band; close falls back below upper band.

    Exit:
        Close reaches middle band.

    Calibration target:
        length, std.
    """
    length = params.get("length", 20)
    std = params.get("std", 2.0)
    lower, upper, mid = _bbands(df["close"], length, std)
    long_entries = (df["low"].shift(1) <= lower.shift(1)) & (df["close"] > lower)
    short_entries = (df["high"].shift(1) >= upper.shift(1)) & (df["close"] < upper)
    long_exits = (df["close"] >= mid) & (df["close"].shift(1) < mid.shift(1))
    short_exits = (df["close"] <= mid) & (df["close"].shift(1) > mid.shift(1))
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_bollinger_walk(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Bollinger Band walk (trend continuation along band).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        length (int): BB period. Default 20.
        std (float): Standard deviation multiplier. Default 2.0.

    Long entry:
        Close crosses above upper band (band walk start).

    Short entry:
        Close crosses below lower band.

    Exit:
        Close crosses back through middle band.

    Calibration target:
        length, std.
    """
    length = params.get("length", 20)
    std = params.get("std", 2.0)
    lower, upper, mid = _bbands(df["close"], length, std)
    long_entries = _crosses_above(df["close"], upper)
    short_entries = _crosses_below(df["close"], lower)
    long_exits = _crosses_below(df["close"], mid)
    short_exits = _crosses_above(df["close"], mid)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_squeeze_break(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    TTM Squeeze release breakout.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        bb_length (int): Bollinger length for squeeze. Default 20.
        bb_std (float): Bollinger std. Default 2.0.
        kc_length (int): Keltner length. Default 20.
        kc_scalar (float): Keltner scalar. Default 1.5.
        breakout_lookback (int): Prior high/low window. Default 20.

    Long entry:
        Squeeze releases and close breaks prior N-bar high.

    Short entry:
        Squeeze releases and close breaks prior N-bar low.

    Exit:
        Opposite breakout or new squeeze on.

    Calibration target:
        bb_length, kc_length, breakout_lookback.
    """
    bb_length = params.get("bb_length", 20)
    bb_std = params.get("bb_std", 2.0)
    kc_length = params.get("kc_length", 20)
    kc_scalar = params.get("kc_scalar", 1.5)
    breakout_lookback = params.get("breakout_lookback", 20)
    bb_lower, bb_upper, _ = _bbands(df["close"], bb_length, bb_std)
    kc_lower, _, kc_upper = _keltner(df, kc_length, kc_scalar)
    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    squeeze_on = squeeze_on.fillna(False).astype(bool)
    squeeze_off = (~squeeze_on).astype(bool)
    release = squeeze_on.shift(1).fillna(False) & squeeze_off
    upper = _rolling_high(df["high"], breakout_lookback)
    lower = _rolling_low(df["low"], breakout_lookback)
    long_entries = release & _crosses_above(df["close"], upper)
    short_entries = release & _crosses_below(df["close"], lower)
    long_exits = squeeze_on | _crosses_below(df["close"], lower)
    short_exits = squeeze_on | _crosses_above(df["close"], upper)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_atr_expansion(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    ATR expansion with directional range break.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        atr_period (int): ATR length. Default 14.
        atr_ma_period (int): ATR average window. Default 20.
        expansion_mult (float): Expansion factor vs ATR mean. Default 1.5.
        lookback (int): Breakout lookback. Default 20.

    Long entry:
        ATR exceeds expanded threshold and close breaks prior high.

    Short entry:
        ATR exceeds expanded threshold and close breaks prior low.

    Exit:
        ATR contracts below average or opposite break.

    Calibration target:
        atr_period, expansion_mult, lookback.
    """
    atr_period = params.get("atr_period", 14)
    atr_ma_period = params.get("atr_ma_period", 20)
    expansion_mult = params.get("expansion_mult", 1.5)
    lookback = params.get("lookback", 20)
    atr = _atr(df, atr_period)
    atr_ma = atr.shift(1).rolling(atr_ma_period).mean()
    expanded = atr > (atr_ma * expansion_mult)
    upper = _rolling_high(df["high"], lookback)
    lower = _rolling_low(df["low"], lookback)
    long_entries = expanded & _crosses_above(df["close"], upper)
    short_entries = expanded & _crosses_below(df["close"], lower)
    contracted = atr < atr_ma
    long_exits = contracted | _crosses_below(df["close"], lower)
    short_exits = contracted | _crosses_above(df["close"], upper)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_zscore_reversion(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Z-score mean reversion from extreme deviations.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        lookback (int): Rolling mean/std window. Default 20.
        z_entry (float): Entry z-score (long side). Default -2.0.
        z_exit (float): Exit z-score toward mean. Default 0.0.

    Long entry:
        Z-score crosses back above z_entry from below.

    Short entry:
        Z-score crosses back below -z_entry from above.

    Exit:
        Z-score reaches z_exit (mean reversion complete).

    Calibration target:
        lookback, z_entry, z_exit.
    """
    lookback = params.get("lookback", 20)
    z_entry = params.get("z_entry", -2.0)
    z_exit = params.get("z_exit", 0.0)
    prior_close = df["close"].shift(1)
    ma = prior_close.rolling(lookback).mean()
    std = prior_close.rolling(lookback).std()
    zscore = (df["close"] - ma) / std
    long_entries = _crosses_above_level(zscore, z_entry)
    short_entries = _crosses_below_level(zscore, -z_entry)
    long_exits = _crosses_above_level(zscore, z_exit)
    short_exits = _crosses_below_level(zscore, -z_exit)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_pin_bar(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Pin bar (hammer/shooting star) reversal pattern.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        wick_ratio (float): Minimum wick-to-body ratio. Default 2.0.
        body_max_ratio (float): Max body as fraction of range. Default 0.35.

    Long entry:
        Bullish pin bar (long lower wick, small body).

    Short entry:
        Bearish pin bar (long upper wick, small body).

    Exit:
        Opposite pin bar or range midpoint break.

    Calibration target:
        wick_ratio, body_max_ratio.
    """
    wick_ratio = params.get("wick_ratio", 2.0)
    body_max_ratio = params.get("body_max_ratio", 0.35)
    body = (df["close"] - df["open"]).abs()
    bar_range = df["high"] - df["low"]
    body_small = body <= (body_max_ratio * bar_range)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    bullish_pin = body_small & (lower_wick >= wick_ratio * body) & (df["close"] > df["open"])
    bearish_pin = body_small & (upper_wick >= wick_ratio * body) & (df["close"] < df["open"])
    midpoint = (df["high"] + df["low"]) / 2
    long_entries = bullish_pin
    short_entries = bearish_pin
    long_exits = bearish_pin | _crosses_below(df["close"], midpoint)
    short_exits = bullish_pin | _crosses_above(df["close"], midpoint)
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def signal_engulfing(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Bullish/bearish engulfing candle pattern.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        min_body_ratio (float): Min current body vs prior body. Default 1.0.

    Long entry:
        Bullish candle fully engulfs prior bearish body.

    Short entry:
        Bearish candle fully engulfs prior bullish body.

    Exit:
        Opposite engulfing pattern.

    Calibration target:
        min_body_ratio.
    """
    min_body_ratio = params.get("min_body_ratio", 1.0)
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    body = (df["close"] - df["open"]).abs()
    prev_body = (prev_close - prev_open).abs()
    body_ok = body >= (min_body_ratio * prev_body)
    bullish = (df["close"] > df["open"]) & (prev_close < prev_open)
    bearish = (df["close"] < df["open"]) & (prev_close > prev_open)
    engulf_long = (df["open"] <= prev_close) & (df["close"] >= prev_open)
    engulf_short = (df["open"] >= prev_close) & (df["close"] <= prev_open)
    long_entries = bullish & engulf_long & body_ok
    short_entries = bearish & engulf_short & body_ok
    long_exits = bearish & engulf_short
    short_exits = bullish & engulf_long
    return _make_output(df.index, long_entries, long_exits, short_entries, short_exits)


def _stub_signal(name: str, description: str, params_doc: str,
                 long_entry: str, short_entry: str, exit_doc: str,
                 calibration: str):
  """Factory not used at runtime — stubs defined explicitly below."""


def signal_atr_channel(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    ATR channel breakout (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        atr_period, channel_mult, lookback.

    Long entry:
        Close breaks above upper ATR channel.

    Short entry:
        Close breaks below lower ATR channel.

    Exit:
        Close returns inside channel.

    Calibration target:
        atr_period, channel_mult.
    """
    raise NotImplementedError("TODO: implement signal_atr_channel")


def signal_volatility_breakout(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    ATR expansion breakout with optional directional close filter.

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters (see below).

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        atr_period (int): ATR length. Default 14.
        expansion_mult (float): Expansion factor vs ATR mean. Default 1.5.
        directional_close (bool): Require bullish/bearish close. Default True.
        atr_ma_period (int): ATR average window. Default 20.

    Long entry:
        ATR expands beyond expansion_mult times its recent average with bullish close.

    Short entry:
        ATR expands beyond expansion_mult times its recent average with bearish close.

    Exit:
        ATR contracts back below its moving average.

    Calibration target:
        15-40 trades per pair per year on daily timeframe.
    """
    atr_period = params.get("atr_period", 14)
    expansion_mult = params.get("expansion_mult", 1.5)
    directional_close = params.get("directional_close", True)
    atr_ma_period = params.get("atr_ma_period", 20)
    atr = _atr(df, atr_period)
    atr_ma = atr.shift(1).rolling(atr_ma_period).mean()
    expanded = atr > (atr_ma * expansion_mult)
    if directional_close:
        long_entries = expanded & (df["close"] > df["close"].shift(1))
        short_entries = expanded & (df["close"] < df["close"].shift(1))
    else:
        long_entries = expanded & (df["high"] > df["high"].shift(1))
        short_entries = expanded & (df["low"] < df["low"].shift(1))
    contracted = atr < atr_ma
    return _make_output(df.index, long_entries, contracted, short_entries, contracted)


def signal_nr7_breakout(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    NR7 narrow-range breakout (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        nr_lookback, confirm_bars.

    Long entry:
        Break above high after NR7 bar.

    Short entry:
        Break below low after NR7 bar.

    Exit:
        Opposite range break.

    Calibration target:
        nr_lookback.
    """
    raise NotImplementedError("TODO: implement signal_nr7_breakout")


def signal_macd_histogram_divergence(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    MACD histogram divergence (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        fast, slow, signal, swing_lookback.

    Long entry:
        Bullish divergence: price lower low, histogram higher low.

    Short entry:
        Bearish divergence: price higher high, histogram lower high.

    Exit:
        Histogram zero-line cross.

    Calibration target:
        swing_lookback, fast, slow.
    """
    raise NotImplementedError("TODO: implement signal_macd_histogram_divergence")


def signal_rsi_divergence(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    RSI divergence (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        rsi_period, swing_lookback.

    Long entry:
        Bullish RSI divergence at swing low.

    Short entry:
        Bearish RSI divergence at swing high.

    Exit:
        RSI midline cross.

    Calibration target:
        rsi_period, swing_lookback.
    """
    raise NotImplementedError("TODO: implement signal_rsi_divergence")


def signal_stochastic_divergence(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Stochastic divergence (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        k_period, d_period, swing_lookback.

    Long entry:
        Bullish stochastic divergence.

    Short entry:
        Bearish stochastic divergence.

    Exit:
        Stochastic midline cross.

    Calibration target:
        k_period, swing_lookback.
    """
    raise NotImplementedError("TODO: implement signal_stochastic_divergence")


def signal_rsi_multi_timeframe(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Multi-timeframe RSI alignment (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        rsi_period, higher_tf_mult, oversold, overbought.

    Long entry:
        RSI oversold on both timeframes turning up.

    Short entry:
        RSI overbought on both timeframes turning down.

    Exit:
        RSI neutral on higher timeframe.

    Calibration target:
        rsi_period, higher_tf_mult.
    """
    raise NotImplementedError("TODO: implement signal_rsi_multi_timeframe")


def signal_volume_momentum(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Volume-confirmed momentum (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        vol_ma_period, momentum_period, vol_mult.

    Long entry:
        Up move with volume above average.

    Short entry:
        Down move with volume above average.

    Exit:
        Volume dries up or momentum fades.

    Calibration target:
        vol_ma_period, momentum_period.
    """
    raise NotImplementedError("TODO: implement signal_volume_momentum")


def signal_price_acceleration(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Price acceleration breakout (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        accel_period, threshold.

    Long entry:
        Positive price acceleration exceeds threshold.

    Short entry:
        Negative acceleration below threshold.

    Exit:
        Acceleration returns to zero.

    Calibration target:
        accel_period, threshold.
    """
    raise NotImplementedError("TODO: implement signal_price_acceleration")


def signal_linear_regression(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Linear regression channel signal (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        period, std_mult.

    Long entry:
        Price crosses above regression upper band.

    Short entry:
        Price crosses below regression lower band.

    Exit:
        Price returns to regression midline.

    Calibration target:
        period, std_mult.
    """
    raise NotImplementedError("TODO: implement signal_linear_regression")


def signal_key_level_retest(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Key level retest entry (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        level_lookback, retest_tolerance_atr.

    Long entry:
        Breakout above key level then successful retest.

    Short entry:
        Breakdown below key level then failed retest.

    Exit:
        Level violation.

    Calibration target:
        level_lookback, retest_tolerance_atr.
    """
    raise NotImplementedError("TODO: implement signal_key_level_retest")


def signal_weekly_range_break(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Prior weekly range breakout (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        confirm_bars.

    Long entry:
        Close breaks above prior week high.

    Short entry:
        Close breaks below prior week low.

    Exit:
        Return inside weekly range.

    Calibration target:
        confirm_bars.
    """
    raise NotImplementedError("TODO: implement signal_weekly_range_break")


def signal_asian_session_break(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Asian session range breakout (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        session_start_utc, session_end_utc.

    Long entry:
        Break above Asian session high.

    Short entry:
        Break below Asian session low.

    Exit:
        End of London session or range re-entry.

    Calibration target:
        session boundaries.
    """
    raise NotImplementedError("TODO: implement signal_asian_session_break")


def signal_pivot_breakout(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Classic pivot point breakout (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        pivot_type (standard/fib/camarilla).

    Long entry:
        Close breaks above R1.

    Short entry:
        Close breaks below S1.

    Exit:
        Return to pivot.

    Calibration target:
        pivot_type.
    """
    raise NotImplementedError("TODO: implement signal_pivot_breakout")


def signal_consolidation_break(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Consolidation box breakout (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        min_bars, max_range_atr.

    Long entry:
        Break above consolidation high.

    Short entry:
        Break below consolidation low.

    Exit:
        Re-entry into box.

    Calibration target:
        min_bars, max_range_atr.
    """
    raise NotImplementedError("TODO: implement signal_consolidation_break")


def signal_london_open_break(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    London open range breakout (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        open_hour_utc, range_minutes.

    Long entry:
        Break above London open range high.

    Short entry:
        Break below London open range low.

    Exit:
        NY session close or range re-entry.

    Calibration target:
        open_hour_utc, range_minutes.
    """
    raise NotImplementedError("TODO: implement signal_london_open_break")


def signal_false_breakout_fade(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    False breakout fade (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        lookback, fade_confirm_bars.

    Long entry:
        Failed breakdown — price reclaims range low.

    Short entry:
        Failed breakout — price falls back below range high.

    Exit:
        Opposite range extreme.

    Calibration target:
        lookback, fade_confirm_bars.
    """
    raise NotImplementedError("TODO: implement signal_false_breakout_fade")


def signal_extreme_zone_reversion(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Extreme zone mean reversion (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        indicator, upper, lower.

    Long entry:
        Reversal signal from statistically extreme zone.

    Short entry:
        Reversal from opposite extreme.

    Exit:
        Return to neutral zone.

    Calibration target:
        upper, lower thresholds.
    """
    raise NotImplementedError("TODO: implement signal_extreme_zone_reversion")


def signal_support_resistance_flip(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Support/resistance flip (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        swing_lookback, touch_count.

    Long entry:
        Resistance breaks and retests as support.

    Short entry:
        Support breaks and retests as resistance.

    Exit:
        Flip level violation.

    Calibration target:
        swing_lookback.
    """
    raise NotImplementedError("TODO: implement signal_support_resistance_flip")


def signal_fib_retracement(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Fibonacci retracement entry (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        swing_lookback, fib_level (0.382, 0.5, 0.618).

    Long entry:
        Bounce from fib support in uptrend.

    Short entry:
        Rejection at fib resistance in downtrend.

    Exit:
        Swing extreme or next fib level.

    Calibration target:
        fib_level, swing_lookback.
    """
    raise NotImplementedError("TODO: implement signal_fib_retracement")


def signal_mean_reversion_daily(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Daily mean reversion (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        ma_period, z_entry.

    Long entry:
        Close far below daily mean, reverting up.

    Short entry:
        Close far above daily mean, reverting down.

    Exit:
        Close reaches mean.

    Calibration target:
        ma_period, z_entry.
    """
    raise NotImplementedError("TODO: implement signal_mean_reversion_daily")


def signal_pivot_reversal(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Pivot point reversal (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        pivot_type, reversal_confirm.

    Long entry:
        Rejection at S1/S2 with bullish confirmation.

    Short entry:
        Rejection at R1/R2 with bearish confirmation.

    Exit:
        Pivot or next level reached.

    Calibration target:
        pivot_type.
    """
    raise NotImplementedError("TODO: implement signal_pivot_reversal")


def signal_choch(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Change of character — SMC (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        swing_lookback.

    Long entry:
        Bearish structure breaks to bullish CHoCH.

    Short entry:
        Bullish structure breaks to bearish CHoCH.

    Exit:
        Opposite CHoCH.

    Calibration target:
        swing_lookback.
    """
    raise NotImplementedError("TODO: implement signal_choch")


def signal_bos(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Break of structure — SMC (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        swing_lookback.

    Long entry:
        Bullish BOS — close breaks prior swing high.

    Short entry:
        Bearish BOS — close breaks prior swing low.

    Exit:
        Opposite BOS.

    Calibration target:
        swing_lookback.
    """
    raise NotImplementedError("TODO: implement signal_bos")


def signal_equal_hl_hunt(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Equal highs/lows liquidity hunt (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        tolerance_pips, swing_lookback.

    Long entry:
        Sweep of equal lows then reversal up.

    Short entry:
        Sweep of equal highs then reversal down.

    Exit:
        Opposite liquidity pool.

    Calibration target:
        tolerance_pips, swing_lookback.
    """
    raise NotImplementedError("TODO: implement signal_equal_hl_hunt")


def signal_fair_value_gap(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Fair value gap fill — SMC (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        min_gap_atr, max_age_bars.

    Long entry:
        Price enters bullish FVG and rejects upward.

    Short entry:
        Price enters bearish FVG and rejects downward.

    Exit:
        FVG fully filled.

    Calibration target:
        min_gap_atr.
    """
    raise NotImplementedError("TODO: implement signal_fair_value_gap")


def signal_liquidity_void(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Liquidity void fill (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        void_min_atr, lookback.

    Long entry:
        Downside void fill with bullish reaction.

    Short entry:
        Upside void fill with bearish reaction.

    Exit:
        Void fully closed.

    Calibration target:
        void_min_atr.
    """
    raise NotImplementedError("TODO: implement signal_liquidity_void")


def signal_supply_demand_zone(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Supply/demand zone reaction (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        zone_lookback, impulse_min_atr.

    Long entry:
        Price reacts bullishly at demand zone.

    Short entry:
        Price reacts bearishly at supply zone.

    Exit:
        Zone violation.

    Calibration target:
        zone_lookback, impulse_min_atr.
    """
    raise NotImplementedError("TODO: implement signal_supply_demand_zone")


def signal_premium_discount(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Premium/discount array entry — SMC (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        range_lookback, equilibrium_pct.

    Long entry:
        Price in discount array with bullish confirmation.

    Short entry:
        Price in premium array with bearish confirmation.

    Exit:
        Equilibrium reached.

    Calibration target:
        range_lookback.
    """
    raise NotImplementedError("TODO: implement signal_premium_discount")


def signal_swing_failure(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Swing failure pattern (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        swing_lookback.

    Long entry:
        Failed breakdown below swing low (SFP).

    Short entry:
        Failed breakout above swing high.

    Exit:
        Opposite swing extreme.

    Calibration target:
        swing_lookback.
    """
    raise NotImplementedError("TODO: implement signal_swing_failure")


def signal_correlation_divergence(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Cross-pair correlation divergence (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        reference_pair, corr_period, divergence_threshold.

    Long entry:
        Pair underperforms reference, mean-revert long.

    Short entry:
        Pair outperforms reference, mean-revert short.

    Exit:
        Correlation normalizes.

    Calibration target:
        corr_period, divergence_threshold.
    """
    raise NotImplementedError("TODO: implement signal_correlation_divergence")


def signal_momentum_factor(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Quant momentum factor (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        lookback, top_pct.

    Long entry:
        Strong positive momentum factor score.

    Short entry:
        Strong negative momentum factor score.

    Exit:
        Factor rank normalizes.

    Calibration target:
        lookback.
    """
    raise NotImplementedError("TODO: implement signal_momentum_factor")


def signal_mean_reversion_factor(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Quant mean reversion factor (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        lookback, z_entry.

    Long entry:
        Extreme negative factor z-score reverting.

    Short entry:
        Extreme positive factor z-score reverting.

    Exit:
        Z-score returns to zero.

    Calibration target:
        lookback, z_entry.
    """
    raise NotImplementedError("TODO: implement signal_mean_reversion_factor")


def signal_volatility_regime(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Volatility regime filter signal (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        vol_period, high_vol_threshold, low_vol_threshold.

    Long entry:
        Trend signal in expanding vol regime.

    Short entry:
        Trend signal in expanding vol regime (short).

    Exit:
        Regime shift to low volatility.

    Calibration target:
        vol_period, thresholds.
    """
    raise NotImplementedError("TODO: implement signal_volatility_regime")


def signal_vol_adjusted_momentum(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Volatility-adjusted momentum (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        momentum_period, atr_period, threshold.

    Long entry:
        ROC/ATR ratio crosses above threshold.

    Short entry:
        ROC/ATR ratio crosses below negative threshold.

    Exit:
        Ratio returns to zero.

    Calibration target:
        momentum_period, threshold.
    """
    raise NotImplementedError("TODO: implement signal_vol_adjusted_momentum")


def signal_currency_strength(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Currency strength momentum (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        strength_lookback, basket_pairs.

    Long entry:
        Base currency strengthening vs basket.

    Short entry:
        Base currency weakening vs basket.

    Exit:
        Strength momentum fades.

    Calibration target:
        strength_lookback.
    """
    raise NotImplementedError("TODO: implement signal_currency_strength")


def signal_seasonal(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Seasonal calendar pattern (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        month, day_of_month, direction.

    Long entry:
        Seasonal bullish window opens.

    Short entry:
        Seasonal bearish window opens.

    Exit:
        Seasonal window closes.

    Calibration target:
        calendar window dates.
    """
    raise NotImplementedError("TODO: implement signal_seasonal")


def signal_cross_pair_momentum(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Cross-pair relative momentum (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        reference_pair, momentum_period.

    Long entry:
        Pair outperforming reference momentum.

    Short entry:
        Pair underperforming reference momentum.

    Exit:
        Relative momentum neutralizes.

    Calibration target:
        momentum_period.
    """
    raise NotImplementedError("TODO: implement signal_cross_pair_momentum")


def signal_session_range_break(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Session range breakout (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        session_start_utc, session_end_utc.

    Long entry:
        Break above defined session high.

    Short entry:
        Break below defined session low.

    Exit:
        Session end or range re-entry.

    Calibration target:
        session boundaries.
    """
    raise NotImplementedError("TODO: implement signal_session_range_break")


def signal_eod_flow(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    End-of-day flow pattern (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        eod_hour_utc, flow_lookback.

    Long entry:
        Bullish EOD flow detected.

    Short entry:
        Bearish EOD flow detected.

    Exit:
        Next session open.

    Calibration target:
        eod_hour_utc.
    """
    raise NotImplementedError("TODO: implement signal_eod_flow")


def signal_nfp_setup(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    NFP release setup (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        pre_nfp_hours, post_nfp_hours.

    Long entry:
        Pre-NFP positioning or post-NFP momentum long.

    Short entry:
        Pre-NFP positioning or post-NFP momentum short.

    Exit:
        Post-release window close.

    Calibration target:
        pre/post NFP hours.
    """
    raise NotImplementedError("TODO: implement signal_nfp_setup")


def signal_cb_cycle(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Central bank cycle positioning (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        event_dates, pre_event_days, post_event_days.

    Long entry:
        CB hawkish/dovish cycle favors long.

    Short entry:
        CB cycle favors short.

    Exit:
        Event window close.

    Calibration target:
        pre/post event days.
    """
    raise NotImplementedError("TODO: implement signal_cb_cycle")


def signal_mtf_trend_alignment(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Multi-timeframe trend alignment (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        fast_period, slow_period, higher_tf_mult.

    Long entry:
        Trend aligned bullish on current and higher TF.

    Short entry:
        Trend aligned bearish on current and higher TF.

    Exit:
        Higher TF trend breaks.

    Calibration target:
        fast_period, slow_period, higher_tf_mult.
    """
    raise NotImplementedError("TODO: implement signal_mtf_trend_alignment")


def signal_star_pattern(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Morning/evening star candlestick pattern (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        body_ratio, gap_required.

    Long entry:
        Morning star pattern confirmed.

    Short entry:
        Evening star pattern confirmed.

    Exit:
        Opposite star pattern or midpoint break.

    Calibration target:
        body_ratio.
    """
    raise NotImplementedError("TODO: implement signal_star_pattern")


def signal_apex_native_placeholder(df: pd.DataFrame, params: dict) -> SignalOutput:
    """
    Placeholder for APEX-native strategies (stub).

    Args:
        df: OHLCV DataFrame.
        params: Strategy parameters.

    Returns:
        SignalOutput with long/short entry and exit booleans.

    Params:
        strategy_id: Reference to legacy APEX strategy.

    Long entry:
        Defined by referenced APEX-native rule set.

    Short entry:
        Defined by referenced APEX-native rule set.

    Exit:
        Defined by referenced APEX-native rule set.

    Calibration target:
        strategy_id.
    """
    raise NotImplementedError("TODO: implement signal_apex_native_placeholder")


SIGNAL_REGISTRY: dict[str, Callable[[pd.DataFrame, dict], SignalOutput]] = {
    "signal_ema_cross": signal_ema_cross,
    "signal_ema_pullback": signal_ema_pullback,
    "signal_ma_ribbon": signal_ma_ribbon,
    "signal_adx_trend": signal_adx_trend,
    "signal_supertrend": signal_supertrend,
    "signal_hull_ma_cross": signal_hull_ma_cross,
    "signal_ichimoku_cloud_break": signal_ichimoku_cloud_break,
    "signal_ichimoku_tk_cross": signal_ichimoku_tk_cross,
    "signal_parabolic_sar_trend": signal_parabolic_sar_trend,
    "signal_donchian_break": signal_donchian_break,
    "signal_ema_200_bounce": signal_ema_200_bounce,
    "signal_hh_hl_continuation": signal_hh_hl_continuation,
    "signal_n_bar_break": signal_n_bar_break,
    "signal_rsi_threshold": signal_rsi_threshold,
    "signal_rsi_momentum_break": signal_rsi_momentum_break,
    "signal_macd_cross": signal_macd_cross,
    "signal_cci_momentum": signal_cci_momentum,
    "signal_cci_extreme": signal_cci_extreme,
    "signal_williams_r": signal_williams_r,
    "signal_roc_threshold": signal_roc_threshold,
    "signal_stochastic_cross": signal_stochastic_cross,
    "signal_bollinger_reversion": signal_bollinger_reversion,
    "signal_bollinger_walk": signal_bollinger_walk,
    "signal_squeeze_break": signal_squeeze_break,
    "signal_atr_expansion": signal_atr_expansion,
    "signal_zscore_reversion": signal_zscore_reversion,
    "signal_pin_bar": signal_pin_bar,
    "signal_engulfing": signal_engulfing,
    "signal_atr_channel": signal_atr_channel,
    "signal_volatility_breakout": signal_volatility_breakout,
    "signal_nr7_breakout": signal_nr7_breakout,
    "signal_macd_histogram_divergence": signal_macd_histogram_divergence,
    "signal_rsi_divergence": signal_rsi_divergence,
    "signal_stochastic_divergence": signal_stochastic_divergence,
    "signal_rsi_multi_timeframe": signal_rsi_multi_timeframe,
    "signal_volume_momentum": signal_volume_momentum,
    "signal_price_acceleration": signal_price_acceleration,
    "signal_linear_regression": signal_linear_regression,
    "signal_key_level_retest": signal_key_level_retest,
    "signal_weekly_range_break": signal_weekly_range_break,
    "signal_asian_session_break": signal_asian_session_break,
    "signal_pivot_breakout": signal_pivot_breakout,
    "signal_consolidation_break": signal_consolidation_break,
    "signal_london_open_break": signal_london_open_break,
    "signal_false_breakout_fade": signal_false_breakout_fade,
    "signal_extreme_zone_reversion": signal_extreme_zone_reversion,
    "signal_support_resistance_flip": signal_support_resistance_flip,
    "signal_fib_retracement": signal_fib_retracement,
    "signal_mean_reversion_daily": signal_mean_reversion_daily,
    "signal_pivot_reversal": signal_pivot_reversal,
    "signal_choch": signal_choch,
    "signal_bos": signal_bos,
    "signal_equal_hl_hunt": signal_equal_hl_hunt,
    "signal_fair_value_gap": signal_fair_value_gap,
    "signal_liquidity_void": signal_liquidity_void,
    "signal_supply_demand_zone": signal_supply_demand_zone,
    "signal_premium_discount": signal_premium_discount,
    "signal_swing_failure": signal_swing_failure,
    "signal_correlation_divergence": signal_correlation_divergence,
    "signal_momentum_factor": signal_momentum_factor,
    "signal_mean_reversion_factor": signal_mean_reversion_factor,
    "signal_volatility_regime": signal_volatility_regime,
    "signal_vol_adjusted_momentum": signal_vol_adjusted_momentum,
    "signal_currency_strength": signal_currency_strength,
    "signal_seasonal": signal_seasonal,
    "signal_cross_pair_momentum": signal_cross_pair_momentum,
    "signal_session_range_break": signal_session_range_break,
    "signal_eod_flow": signal_eod_flow,
    "signal_nfp_setup": signal_nfp_setup,
    "signal_cb_cycle": signal_cb_cycle,
    "signal_mtf_trend_alignment": signal_mtf_trend_alignment,
    "signal_star_pattern": signal_star_pattern,
    "signal_apex_native_placeholder": signal_apex_native_placeholder,
}

if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet("data/cache/EURUSD_1d.parquet")
    from strategies.signals import signal_ema_cross, signal_supertrend, signal_rsi_threshold

    print(signal_ema_cross(df, {}))
    print(signal_supertrend(df, {}))
    print(signal_rsi_threshold(df, {}))
    print("All OK")
