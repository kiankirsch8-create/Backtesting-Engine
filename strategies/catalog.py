from __future__ import annotations

from dataclasses import replace

from strategies.schema import Strategy, StrategyCatalog, StrategyStatus, StrategyType

# APEX Rule 1: M03 is LOCKED and excluded from this catalog (pair restrictions apply to M03 only).

_RECIPE_BOOST_PAIRS = ["CADJPY", "USDJPY", "CHFJPY"]

STRATEGIES: list[Strategy] = [
    # -------------------------------------------------------------------------
    # Tier A: NEW_CANDIDATE strategies
    # -------------------------------------------------------------------------
    # TREND — T11-T18
    Strategy(
        id="T11_SUPERTREND",
        name="SuperTrend ATR-based Trend Follower",
        strategy_type="TREND",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_supertrend",
        params={"atr_period": 10, "multiplier": 3.0, "adx_filter": 20.0},
        timeframes=["1d", "1wk"],
        description="ATR-based dynamic stop/reverse system. Long when SuperTrend flips bullish with ADX confirmation.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="T12_HULL_MA_CROSS",
        name="Hull Moving Average Cross",
        strategy_type="TREND",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_hull_ma_cross",
        params={"fast_period": 9, "slow_period": 21},
        timeframes=["1d", "1wk"],
        description="Hull MA reduces lag versus EMA. Cross of fast over slow generates entries.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="T13_ICHIMOKU_CLOUD_BREAK",
        name="Ichimoku Cloud Breakout",
        strategy_type="TREND",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_ichimoku_cloud_break",
        params={"tenkan": 9, "kijun": 26, "senkou": 52},
        timeframes=["1d", "1wk"],
        description="Price closing above/below the Ichimoku cloud with Chikou span confirmation.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="T14_PARABOLIC_SAR_TREND",
        name="Parabolic SAR with Trend Filter",
        strategy_type="TREND",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_parabolic_sar_trend",
        params={"af": 0.02, "max_af": 0.2, "trend_ema": 200},
        timeframes=["1d", "1wk"],
        description="Parabolic SAR flip in direction of the 200 EMA trend.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="T15_TRIPLE_EMA",
        name="Triple EMA Ribbon Alignment",
        strategy_type="TREND",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_ma_ribbon",
        params={"periods": [9, 21, 55], "confirm_bars": 2},
        timeframes=["1d", "1wk"],
        description="Three EMAs (9, 21, 55) all aligned in same direction for confirm_bars bars.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="T16_TK_CROSS",
        name="Ichimoku Tenkan-Kijun Cross",
        strategy_type="TREND",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_ichimoku_tk_cross",
        params={"tenkan": 9, "kijun": 26},
        timeframes=["1d", "1wk"],
        description="Tenkan crosses Kijun in direction of price relative to cloud.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="T17_LINEAR_REGRESSION_TREND",
        name="Linear Regression Slope Trend",
        strategy_type="TREND",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_linear_regression",
        params={"period": 50, "slope_threshold": 0.0001},
        timeframes=["1d", "1wk"],
        description="LR slope crosses threshold with price above LR midline.",
        source_rationale="QUANT_RESEARCH",
    ),
    Strategy(
        id="T18_HH_HL_TREND_COUNT",
        name="Higher High / Higher Low Trend Count",
        strategy_type="TREND",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_hh_hl_continuation",
        params={"lookback": 8, "min_hh_hl_count": 3},
        timeframes=["1d", "1wk"],
        description="3+ consecutive HH and HL swing structure with new HH triggering entry.",
        source_rationale="TECHNICAL",
    ),
    # MOMENTUM — M07-M12
    Strategy(
        id="M07_CCI_MOMENTUM",
        name="CCI Momentum Cross",
        strategy_type="MOMENTUM",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_cci_momentum",
        params={"cci_period": 20, "threshold": 100.0, "trend_ema": 50},
        timeframes=["1d", "1wk"],
        description="CCI crosses ±100 in direction of the 50 EMA trend.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="M08_DUAL_RSI_ALIGNMENT",
        name="Dual Timeframe RSI Alignment",
        strategy_type="MOMENTUM",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_rsi_multi_timeframe",
        params={"rsi_period": 14, "high_tf_threshold": 50.0, "low_tf_threshold": 50.0},
        timeframes=["1d"],
        description="Weekly RSI above 50 AND daily RSI crossing 50 from below for long entries.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="M09_ROC_THRESHOLD",
        name="Rate of Change Threshold",
        strategy_type="MOMENTUM",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_roc_threshold",
        params={"roc_period": 14, "threshold_pct": 1.0, "trend_ema": 50},
        timeframes=["1d", "1wk"],
        description="ROC(14) crosses ±1% with 50 EMA trend alignment.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="M10_STOCHASTIC_CROSS_TREND",
        name="Stochastic Cross with Trend Filter",
        strategy_type="MOMENTUM",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_stochastic_cross",
        params={"k_period": 14, "d_period": 3, "smooth_k": 3, "trend_filter": True},
        timeframes=["1d", "1wk"],
        description="%K/%D cross in oversold/overbought with 50 EMA trend filter.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="M11_MACD_HISTOGRAM_DIVERGENCE",
        name="MACD Histogram Divergence",
        strategy_type="MOMENTUM",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_macd_histogram_divergence",
        params={"fast": 12, "slow": 26, "signal": 9, "divergence_lookback": 10},
        timeframes=["1d", "1wk"],
        description="Price makes new extreme but MACD histogram fails to confirm (divergence).",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="M12_MOMENTUM_BURST",
        name="Short-Term Momentum Burst",
        strategy_type="MOMENTUM",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_roc_threshold",
        params={"roc_period": 5, "threshold_pct": 0.8, "trend_ema": 20},
        timeframes=["1d"],
        description="Shorter ROC(5) momentum surge with EMA(20) trend filter — fast signals.",
        source_rationale="TECHNICAL",
    ),
    # BREAKOUT — B11-B16
    Strategy(
        id="B11_ASIAN_SESSION_BREAK",
        name="Asian Session Range Breakout",
        strategy_type="BREAKOUT",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_asian_session_break",
        params={"session_start_utc": 0, "session_end_utc": 8, "min_range_atr_ratio": 0.5},
        timeframes=["1h"],
        description="Breakout of Asia session range (00:00–08:00 UTC) at London open with minimum range filter.",
        source_rationale="CALENDAR",
    ),
    Strategy(
        id="B12_WEEKLY_PIVOT_BREAK",
        name="Weekly Pivot Point Breakout",
        strategy_type="BREAKOUT",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_pivot_breakout",
        params={"pivot_timeframe": "1wk", "pivot_levels": ["R1", "S1"], "confirm_close": True},
        timeframes=["1d"],
        description="Price closing beyond weekly R1 or S1 pivot levels.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="B13_CONSOLIDATION_BREAK",
        name="N-Bar Consolidation Breakout",
        strategy_type="BREAKOUT",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_consolidation_break",
        params={"consolidation_bars": 15, "max_atr_ratio": 0.6, "min_breakout_pct": 0.3},
        timeframes=["1d", "1wk"],
        description="15-bar tight range (avg ATR < 60% of normal) followed by directional expansion.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="B14_LONDON_OPEN_BREAK",
        name="London Open Range Breakout",
        strategy_type="BREAKOUT",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_london_open_break",
        params={"london_open_utc": 8, "range_window_minutes": 30, "trend_ema": 50},
        timeframes=["1h"],
        description="Breakout of first 30-min London candle range, in direction of 50 EMA.",
        source_rationale="CALENDAR",
    ),
    Strategy(
        id="B15_N_BAR_HIGH_BREAK",
        name="20-Bar High/Low Breakout",
        strategy_type="BREAKOUT",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_n_bar_break",
        params={"period": 20, "confirm_close": True},
        timeframes=["1d", "1wk"],
        description="Close beyond 20-bar high/low. Shorter than T08 Donchian for more signals.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="B16_FALSE_BREAKOUT_FADE",
        name="Failed Breakout Reversal",
        strategy_type="BREAKOUT",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_false_breakout_fade",
        params={"breakout_lookback": 20, "reversal_bars": 2},
        timeframes=["1d", "1wk"],
        description="Breakout of key level fails and reverses within 2 bars — fade entry.",
        source_rationale="TECHNICAL",
    ),
    # REVERSION — R06, R07, R08, R11, R12
    Strategy(
        id="R06_PIVOT_POINT_REVERSAL",
        name="Monthly Pivot R2/S2 Reversal",
        strategy_type="REVERSION",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_pivot_reversal",
        params={"pivot_timeframe": "1mo", "pivot_levels": ["R2", "S2"], "exhaustion_rsi": 70.0},
        timeframes=["1d"],
        description="Price tagging monthly R2/S2 with RSI exhaustion (>70 or <30).",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="R07_CCI_EXTREME_REVERSAL",
        name="CCI Extreme ±200 Reversal",
        strategy_type="REVERSION",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_cci_extreme",
        params={"cci_period": 20, "extreme_threshold": 200.0, "return_threshold": 100.0},
        timeframes=["1d", "1wk"],
        description="CCI reaches ±200 then crosses back through ±100 — extreme reversal entry.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="R08_WILLIAMS_R_REVERSION",
        name="Williams %R Mean Reversion",
        strategy_type="REVERSION",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_williams_r",
        params={"period": 14, "oversold": -80.0, "overbought": -20.0},
        timeframes=["1d", "1wk"],
        description="Williams %R crosses out of oversold/overbought zones with reversal candle.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="R11_STOCHASTIC_DIVERGENCE",
        name="Stochastic Divergence Reversal",
        strategy_type="REVERSION",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_stochastic_divergence",
        params={"k_period": 14, "d_period": 3, "divergence_lookback": 10},
        timeframes=["1d", "1wk"],
        description="Price makes new extreme but stochastic diverges — reversal signal.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="R12_RSI_DIVERGENCE",
        name="Classic RSI Divergence Entry",
        strategy_type="REVERSION",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_rsi_divergence",
        params={"rsi_period": 14, "divergence_lookback": 10, "swing_break_confirm": True},
        timeframes=["1d", "1wk"],
        description="RSI divergence with break of recent swing as entry trigger.",
        source_rationale="TECHNICAL",
    ),
    # VOLATILITY — V04-V08
    Strategy(
        id="V04_BOLLINGER_WALK",
        name="Bollinger Band Trend Walk",
        strategy_type="VOLATILITY",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_bollinger_walk",
        params={"period": 20, "std": 2.0, "consecutive_touches": 2},
        timeframes=["1d", "1wk"],
        description="Price rides upper/lower band for 2+ consecutive bars with ADX confirmation.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="V05_ATR_CHANNEL_TREND",
        name="ATR Channel Trend Riding",
        strategy_type="VOLATILITY",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_atr_channel",
        params={"atr_period": 14, "atr_multiplier": 2.0, "slope_filter": True},
        timeframes=["1d", "1wk"],
        description="Price holds above/below 2x ATR channel with slope direction filter.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="V06_SQUEEZE_BREAK",
        name="Bollinger/Keltner Squeeze Release",
        strategy_type="VOLATILITY",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_squeeze_break",
        params={"bb_period": 20, "bb_std": 2.0, "kc_period": 20, "kc_mult": 1.5},
        timeframes=["1d", "1wk"],
        description="BB contracts inside KC (squeeze), then breakout direction is the trade.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="V07_VOLATILITY_BREAKOUT",
        name="ATR Expansion Breakout",
        strategy_type="VOLATILITY",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_volatility_breakout",
        params={"atr_period": 14, "expansion_mult": 1.5, "directional_close": True},
        timeframes=["1d", "1wk"],
        description="ATR expands beyond 1.5x recent average with directional close.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="V08_NR7_BREAK",
        name="Narrowest Range 7-Bar Breakout",
        strategy_type="VOLATILITY",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_nr7_breakout",
        params={"nr_window": 7, "breakout_atr_mult": 0.5},
        timeframes=["1d"],
        description="Narrowest range of last 7 bars followed by 0.5+ ATR breakout.",
        source_rationale="TECHNICAL",
    ),
    # SMC — SMC06-SMC09
    Strategy(
        id="SMC06_BOS_CONTINUATION",
        name="Break of Structure Continuation",
        strategy_type="SMC",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_bos",
        params={"swing_lookback": 10, "retest_bars": 5},
        timeframes=["1d", "1wk"],
        description="BOS confirmed with retest of broken level as entry.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="SMC07_SUPPLY_DEMAND_ZONE",
        name="Supply/Demand Zone Rejection",
        strategy_type="SMC",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_supply_demand_zone",
        params={"zone_lookback": 50, "zone_min_atr": 1.0, "rejection_candle_required": True},
        timeframes=["1d", "1wk"],
        description="Historical supply/demand zone retests with rejection candle entry.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="SMC08_PREMIUM_DISCOUNT",
        name="Premium/Discount Zone Entry",
        strategy_type="SMC",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_premium_discount",
        params={"swing_lookback": 30, "premium_fib": 0.618, "discount_fib": 0.382},
        timeframes=["1d", "1wk"],
        description="Sell at premium (>61.8% Fib), buy at discount (<38.2% Fib), trend-filtered.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="SMC09_SWING_FAILURE",
        name="Swing Failure Pattern",
        strategy_type="SMC",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_swing_failure",
        params={"swing_lookback": 10, "snap_back_bars": 2},
        timeframes=["1d", "1wk"],
        description="Price briefly breaks swing high/low then snaps back within 2 bars.",
        source_rationale="TECHNICAL",
    ),
    # QUANT — Q07-Q12
    Strategy(
        id="Q07_ZSCORE_REVERSION",
        name="Z-Score Mean Reversion",
        strategy_type="QUANT",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_zscore_reversion",
        params={"period": 20, "entry_zscore": 2.0, "exit_zscore": 0.0},
        timeframes=["1d", "1wk"],
        description="Price z-score beyond ±2 standard deviations from 20-day mean → reversion.",
        source_rationale="QUANT_RESEARCH",
    ),
    Strategy(
        id="Q08_VOL_ADJ_MOMENTUM",
        name="Volatility-Adjusted Momentum Factor",
        strategy_type="QUANT",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_vol_adjusted_momentum",
        params={"momentum_period": 20, "atr_period": 14, "threshold_ratio": 1.5},
        timeframes=["1d", "1wk"],
        description="Returns normalized by ATR; signal when ratio exceeds threshold.",
        source_rationale="QUANT_RESEARCH",
    ),
    Strategy(
        id="Q09_CURRENCY_STRENGTH",
        name="Relative Currency Strength Momentum",
        strategy_type="QUANT",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_currency_strength",
        params={"strength_period": 14, "rank_top_n": 3, "rank_bottom_n": 3},
        timeframes=["1d", "1wk"],
        description="Rank all 8 majors by relative strength; trade strongest vs weakest pairs.",
        source_rationale="QUANT_RESEARCH",
    ),
    Strategy(
        id="Q10_SEASONAL_TENDENCY",
        name="Forex Seasonal Pattern",
        strategy_type="QUANT",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_seasonal",
        params={"min_years_history": 5, "min_win_rate_historical": 0.6},
        timeframes=["1d"],
        description="Trade month-specific historical biases (e.g. September GBPUSD weakness).",
        source_rationale="QUANT_RESEARCH",
    ),
    Strategy(
        id="Q11_CROSS_PAIR_MOMENTUM",
        name="Cross-Pair Momentum Divergence",
        strategy_type="QUANT",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_cross_pair_momentum",
        params={"momentum_period": 14, "divergence_threshold_pct": 0.5},
        timeframes=["1d"],
        description="Same-base pairs (e.g. EURUSD, EURGBP) diverging in momentum → trade laggard.",
        source_rationale="QUANT_RESEARCH",
    ),
    Strategy(
        id="Q12_VOLATILITY_REGIME_FILTER",
        name="Volatility Regime Detection",
        strategy_type="QUANT",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_volatility_regime",
        params={"vol_period": 20, "high_vol_percentile": 0.8, "low_vol_percentile": 0.2},
        timeframes=["1d", "1wk"],
        description="Identifies high/low vol regimes; flags regime changes as setup signals.",
        source_rationale="QUANT_RESEARCH",
    ),
    # SESSION — S01-S04
    Strategy(
        id="S01_MONDAY_RANGE_BREAK",
        name="Monday Opening Range Breakout",
        strategy_type="SESSION",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_session_range_break",
        params={"day_of_week": 0, "range_window_hours": 4, "session_tz_utc": True},
        timeframes=["1h"],
        description="Breakout of first 4-hour Monday range. Complements Monday boost rule.",
        source_rationale="CALENDAR",
    ),
    Strategy(
        id="S02_EOD_FLOWS",
        name="End of Month Flow Direction",
        strategy_type="SESSION",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_eod_flow",
        params={"days_before_eom": 3, "trend_ema": 20},
        timeframes=["1d"],
        description="Last 3 days of month directional bias in line with 20 EMA.",
        source_rationale="CALENDAR",
        min_trades=20,
    ),
    Strategy(
        id="S03_NFP_WEEK_SETUP",
        name="NFP Week Pre-Friday Setup",
        strategy_type="SESSION",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_nfp_setup",
        params={"setup_days": ["Tuesday", "Wednesday", "Thursday"]},
        timeframes=["1d"],
        description="Tue-Thu directional setups in the week of NFP.",
        source_rationale="CALENDAR",
        allowed_pairs=["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"],
        min_trades=20,
    ),
    Strategy(
        id="S04_CB_CYCLE_POSITIONING",
        name="Central Bank Meeting Cycle",
        strategy_type="SESSION",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_cb_cycle",
        params={"pre_meeting_days": 5, "post_meeting_days": 5},
        timeframes=["1d"],
        description="5-day pre/post central bank meeting positioning patterns.",
        source_rationale="CALENDAR",
        min_trades=15,
    ),
    # MTF — X01-X03
    Strategy(
        id="X01_MTF_TREND_ALIGN",
        name="Weekly+Daily+4h Trend Alignment",
        strategy_type="MTF",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_mtf_trend_alignment",
        params={"timeframes_to_align": ["1wk", "1d", "1h"], "trend_ema": 50},
        timeframes=["1d"],
        description="All three timeframes aligned (above/below 50 EMA) → entry on lowest TF.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="X02_HTF_LEVEL_LTF_ENTRY",
        name="Higher Timeframe Level, Lower Timeframe Entry",
        strategy_type="MTF",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_mtf_trend_alignment",
        params={"timeframes_to_align": ["1d", "1h"], "trend_ema": 200, "ltf_signal": "rsi_cross"},
        timeframes=["1h"],
        description="Daily key level identified, 4h/1h RSI confirmation entry.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="X03_MTF_RSI_CONFLUENCE",
        name="Multi-Timeframe RSI Confluence",
        strategy_type="MTF",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_rsi_multi_timeframe",
        params={"rsi_period": 14, "high_tf_threshold": 50.0, "low_tf_threshold": 50.0},
        timeframes=["1d"],
        description="Weekly RSI and daily RSI both above/below 50 (confluence entry).",
        source_rationale="TECHNICAL",
    ),
    # PRICE ACTION — PA01-PA03
    Strategy(
        id="PA01_PIN_BAR_AT_LEVEL",
        name="Pin Bar at Key Level",
        strategy_type="PRICE_ACTION",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_pin_bar",
        params={"min_shadow_ratio": 2.0, "level_lookback": 20},
        timeframes=["1d", "1wk"],
        description="Pin bar with 2:1+ shadow ratio at significant rolling high/low.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="PA02_ENGULFING_AT_LEVEL",
        name="Engulfing Pattern at Key Level",
        strategy_type="PRICE_ACTION",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_engulfing",
        params={"level_lookback": 20, "min_body_ratio": 1.2},
        timeframes=["1d", "1wk"],
        description="Bullish/bearish engulfing at recent swing high/low.",
        source_rationale="TECHNICAL",
    ),
    Strategy(
        id="PA03_STAR_PATTERN",
        name="Morning/Evening Star Reversal",
        strategy_type="PRICE_ACTION",
        status="NEW_CANDIDATE",
        signal_fn_name="signal_star_pattern",
        params={"middle_candle_max_body_ratio": 0.3, "confirmation_atr_mult": 0.5},
        timeframes=["1d", "1wk"],
        description="3-candle morning/evening star pattern at extended move.",
        source_rationale="TECHNICAL",
    ),
    # -------------------------------------------------------------------------
    # Tier B: APEX TESTING strategies
    # -------------------------------------------------------------------------
    Strategy(
        id="T02_EMA_CROSSOVER",
        name="EMA Crossover",
        strategy_type="TREND",
        status="TESTING",
        signal_fn_name="signal_ema_cross",
        params={"fast_period": 9, "slow_period": 21},
        timeframes=["1d", "1wk"],
        description="APEX native EMA crossover.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="T05_HIGHER_TIMEFRAME_TREND",
        name="Higher Timeframe Trend",
        strategy_type="TREND",
        status="TESTING",
        signal_fn_name="signal_apex_native_placeholder",
        params={},
        timeframes=["1d"],
        description="APEX native — port from v5_data.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="T06_TREND_STRENGTH_FILTER",
        name="Trend Strength Filter",
        strategy_type="TREND",
        status="TESTING",
        signal_fn_name="signal_apex_native_placeholder",
        params={},
        timeframes=["1d"],
        description="APEX native — port from v5_data.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="T07_MA_RIBBON_ALIGNMENT",
        name="MA Ribbon Alignment",
        strategy_type="TREND",
        status="TESTING",
        signal_fn_name="signal_ma_ribbon",
        params={"periods": [5, 10, 20, 50], "confirm_bars": 2},
        timeframes=["1d"],
        description="APEX native MA ribbon.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="M02_MACD_ZERO_CROSS",
        name="MACD Zero Cross",
        strategy_type="MOMENTUM",
        status="TESTING",
        signal_fn_name="signal_macd_cross",
        params={"fast": 12, "slow": 26, "signal": 9, "zero_line_filter": True},
        timeframes=["1d", "1wk"],
        description="APEX native MACD zero cross. FRAGILE — runs only in good periods.",
        source_rationale="APEX_NATIVE",
        fragile=True,
    ),
    Strategy(
        id="M04_VOLUME_MOMENTUM",
        name="Volume Momentum",
        strategy_type="MOMENTUM",
        status="TESTING",
        signal_fn_name="signal_volume_momentum",
        params={},
        timeframes=["1d"],
        description="APEX native — port from v5_data.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="M05_STOCHASTIC_MOMENTUM",
        name="Stochastic Momentum",
        strategy_type="MOMENTUM",
        status="TESTING",
        signal_fn_name="signal_stochastic_cross",
        params={"k_period": 14, "d_period": 3, "smooth_k": 3, "trend_filter": True},
        timeframes=["1d"],
        description="APEX native stochastic momentum.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="B08_KEY_LEVEL_RETEST",
        name="Key Level Retest",
        strategy_type="BREAKOUT",
        status="TESTING",
        signal_fn_name="signal_key_level_retest",
        params={},
        timeframes=["1d", "1wk"],
        description="APEX native — port from v5_data.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="B10_WEEKLY_RANGE_BREAK",
        name="Weekly Range Breakout",
        strategy_type="BREAKOUT",
        status="TESTING",
        signal_fn_name="signal_weekly_range_break",
        params={},
        timeframes=["1d", "1wk"],
        description="APEX native — port from v5_data.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="R02_BOLLINGER_REVERSION",
        name="Bollinger Reversion",
        strategy_type="REVERSION",
        status="TESTING",
        signal_fn_name="signal_bollinger_reversion",
        params={"period": 20, "std": 2.0, "rsi_confirm": True, "rsi_period": 14},
        timeframes=["1d", "1wk"],
        description="APEX native Bollinger reversion.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="R03_SUPPORT_RESISTANCE_FLIP",
        name="Support/Resistance Flip",
        strategy_type="REVERSION",
        status="TESTING",
        signal_fn_name="signal_support_resistance_flip",
        params={},
        timeframes=["1d", "1wk"],
        description="APEX native — port from v5_data.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="R04_FIBONACCI_RETRACEMENT",
        name="Fibonacci Retracement",
        strategy_type="REVERSION",
        status="TESTING",
        signal_fn_name="signal_fib_retracement",
        params={},
        timeframes=["1d", "1wk"],
        description="APEX native — port from v5_data.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="R05_MEAN_REVERSION_DAILY",
        name="Daily Mean Reversion",
        strategy_type="REVERSION",
        status="TESTING",
        signal_fn_name="signal_mean_reversion_daily",
        params={},
        timeframes=["1d"],
        description="APEX native — port from v5_data.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="V01_VOLATILITY_BREAKOUT",
        name="Volatility Breakout",
        strategy_type="VOLATILITY",
        status="TESTING",
        signal_fn_name="signal_volatility_breakout",
        params={"atr_period": 14, "expansion_mult": 1.5, "directional_close": True},
        timeframes=["1d", "1wk"],
        description="APEX native volatility breakout.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="V03_SQUEEZE_MOMENTUM",
        name="Squeeze Momentum",
        strategy_type="VOLATILITY",
        status="TESTING",
        signal_fn_name="signal_squeeze_break",
        params={"bb_period": 20, "bb_std": 2.0, "kc_period": 20, "kc_mult": 1.5},
        timeframes=["1d", "1wk"],
        description="APEX native squeeze momentum.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="SMC03_FAIR_VALUE_GAP",
        name="Fair Value Gap",
        strategy_type="SMC",
        status="TESTING",
        signal_fn_name="signal_fair_value_gap",
        params={},
        timeframes=["1d", "1wk"],
        description="APEX native — port from v5_data.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="SMC04_LIQUIDITY_VOID",
        name="Liquidity Void",
        strategy_type="SMC",
        status="TESTING",
        signal_fn_name="signal_liquidity_void",
        params={},
        timeframes=["1d", "1wk"],
        description="APEX native — port from v5_data.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="Q01_CORRELATION_DIVERGENCE",
        name="Correlation Divergence",
        strategy_type="QUANT",
        status="TESTING",
        signal_fn_name="signal_correlation_divergence",
        params={},
        timeframes=["1d"],
        description="APEX native — port from v5_data.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="Q02_MOMENTUM_FACTOR",
        name="Momentum Factor",
        strategy_type="QUANT",
        status="TESTING",
        signal_fn_name="signal_momentum_factor",
        params={},
        timeframes=["1d", "1wk"],
        description="APEX native — port from v5_data.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="Q03_MEAN_REVERSION_FACTOR",
        name="Mean Reversion Factor",
        strategy_type="QUANT",
        status="TESTING",
        signal_fn_name="signal_mean_reversion_factor",
        params={},
        timeframes=["1d", "1wk"],
        description="APEX native — port from v5_data.",
        source_rationale="APEX_NATIVE",
    ),
    Strategy(
        id="Q05_VOLATILITY_REGIME",
        name="Volatility Regime",
        strategy_type="QUANT",
        status="TESTING",
        signal_fn_name="signal_volatility_regime",
        params={},
        timeframes=["1d", "1wk"],
        description="APEX native — port from v5_data.",
        source_rationale="APEX_NATIVE",
    ),
]

# Tier C: UNTESTED placeholder slots UNTESTED_001 through UNTESTED_029
for i in range(1, 30):
    STRATEGIES.append(
        Strategy(
            id=f"UNTESTED_{i:03d}",
            name=f"APEX UNTESTED Strategy Slot #{i}",
            strategy_type="QUANT",
            status="UNTESTED",
            signal_fn_name="signal_apex_native_placeholder",
            params={},
            timeframes=["1d"],
            description=(
                "Placeholder for APEX UNTESTED strategy — replace with actual strategy "
                "from strategies_v5_data.py."
            ),
            source_rationale="APEX_NATIVE",
        )
    )


def _apply_recipe_boost() -> None:
    """
    Apply APEX Rule 5 recipe boost pairs to trend strategies.

    Args:
        None.

    Returns:
        None. Mutates STRATEGIES entries in place via replace.
    """
    boost_ids = {"T11_SUPERTREND", "T15_TRIPLE_EMA", "T18_HH_HL_TREND_COUNT"}
    for idx, strategy in enumerate(STRATEGIES):
        if strategy.id in boost_ids:
            STRATEGIES[idx] = replace(strategy, recipe_boost_pairs=list(_RECIPE_BOOST_PAIRS))


_apply_recipe_boost()

CATALOG = StrategyCatalog(
    strategies=STRATEGIES,
    version="v1.0.0-2026-06-07",
)


def get_catalog() -> StrategyCatalog:
    """
    Return the global catalog.

    Args:
        None.

    Returns:
        Global StrategyCatalog instance.
    """
    return CATALOG


def get_strategy(strategy_id: str) -> Strategy:
    """
    Lookup helper.

    Args:
        strategy_id: Unique strategy identifier.

    Returns:
        Matching Strategy instance.
    """
    return CATALOG.get_by_id(strategy_id)


def list_testable() -> list[Strategy]:
    """
    All strategies that the engine will actually test.

    Args:
        None.

    Returns:
        Strategies with status TESTING, UNTESTED, or NEW_CANDIDATE.
    """
    testable_statuses: set[StrategyStatus] = {"TESTING", "UNTESTED", "NEW_CANDIDATE"}
    return [s for s in CATALOG if s.status in testable_statuses]


def list_by_type(strategy_type: str) -> list[Strategy]:
    """
    All strategies of a given type.

    Args:
        strategy_type: Strategy category name.

    Returns:
        Strategies matching the given type.
    """
    return [s for s in CATALOG if s.strategy_type == strategy_type]
