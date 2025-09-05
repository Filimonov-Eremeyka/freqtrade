import logging
from freqtrade.strategy import IStrategy
from pandas import DataFrame

from skd_enhanced import EnhancedSkdIndicator, SkdLevelVisualizer


class SkdVisualization(IStrategy):
    """
    Визуализация СКО под Freqtrade:
    - сигнал на t+1;
    - BUY — зелёный, SELL — красный;
    - отрезки длиной 5 свечей;
    - слоты с cooldown=1, отключение наложений:
        * expire_opposite_on_new=True (гасим противоположные активные уровни)
        * same_side_latest_only=True (оставляем только последний уровень той же стороны)
    - фильтры контрастности/дистанции по умолчанию выключены (показываем всё).
    """
    INTERFACE_VERSION = 3
    logger = logging.getLogger(__name__)

    timeframe = "5m"
    startup_candle_count = 100

    minimal_roi = {"0": 100}
    stoploss = -1.0
    process_only_new_candles = True

    # --- параметры индикатора ---
    skd_method: str = "basic"          # можно: "zscore", "percentile", "rvol", "stoch"
    skd_vol_min: float | None = None

    skd_percentile_lookback: int = 20
    skd_percentile: float = 70.0
    skd_rvol_lookback: int = 20
    skd_rvol_threshold: float = 1.5
    skd_zscore_lookback: int = 20
    skd_zscore_threshold: float = 2.0
    skd_stoch_lookback: int = 14
    skd_stoch_threshold: float = 80.0

    skd_signal_lag_bars: int = 1

    skd_min_distance_mode: str = "pct"
    skd_min_distance_pct: float = 0.0
    skd_atr_period: int = 14
    skd_min_distance_atr_mult: float = 0.0
    skd_filter_ttl_bars: int = 1

    # --- отрезки / визуализация ---
    skd_level_ttl: int = 5
    skd_max_slots: int = 4
    skd_slot_cooldown_bars: int = 1
    skd_expire_opposite_on_new: bool = True
    skd_same_side_latest_only: bool = True

    # цвета
    _BUY_COLOR = "#2ecc71"   # зелёный
    _SELL_COLOR = "#e74c3c"  # красный

    _BUY_LINE = {"mode": "lines", "line": {"color": _BUY_COLOR}}
    _SELL_LINE = {"mode": "lines", "line": {"color": _SELL_COLOR}}
    _BUY_MARKER = {"mode": "markers", "marker": {"symbol": "circle", "size": 8, "color": _BUY_COLOR}}
    _SELL_MARKER = {"mode": "markers", "marker": {"symbol": "circle", "size": 8, "color": _SELL_COLOR}}

    plot_config = {
        "main_plot": {
            "skd_buy_marker":  {"type": "scatter", "plotly": _BUY_MARKER},
            "skd_sell_marker": {"type": "scatter", "plotly": _SELL_MARKER},

            "skd_buy_signal_line_0":  {"type": "scatter", "plotly": _BUY_LINE},
            "skd_buy_signal_line_1":  {"type": "scatter", "plotly": _BUY_LINE},
            "skd_buy_signal_line_2":  {"type": "scatter", "plotly": _BUY_LINE},
            "skd_buy_signal_line_3":  {"type": "scatter", "plotly": _BUY_LINE},

            "skd_sell_signal_line_0": {"type": "scatter", "plotly": _SELL_LINE},
            "skd_sell_signal_line_1": {"type": "scatter", "plotly": _SELL_LINE},
            "skd_sell_signal_line_2": {"type": "scatter", "plotly": _SELL_LINE},
            "skd_sell_signal_line_3": {"type": "scatter", "plotly": _SELL_LINE},
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy().reset_index(drop=True)

        ind = EnhancedSkdIndicator(
            method=self.skd_method,
            vol_min=self.skd_vol_min,
            percentile_lookback=self.skd_percentile_lookback,
            percentile=self.skd_percentile,
            rvol_lookback=self.skd_rvol_lookback,
            rvol_threshold=self.skd_rvol_threshold,
            zscore_lookback=self.skd_zscore_lookback,
            zscore_threshold=self.skd_zscore_threshold,
            stoch_lookback=self.skd_stoch_lookback,
            stoch_threshold=self.skd_stoch_threshold,
            signal_lag_bars=self.skd_signal_lag_bars,
            min_distance_mode=self.skd_min_distance_mode,
            min_distance_pct=self.skd_min_distance_pct,
            atr_period=self.skd_atr_period,
            min_distance_atr_mult=self.skd_min_distance_atr_mult,
            filter_ttl_bars=self.skd_filter_ttl_bars,
        )
        df = ind.compute(df)

        vis = SkdLevelVisualizer(
            max_slots=self.skd_max_slots,
            level_ttl=self.skd_level_ttl,
            slot_cooldown_bars=self.skd_slot_cooldown_bars,
            expire_opposite_on_new=self.skd_expire_opposite_on_new,
            same_side_latest_only=self.skd_same_side_latest_only,
        )
        df = vis.render_levels(df)

        self.logger.info(
            f"[SKD] created={ind.stats.get('levels_created')} | "
            f"filtered={ind.stats.get('levels_filtered_distance')} | "
            f"total_skd={ind.stats.get('total_skd_found')}"
        )
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe
