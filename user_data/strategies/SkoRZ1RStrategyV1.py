# SkoRZ1RStrategyV1.py
# Version: 1.0.2
from typing import Dict, Any, Optional
import pandas as pd
from freqtrade.strategy.interface import IStrategy
from SkoRZIndicatorV1 import SkoRZIndicatorV1, SkoConfig

class SkoRZ1RStrategyV1(IStrategy):
    __version__ = "1.0.2"

    can_short: bool = True
    timeframe: str = "5m"
    process_only_new_candles: bool = True
    startup_candle_count: int = 100

    minimal_roi: Dict[str, float] = {"0": 100}
    use_custom_stoploss: bool = False
    stoploss: float = -1.0
    trailing_stop: bool = False

    cfg = SkoConfig(
        min_vol_ratio_prev=1.2,
        min_vol_ratio_next=1.3,
        zscore_lookback=100,
        zscore_threshold=2.0,
        use_log_volume=True,
        signal_lag_bars=1,
    )
    _ind = SkoRZIndicatorV1(cfg)

    plot_config = {
        "main_plot": {
            "sko_entry_long":  {"type": "scatter"},
            "sko_entry_short": {"type": "scatter"},
            "sko_stop_long":   {"type": "scatter"},
            "sko_take_long":   {"type": "scatter"},
            "sko_stop_short":  {"type": "scatter"},
            "sko_take_short":  {"type": "scatter"},
        },
        "subplots": {
            "Volume": { "volume": {"type": "bar"} },
            "SKO z":  { "sko_zscore": {"type": "scatter"} },
        },
    }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return self._ind.compute(dataframe)

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe.copy()
        df.loc[df["sko_entry_long"].notna(), "enter_long"] = 1
        df.loc[df["sko_entry_short"].notna(), "enter_short"] = 1
        return df

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: pd.Timestamp,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> Optional[str]:
        df: pd.DataFrame = kwargs.get("dataframe", None)
        candle = kwargs.get("current_candle", None)
        if df is None or df.empty or not candle:
            return None

        open_time = trade.open_date_utc
        try:
            row = df.loc[open_time]
        except Exception:
            try:
                pos = df.index.get_indexer([open_time], method="ffill")[0]
                row = df.iloc[pos]
            except Exception:
                return None

        # Определяем направление (универсально)
        is_short_attr = getattr(trade, "is_short", None)
        is_long = (trade.amount > 0) if is_short_attr is None else (not trade.is_short)

        sl = row.get("sko_stop_long" if is_long else "sko_stop_short")
        tp = row.get("sko_take_long" if is_long else "sko_take_short")
        if sl is None or tp is None or pd.isna(sl) or pd.isna(tp):
            return None

        chigh = float(candle["high"]); clow = float(candle["low"])
        if is_long:
            if clow <= float(sl): return "stop_loss"
            if chigh >= float(tp): return "take_profit"
        else:
            if chigh >= float(sl): return "stop_loss"
            if clow <= float(tp): return "take_profit"
        return None
