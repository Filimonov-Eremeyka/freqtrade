# SkoRZ1RStrategyV2.py
# Version: 2.0.0
# Изменения:
# - SL/TP пересчитываются в custom_exit "на лету" по свечe SKO (бар t = entry-1)
# - Убран дублирующий подграфик Volume
# - Добавлен диагностический лог по входам/выходам (DEBUG_EXITS)

from typing import Dict, Any, Optional
import pandas as pd

from freqtrade.strategy.interface import IStrategy
from freqtrade.exchange import timeframe_to_minutes

from SkoRZIndicatorV1 import SkoRZIndicatorV1, SkoConfig


class SkoRZ1RStrategyV2(IStrategy):
    __version__ = "2.0.0"

    can_short: bool = True
    timeframe: str = "5m"
    process_only_new_candles: bool = True
    startup_candle_count: int = 100

    # отключаем все внешние выходы, управляем custom_exit'ом
    minimal_roi: Dict[str, float] = {"0": 100}
    use_custom_stoploss: bool = False
    stoploss: float = -1.0
    trailing_stop: bool = False

    # отладка
    DEBUG_EXITS: bool = True   # выключить при необходимости

    # конфигурация индикатора: ratio + zscore
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
            "SKO z":  { "sko_zscore": {"type": "scatter"} },
        },
    }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return self._ind.compute(dataframe)

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe.copy()
        # Индикатор уже формирует сигналы на свече t+1 (entry-бар) — просто ставим флажки
        df.loc[df["sko_entry_long"].notna(), "enter_long"] = 1
        df.loc[df["sko_entry_short"].notna(), "enter_short"] = 1
        return df

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # все выходы — через custom_exit
        return dataframe

    # ---------------------------
    #  CUSTOM EXIT (SL/TP on-the-fly)
    # ---------------------------
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

        # 1) индекс бара входа (open_date_utc совпадает с индексом ряда)
        entry_time = trade.open_date_utc
        try:
            entry_pos = df.index.get_indexer([entry_time], method="pad")[0]
        except Exception:
            return None

        # 2) считаем, что свеча СКО = бар (entry_pos - signal_lag_bars)
        s = int(self.cfg.signal_lag_bars)
        sk_pos = entry_pos - s
        if sk_pos < 0:
            return None

        # 3) извлекаем цены: entry на open(entry_pos), диапазон = high-low на SKO-свече
        try:
            entry_open = float(df.iloc[entry_pos]["open"])
            sk_high = float(df.iloc[sk_pos]["high"])
            sk_low  = float(df.iloc[sk_pos]["low"])
        except Exception:
            return None

        rng = sk_high - sk_low
        if rng <= 0:
            return None

        # 4) определяем направление сделки
        #    (новые версии имеют trade.is_short; старые различают по sign(amount))
        is_short_attr = getattr(trade, "is_short", None)
        is_long = (trade.amount > 0) if is_short_attr is None else (not trade.is_short)

        # 5) рассчитываем уровни SL/TP в момент входа
        if is_long:
            sl = sk_low
            tp = entry_open + rng
        else:
            sl = sk_high
            tp = entry_open - rng

        # 6) проверяем касание на текущей свече
        chigh = float(candle["high"])
        clow  = float(candle["low"])

        # Диагностика (один лог при первом заходе в сделку и при выходе)
        if self.DEBUG_EXITS and candle.get("_logged", False) is False:
            # Помечаем, чтобы не спамить на каждой свече
            candle["_logged"] = True
            self.logger.info(
                f"[{pair}] trade#{trade.trade_id} {'LONG' if is_long else 'SHORT'} "
                f"entry@{entry_open:.2f} SKO[hi/lo]={sk_high:.2f}/{sk_low:.2f} "
                f"rng={rng:.2f} -> SL={sl:.2f} TP={tp:.2f} "
                f"| now [{current_time}] H/L={chigh:.2f}/{clow:.2f}"
            )

        if is_long:
            if clow <= sl:
                if self.DEBUG_EXITS:
                    self.logger.info(f"[{pair}] trade#{trade.trade_id} LONG -> STOP @ {sl:.2f}")
                return "stop_loss"
            if chigh >= tp:
                if self.DEBUG_EXITS:
                    self.logger.info(f"[{pair}] trade#{trade.trade_id} LONG -> TAKE @ {tp:.2f}")
                return "take_profit"
        else:
            if chigh >= sl:
                if self.DEBUG_EXITS:
                    self.logger.info(f"[{pair}] trade#{trade.trade_id} SHORT -> STOP @ {sl:.2f}")
                return "stop_loss"
            if clow <= tp:
                if self.DEBUG_EXITS:
                    self.logger.info(f"[{pair}] trade#{trade.trade_id} SHORT -> TAKE @ {tp:.2f}")
                return "take_profit"

        return None
