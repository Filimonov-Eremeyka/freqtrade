# SkoRZ1RStrategyV2LogFix.py
# Version: 2.1.1
# ЧТО ИСПРАВЛЕНО:
# - Больше не обращаемся к self.logger внутри __init__ (в 2025.7 он может быть ещё не создан).
# - Отдельный логгер для файла self._file_logger. Пишем в него и в self.logger (когда он доступен).
# - Остальная логика та же: SL/TP считаются "на лету", подробные диагностики по каждой свече.

from typing import Dict, Any, Optional
import os
import logging
import pandas as pd
from freqtrade.strategy.interface import IStrategy
from SkoRZIndicatorV1 import SkoRZIndicatorV1, SkoConfig


class SkoRZ1RStrategyV2LogFix(IStrategy):
    __version__ = "2.1.1"

    can_short: bool = True
    timeframe: str = "5m"
    process_only_new_candles: bool = True
    startup_candle_count: int = 100

    minimal_roi: Dict[str, float] = {"0": 100}
    use_custom_stoploss: bool = False
    stoploss: float = -1.0
    trailing_stop: bool = False

    DEBUG_EXITS: bool = True
    LOG_TO_FILE: bool = True
    LOG_FILEPATH: str = "user_data/logs/sko_v2_debug.log"

    _trade_state: Dict[int, bool] = {}

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

    def __init__(self, config: dict) -> None:
        # Не трогаем self.logger здесь — он может ещё не быть инициализирован.
        super().__init__(config)
        self._file_logger: Optional[logging.Logger] = None

        if self.LOG_TO_FILE:
            try:
                os.makedirs(os.path.dirname(self.LOG_FILEPATH), exist_ok=True)
                lg = logging.getLogger(self.__class__.__name__ + ".file")
                lg.setLevel(logging.INFO)
                # Проверим, не добавлен ли уже наш хендлер
                has_file = False
                for h in lg.handlers:
                    if isinstance(h, logging.FileHandler) and getattr(h, "_sko_file", False):
                        has_file = True
                        break
                if not has_file:
                    fh = logging.FileHandler(self.LOG_FILEPATH, mode="a", encoding="utf-8")
                    fh.setLevel(logging.INFO)
                    fmt = logging.Formatter(
                        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                    fh.setFormatter(fmt)
                    setattr(fh, "_sko_file", True)
                    lg.addHandler(fh)
                self._file_logger = lg
                self._log_any(f"[init] File logging enabled: {self.LOG_FILEPATH}")
            except Exception as e:
                # Если файл-логгер не поднялся — не критично
                self._file_logger = None

    # --------- helpers ----------
    def _log_any(self, msg: str) -> None:
        if self._file_logger:
            self._file_logger.info(msg)
        # self.logger обычно уже есть после инициализации стратегии в рантайме
        try:
            self.logger.info(msg)
        except Exception:
            pass

    def _log_exit(self, msg: str) -> None:
        if not self.DEBUG_EXITS:
            return
        self._log_any(msg)

    # --------- freqtrade iface ----------
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

    # --------- levels from SKO ----------
    def _compute_levels_from_sko(
        self,
        df: pd.DataFrame,
        entry_pos: int,
        is_long: bool,
        signal_lag_bars: int,
    ) -> Optional[Dict[str, float]]:
        sk_pos = entry_pos - int(signal_lag_bars)
        if sk_pos < 0 or entry_pos >= len(df):
            return None
        try:
            entry_open = float(df.iloc[entry_pos]["open"])
            sk_high = float(df.iloc[sk_pos]["high"])
            sk_low  = float(df.iloc[sk_pos]["low"])
        except Exception:
            return None
        rng = sk_high - sk_low
        if rng <= 0:
            return None
        if is_long:
            sl = sk_low
            tp = entry_open + rng
        else:
            sl = sk_high
            tp = entry_open - rng
        return {
            "entry_open": entry_open, "sk_high": sk_high, "sk_low": sk_low,
            "rng": rng, "sl": sl, "tp": tp, "sk_pos": sk_pos
        }

    # --------- custom exit ----------
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

        entry_time = trade.open_date_utc
        try:
            entry_pos = df.index.get_indexer([entry_time], method="pad")[0]
        except Exception:
            return None

        is_short_attr = getattr(trade, "is_short", None)
        is_long = (trade.amount > 0) if is_short_attr is None else (not trade.is_short)

        levels = self._compute_levels_from_sko(df, entry_pos, is_long, self.cfg.signal_lag_bars)
        if not levels:
            return None

        entry_open = levels["entry_open"]
        sl = levels["sl"]; tp = levels["tp"]
        sk_hi = levels["sk_high"]; sk_lo = levels["sk_low"]
        rng = levels["rng"]; sk_pos = levels["sk_pos"]

        chigh = float(candle["high"]); clow  = float(candle["low"])

        # Паспорт сделки (логим один раз)
        if self.DEBUG_EXITS and not self._trade_state.get(trade.trade_id, False):
            self._trade_state[trade.trade_id] = True
            self._log_exit(
                f"[{pair}] trade#{trade.trade_id} "
                f"{'LONG' if is_long else 'SHORT'} | entry_time={entry_time} pos={entry_pos} "
                f"entry_open={entry_open:.2f} | SKO[pos={sk_pos}] hi/lo={sk_hi:.2f}/{sk_lo:.2f} "
                f"rng={rng:.2f} -> SL={sl:.2f} TP={tp:.2f}"
            )

        # Покадровый лог
        self._log_exit(
            f"[{pair}] trade#{trade.trade_id} tick @{current_time} H/L={chigh:.2f}/{clow:.2f} "
            f"check SL={sl:.2f} TP={tp:.2f}"
        )

        # Проверка касания
        if is_long:
            if clow <= sl:
                self._log_exit(f"[{pair}] trade#{trade.trade_id} LONG -> STOP @ {sl:.2f}")
                return "stop_loss"
            if chigh >= tp:
                self._log_exit(f"[{pair}] trade#{trade.trade_id} LONG -> TAKE @ {tp:.2f}")
                return "take_profit"
        else:
            if chigh >= sl:
                self._log_exit(f"[{pair}] trade#{trade.trade_id} SHORT -> STOP @ {sl:.2f}")
                return "stop_loss"
            if clow <= tp:
                self._log_exit(f"[{pair}] trade#{trade.trade_id} SHORT -> TAKE @ {tp:.2f}")
                return "take_profit"

        return None
