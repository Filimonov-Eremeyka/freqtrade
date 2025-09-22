# SkoRZ1RStrategyV2LogFix.py
# Version: 2.2.0
# ИСПРАВЛЕНО:
# - Правильная логика определения направления СКО-свечи
# - Корректный расчет уровней SL/TP относительно СКО-бара
# - Улучшенное логирование с детальной диагностикой
# - Очистка состояния трейдов после закрытия

from typing import Dict, Any, Optional
import os
import logging
import pandas as pd
import numpy as np
from freqtrade.strategy.interface import IStrategy
from SkoRZIndicatorV1 import SkoRZIndicatorV1, SkoConfig


class SkoRZ1RStrategyV2LogFix(IStrategy):
    __version__ = "2.2.0"

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
    LOG_FILEPATH: str = "user_data/logs/sko_debug.log"

    _trade_state: Dict[int, Dict[str, Any]] = {}

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
            "sko_entry_long":  {"type": "scatter", "color": "green"},
            "sko_entry_short": {"type": "scatter", "color": "red"},
            "sko_stop_long":   {"type": "scatter", "color": "orange"},
            "sko_take_long":   {"type": "scatter", "color": "lime"},
            "sko_stop_short":  {"type": "scatter", "color": "orange"},
            "sko_take_short":  {"type": "scatter", "color": "pink"},
        },
        "subplots": {
            "SKO z-score": {
                "sko_zscore": {"type": "scatter", "color": "blue"}
            },
        },
    }

    def __init__(self, config: dict) -> None:
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
                self._log_any(f"[INIT] Strategy {self.__version__} initialized. File logging: {self.LOG_FILEPATH}")
                
            except Exception as e:
                self._file_logger = None
                print(f"Failed to setup file logger: {e}")

    # --------- helpers ----------
    def _log_any(self, msg: str) -> None:
        """Log to both file and strategy logger"""
        if self._file_logger:
            self._file_logger.info(msg)
        try:
            if hasattr(self, 'logger') and self.logger:
                self.logger.info(msg)
        except Exception:
            pass

    def _log_exit(self, msg: str) -> None:
        """Log exit-related messages"""
        if not self.DEBUG_EXITS:
            return
        self._log_any(msg)

    # --------- freqtrade interface ----------
    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Compute SKO indicators"""
        df = self._ind.compute(dataframe)
        
        # Добавим диагностические колонки
        if self.DEBUG_EXITS:
            # Подсчитаем количество валидных СКО
            sko_count = df["sko_is_valid"].sum()
            self._log_any(f"[INDICATORS] Total SKO candles found: {sko_count}")
            
        return df

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Mark entry signals based on SKO indicator"""
        df = dataframe.copy()
        
        # Установка сигналов входа
        long_signals = df["sko_entry_long"].notna()
        short_signals = df["sko_entry_short"].notna()
        
        df.loc[long_signals, "enter_long"] = 1
        df.loc[short_signals, "enter_short"] = 1
        
        # Логирование количества сигналов
        if self.DEBUG_EXITS:
            n_long = long_signals.sum()
            n_short = short_signals.sum()
            self._log_any(f"[ENTRY_TREND] Long signals: {n_long}, Short signals: {n_short}")
        
        return df

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Exit handled by custom_exit"""
        return dataframe

    # --------- levels calculation ----------
    def _compute_levels_from_sko(
        self,
        df: pd.DataFrame,
        entry_pos: int,
        trade,
    ) -> Optional[Dict[str, Any]]:
        """
        Вычисляем уровни SL/TP на основе СКО-свечи.
        entry_pos - позиция бара входа (где открылась сделка)
        СКО-бар = entry_pos - signal_lag_bars
        """
        signal_lag = int(self.cfg.signal_lag_bars)
        sko_pos = entry_pos - signal_lag
        
        if sko_pos < 0 or entry_pos >= len(df):
            self._log_exit(f"[LEVELS] Invalid positions: sko_pos={sko_pos}, entry_pos={entry_pos}, len={len(df)}")
            return None
            
        try:
            # Данные бара входа
            entry_row = df.iloc[entry_pos]
            entry_open = float(entry_row["open"])
            entry_time = entry_row.name
            
            # Данные СКО-бара
            sko_row = df.iloc[sko_pos]
            sko_high = float(sko_row["high"])
            sko_low = float(sko_row["low"])
            sko_close = float(sko_row["close"])
            sko_open = float(sko_row["open"])
            sko_time = sko_row.name
            
            # Определяем направление по СКО-свече
            sko_is_green = sko_close > sko_open
            
            # Проверяем, что СКО-свеча действительно валидная
            if not sko_row.get("sko_is_valid", False):
                self._log_exit(f"[LEVELS] Warning: SKO bar at {sko_pos} is not marked as valid!")
            
        except Exception as e:
            self._log_exit(f"[LEVELS] Error extracting data: {e}")
            return None
            
        # Диапазон СКО-свечи
        rng = sko_high - sko_low
        if rng <= 0:
            self._log_exit(f"[LEVELS] Invalid range: high={sko_high}, low={sko_low}")
            return None
            
        # Определяем направление трейда
        is_short_attr = getattr(trade, "is_short", None)
        if is_short_attr is not None:
            is_long = not trade.is_short
        else:
            is_long = trade.amount > 0
            
        # Расчет SL/TP
        if is_long:
            sl = sko_low
            tp = entry_open + rng
        else:
            sl = sko_high
            tp = entry_open - rng
            
        return {
            "entry_open": entry_open,
            "entry_time": entry_time,
            "entry_pos": entry_pos,
            "sko_high": sko_high,
            "sko_low": sko_low,
            "sko_close": sko_close,
            "sko_open": sko_open,
            "sko_time": sko_time,
            "sko_pos": sko_pos,
            "sko_is_green": sko_is_green,
            "rng": rng,
            "sl": sl,
            "tp": tp,
            "is_long": is_long,
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
        """
        Проверяем условия выхода по SL/TP на каждом тике.
        """
        df: pd.DataFrame = kwargs.get("dataframe", None)
        candle = kwargs.get("current_candle", None)
        
        if df is None or df.empty or candle is None:
            return None
            
        # Находим позицию бара входа
        entry_time = trade.open_date_utc
        try:
            # Используем get_indexer с method='pad' для поиска ближайшего времени
            entry_pos = df.index.get_indexer([entry_time], method="pad")[0]
            if entry_pos < 0:
                # Fallback на точный поиск
                entry_pos = df.index.get_loc(entry_time)
        except Exception as e:
            self._log_exit(f"[EXIT] Cannot find entry position for {entry_time}: {e}")
            return None
            
        # Получаем или вычисляем уровни
        trade_id = trade.trade_id
        
        if trade_id not in self._trade_state:
            # Первый тик этой сделки - вычисляем и сохраняем уровни
            levels = self._compute_levels_from_sko(df, entry_pos, trade)
            if not levels:
                self._log_exit(f"[EXIT] Failed to compute levels for trade #{trade_id}")
                return None
                
            self._trade_state[trade_id] = {
                "levels": levels,
                "tick_count": 0,
                "logged_passport": False,
            }
            
        state = self._trade_state[trade_id]
        levels = state["levels"]
        state["tick_count"] += 1
        
        # Логируем паспорт сделки (один раз)
        if not state["logged_passport"]:
            state["logged_passport"] = True
            direction = "LONG" if levels["is_long"] else "SHORT"
            sko_color = "GREEN" if levels["sko_is_green"] else "RED"
            
            self._log_exit(
                f"\n{'='*80}\n"
                f"[PASSPORT] Trade #{trade_id} | {pair} | {direction}\n"
                f"  Entry: time={levels['entry_time']} pos={levels['entry_pos']} open={levels['entry_open']:.2f}\n"
                f"  SKO: time={levels['sko_time']} pos={levels['sko_pos']} {sko_color} "
                f"O={levels['sko_open']:.2f} C={levels['sko_close']:.2f} "
                f"H={levels['sko_high']:.2f} L={levels['sko_low']:.2f}\n"
                f"  Range: R={levels['rng']:.2f} | SL={levels['sl']:.2f} TP={levels['tp']:.2f}\n"
                f"{'='*80}"
            )
            
        # Текущие данные свечи
        try:
            chigh = float(candle["high"])
            clow = float(candle["low"])
            cclose = float(candle.get("close", current_rate))
        except Exception as e:
            self._log_exit(f"[EXIT] Error reading candle data: {e}")
            return None
            
        sl = levels["sl"]
        tp = levels["tp"]
        
        # Логируем тик
        self._log_exit(
            f"[TICK #{state['tick_count']}] Trade #{trade_id} @ {current_time} | "
            f"Candle H={chigh:.2f} L={clow:.2f} C={cclose:.2f} | "
            f"Check SL={sl:.2f} TP={tp:.2f}"
        )
        
        # Проверка условий выхода
        exit_reason = None
        exit_price = None
        
        if levels["is_long"]:
            # LONG: выход по SL если low <= SL
            if clow <= sl:
                exit_reason = "stop_loss"
                exit_price = sl
                self._log_exit(
                    f"[EXIT] Trade #{trade_id} LONG -> STOP LOSS triggered! "
                    f"Low={clow:.2f} <= SL={sl:.2f}"
                )
            # LONG: выход по TP если high >= TP
            elif chigh >= tp:
                exit_reason = "take_profit"
                exit_price = tp
                self._log_exit(
                    f"[EXIT] Trade #{trade_id} LONG -> TAKE PROFIT triggered! "
                    f"High={chigh:.2f} >= TP={tp:.2f}"
                )
        else:
            # SHORT: выход по SL если high >= SL
            if chigh >= sl:
                exit_reason = "stop_loss"
                exit_price = sl
                self._log_exit(
                    f"[EXIT] Trade #{trade_id} SHORT -> STOP LOSS triggered! "
                    f"High={chigh:.2f} >= SL={sl:.2f}"
                )
            # SHORT: выход по TP если low <= TP
            elif clow <= tp:
                exit_reason = "take_profit"
                exit_price = tp
                self._log_exit(
                    f"[EXIT] Trade #{trade_id} SHORT -> TAKE PROFIT triggered! "
                    f"Low={clow:.2f} <= TP={tp:.2f}"
                )
                
        # Если сработал выход - очищаем состояние
        if exit_reason:
            if trade_id in self._trade_state:
                del self._trade_state[trade_id]
            self._log_exit(f"[EXIT] Trade #{trade_id} closed with {exit_reason} @ {exit_price:.2f}\n")
            
        return exit_reason