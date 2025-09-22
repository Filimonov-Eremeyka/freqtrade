# SkoRZ1RStrategyBlockFix.py
# Version: 0.2.0 (pending emulation, SCO+1 strict)
from typing import Any, Dict, Optional
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter, CategoricalParameter


class SkoRZ1RStrategyBlockFix(IStrategy):
    timeframe = '5m'
    can_short = True

    minimal_roi = {"0": 10}
    stoploss = -0.99
    trailing_stop = False
    process_only_new_candles = True

    # === параметры индикатора ===
    zscore_lookback = IntParameter(50, 300, default=100, space='indicator', optimize=False)
    zscore_threshold = DecimalParameter(1.0, 4.0, default=2.0, decimals=1, space='indicator', optimize=False)

    # Сигнальная свеча = СКО + s
    signal_lag_bars = IntParameter(1, 1, default=1, space='indicator', optimize=False)

    # RR и порог минимального размера СКО (в % от цены)
    RR_MULT   = DecimalParameter(1.0, 4.0, default=2.0, decimals=1, space='protection', optimize=False)
    MIN_R_PCT = DecimalParameter(0.05, 0.50, default=0.15, decimals=2, space='protection', optimize=False)

    # Эмуляция «отложки»: сколько свечей после сигнальной ждём касания триггера
    PENDING_BARS = IntParameter(1, 20, default=4, space='protection', optimize=False)

    # Что считаем сигналом (можно оба сразу)
    ENTRY_MODE = CategoricalParameter(["REVERSAL", "CONTINUATION", "BOTH"], default="BOTH",
                                      space='protection', optimize=False)

    DIAG_LOG = True

    # — визуализация —
    plot_config = {
        "main_plot": {
            # уровни «отложек» (горизонтали вперёд)
            "vis_trigger_long":  {"type": "scatter", "color": "orange"},
            "vis_trigger_short": {"type": "scatter", "color": "orange"},
            # стоп/тейк уровни (на участке ожидания)
            "vis_sl_long":       {"type": "scatter", "color": "blue"},
            "vis_tp_long":       {"type": "scatter", "color": "blue"},
            "vis_sl_short":      {"type": "scatter", "color": "purple"},
            "vis_tp_short":      {"type": "scatter", "color": "purple"},
            # маркеры фактического входа (когда «взяли отложку»)
            "mark_entry_long":   {"type": "scatter", "color": "green"},
            "mark_entry_short":  {"type": "scatter", "color": "red"},
        },
        "subplots": {
            "SKO z": { "sko_zscore": {"type": "scatter"} },
            "R%":    { "sko_Rpct_signal": {"type": "scatter"} },
        }
    }

    # ===== helpers =====
    @staticmethod
    def _series_color(close: pd.Series, open_: pd.Series) -> pd.Series:
        return (close > open_)

    @staticmethod
    def _to_utc_series(x: pd.Series | pd.Index) -> pd.DatetimeIndex:
        dt = pd.to_datetime(x, utc=True, errors='coerce')
        if not isinstance(dt, pd.DatetimeIndex):
            dt = pd.DatetimeIndex(dt)
        return dt

    def _locate_entry_pos(self, df: pd.DataFrame, entry_dt: datetime) -> Optional[int]:
        if entry_dt.tzinfo is None:
            entry_ts = pd.Timestamp(entry_dt).tz_localize('UTC')
        else:
            entry_ts = pd.Timestamp(entry_dt).tz_convert('UTC')

        if isinstance(df.index, pd.DatetimeIndex):
            dt_index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
        elif 'date' in df.columns:
            dt_index = self._to_utc_series(df['date'])
        else:
            return None

        pos = dt_index.get_indexer([entry_ts], method='nearest')[0]
        pos = 0 if pos < 0 else (len(df) - 1 if pos >= len(df) else pos)
        return int(pos)

    # ===== indicators + pending emulation =====
    def _compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        s  = int(self.signal_lag_bars.value)
        rr = float(self.RR_MULT.value)
        pend = int(self.PENDING_BARS.value)
        mode = str(self.ENTRY_MODE.value)

        # --- SKO по z-score объёма ---
        vol = df["volume"].astype("float64")
        lb  = int(self.zscore_lookback.value)
        mean = vol.rolling(lb, min_periods=lb).mean()
        std  = vol.rolling(lb, min_periods=lb).std(ddof=0)
        z = (vol - mean) / (std.replace(0, np.nan))
        df["sko_zscore"] = z

        thr = float(self.zscore_threshold.value)
        is_sko = (z > thr).fillna(False)

        # цвет свечи (зелёная/красная)
        green = self._series_color(df["close"], df["open"])
        # цвет СИГНАЛЬНОЙ свечи (= СКО + s) — сдвиг строго ВПЕРЁД
        shifted = green.shift(s)
        sig_green = shifted.where(shifted.notna(), False).astype(bool)

        # диапазон СКО, точка входа/триггер и фильтр по размеру
        R = (df["high"] - df["low"]).astype("float64")
        # цена на сигнальной свече (СКО+1): open для CONT, close для REV
        sig_open  = df["open"].shift(s).astype("float64")
        sig_close = df["close"].shift(s).astype("float64")

        # для графика и фильтра:
        entry_ref = sig_open.copy()  # просто чтобы Rpct считать стабильно
        df["sko_Rpct_signal"] = (R / entry_ref * 100.0)

        min_r_pct = float(self.MIN_R_PCT.value)
        big_enough = (df["sko_Rpct_signal"] >= min_r_pct) & entry_ref.notna()

        valid_base = is_sko & big_enough

        # --- триггеры по режимам ---
        want_rev  = (mode in ("REVERSAL", "BOTH"))
        want_cont = (mode in ("CONTINUATION", "BOTH"))

        # триггер-цена на СИГНАЛЬНОЙ свече:
        trig_long  = np.where(sig_green, (sig_close if want_rev else sig_open), np.nan)
        trig_short = np.where(~sig_green, (sig_close if want_rev else sig_open), np.nan)

        # «база»: есть СКО в t и мы рисуем сигналы на t+1
        base_mask = valid_base

        # сдвигаем на t+1 (там и лежит триггер)
        trig_long  = pd.Series(trig_long,  index=df.index).shift(+s)
        trig_short = pd.Series(trig_short, index=df.index).shift(+s)
        base_mask  = base_mask.shift(+s).fillna(False)

        # --- эмуляция «отложки»: берём ПЕРВУЮ свечу в окне [t+1 .. t+pend] где коснулись триггера ---
        n = len(df)
        enter_long  = pd.Series(False, index=df.index)
        enter_short = pd.Series(False, index=df.index)

        mark_entry_long  = pd.Series(np.nan, index=df.index, dtype="float64")
        mark_entry_short = pd.Series(np.nan, index=df.index, dtype="float64")

        # уровни SL/TP на участке ожидания (для визуализации)
        vis_trigger_long  = pd.Series(np.nan, index=df.index, dtype="float64")
        vis_trigger_short = pd.Series(np.nan, index=df.index, dtype="float64")
        vis_sl_long = pd.Series(np.nan, index=df.index, dtype="float64")
        vis_tp_long = pd.Series(np.nan, index=df.index, dtype="float64")
        vis_sl_short = pd.Series(np.nan, index=df.index, dtype="float64")
        vis_tp_short = pd.Series(np.nan, index=df.index, dtype="float64")

        highs = df["high"].to_numpy(dtype="float64")
        lows  = df["low"].to_numpy(dtype="float64")
        tlong  = trig_long.to_numpy(dtype="float64")
        tshort = trig_short.to_numpy(dtype="float64")
        base   = base_mask.to_numpy()

        sko_low  = df["low"].shift(+s - 1)   # low СКО-бара (t)
        sko_high = df["high"].shift(+s - 1)  # high СКО-бара (t)

        for i in range(n):
            if not base[i]:
                continue

            # окна ожидания: [i .. i+pend-1]
            j_to = min(i + pend, n)
            if np.isfinite(tlong[i]):
                trig = tlong[i]
                # проверяем первое касание для LONG: high >= trig
                hit_idx = None
                for j in range(i, j_to):
                    if highs[j] >= trig:
                        hit_idx = j
                        break
                # рисуем полосы ожидания
                vis_trigger_long.iloc[i:j_to] = trig
                if pd.notna(sko_low.iloc[i]) and pd.notna(sko_high.iloc[i]):
                    R_i = float(sko_high.iloc[i] - sko_low.iloc[i])
                    vis_sl_long.iloc[i:j_to] = float(sko_low.iloc[i])
                    vis_tp_long.iloc[i:j_to] = float(trig + R_i * float(self.RR_MULT.value))
                # фиксируем вход на первом касании
                if hit_idx is not None:
                    enter_long.iloc[hit_idx] = True
                    mark_entry_long.iloc[hit_idx] = highs[hit_idx]  # просто маркер на графике

            if np.isfinite(tshort[i]):
                trig = tshort[i]
                # проверяем первое касание для SHORT: low <= trig
                hit_idx = None
                for j in range(i, j_to):
                    if lows[j] <= trig:
                        hit_idx = j
                        break
                vis_trigger_short.iloc[i:j_to] = trig
                if pd.notna(sko_low.iloc[i]) and pd.notna(sko_high.iloc[i]):
                    R_i = float(sko_high.iloc[i] - sko_low.iloc[i])
                    vis_sl_short.iloc[i:j_to] = float(sko_high.iloc[i])
                    vis_tp_short.iloc[i:j_to] = float(trig - R_i * float(self.RR_MULT.value))
                if hit_idx is not None:
                    enter_short.iloc[hit_idx] = True
                    mark_entry_short.iloc[hit_idx] = lows[hit_idx]

        # сохраним в df то, что рисуем
        df["vis_trigger_long"]  = vis_trigger_long
        df["vis_trigger_short"] = vis_trigger_short
        df["vis_sl_long"]   = vis_sl_long
        df["vis_tp_long"]   = vis_tp_long
        df["vis_sl_short"]  = vis_sl_short
        df["vis_tp_short"]  = vis_tp_short
        df["mark_entry_long"]  = mark_entry_long
        df["mark_entry_short"] = mark_entry_short

        # эти флаги пойдут в populate_entry_trend
        df["sig_enter_long"]  = enter_long
        df["sig_enter_short"] = enter_short

        # сохраним также «по факту» уровни SL/TP для custom_exit (по цене триггера)
        # нам нужен R от СКО и реальная entry_price = триггер
        df["pending_trig_long"]  = trig_long
        df["pending_trig_short"] = trig_short
        df["sko_low_at_sig"]  = sko_low
        df["sko_high_at_sig"] = sko_high

        return df

    # ===== Freqtrade hooks =====
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return self._compute_all(dataframe.copy())

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe
        df.loc[:, "enter_long"]  = df["sig_enter_long"].fillna(False)
        df.loc[:, "enter_short"] = df["sig_enter_short"].fillna(False)
        df.loc[:, "enter_tag"] = np.where(df["enter_long"], "PENDING_LONG",
                                   np.where(df["enter_short"], "PENDING_SHORT", ""))
        return df

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0
        return dataframe

    # ===== custom_exit =====
    def custom_exit(self, pair: str, trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        except Exception:
            return None
        if df is None or df.empty:
            return None

        entry_dt = trade.open_date_utc
        if entry_dt is None:
            return None

        entry_pos = self._locate_entry_pos(df, entry_dt)
        if entry_pos is None:
            return None

        # восстановим уровни из ближайшей строки
        row = df.iloc[entry_pos]
        is_long = (trade.is_short is False)

        # entry мы считаем триггером (ему соответствует mark_entry_*), SL/TP — от СКО
        if is_long:
            trig = float(row.get("pending_trig_long", np.nan))
            sko_low  = float(row.get("sko_low_at_sig", np.nan))
            sko_high = float(row.get("sko_high_at_sig", np.nan))
            if np.isnan(trig) or np.isnan(sko_low) or np.isnan(sko_high):
                return None
            R = max(sko_high - sko_low, 0.0)
            sl = sko_low
            tp = trig + R * float(self.RR_MULT.value)
        else:
            trig = float(row.get("pending_trig_short", np.nan))
            sko_low  = float(row.get("sko_low_at_sig", np.nan))
            sko_high = float(row.get("sko_high_at_sig", np.nan))
            if np.isnan(trig) or np.isnan(sko_low) or np.isnan(sko_high):
                return None
            R = max(sko_high - sko_low, 0.0)
            sl = sko_high
            tp = trig - R * float(self.RR_MULT.value)

        # текущая свеча
        last = df.iloc[-1]
        ch, cl = float(last["high"]), float(last["low"])

        if is_long:
            if cl <= sl:  return "stop_loss"
            if ch >= tp:  return "take_profit"
        else:
            if ch >= sl:  return "stop_loss"
            if cl <= tp:  return "take_profit"
        return None
