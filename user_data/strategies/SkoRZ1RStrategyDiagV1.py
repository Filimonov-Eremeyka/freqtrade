# user_data/strategies/SkoRZ1RStrategyDiagV1.py
# Version: 1.0.0 (diag)
# Назначение: параллельно с торговлей сохраняет CSV всех СКО-сигналов с расчётом SL/TP.
# Если DIAG_ONLY = True, то сделки не открывает — только CSV.

from typing import Dict, Any, Optional, List
import os
import csv
import pandas as pd
from freqtrade.strategy.interface import IStrategy
from SkoRZIndicatorV1 import SkoRZIndicatorV1, SkoConfig

class SkoRZ1RStrategyDiagV1(IStrategy):
    __version__ = "1.0.0"

    # === базовые настройки
    can_short: bool = True
    timeframe: str = "5m"
    process_only_new_candles: bool = True
    startup_candle_count: int = 100

    minimal_roi: Dict[str, float] = {"0": 100}
    use_custom_stoploss: bool = False
    stoploss: float = -1.0
    trailing_stop: bool = False

    # === диагностика
    DIAG_ONLY: bool = False
    DIAG_PATH: str = "user_data/logs/sko_signals_diag.csv"
    _diag_written_header: bool = False

    # индикатор (ratio + zscore)
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

    # ---------- helpers ----------
    def _ensure_diag_file(self):
        os.makedirs(os.path.dirname(self.DIAG_PATH), exist_ok=True)
        if not os.path.exists(self.DIAG_PATH):
            self._diag_written_header = False

    def _diag_write_rows(self, rows: List[Dict[str, Any]]):
        if not rows:
            return
        self._ensure_diag_file()
        fieldnames = list(rows[0].keys())
        write_header = not os.path.exists(self.DIAG_PATH) or not self._diag_written_header
        with open(self.DIAG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
                self._diag_written_header = True
            for r in rows:
                w.writerow(r)

    def _dump_sko_signals(self, df: pd.DataFrame):
        """
        Собираем все свечи, где есть sko_entry_long/short (на t+1), и считаем уровни от СКО (t).
        """
        s = int(self.cfg.signal_lag_bars)
        rows = []

        # индексы t+1 (входные бары)
        enter_idx = df.index[(df["sko_entry_long"].notna()) | (df["sko_entry_short"].notna())]
        for t1 in enter_idx:
            try:
                t1_pos = df.index.get_loc(t1)
            except Exception:
                continue
            t_pos = t1_pos - s
            if t_pos < 0:
                continue

            sko = df.iloc[t_pos]
            ent = df.iloc[t1_pos]

            # вход по open[t+1]
            entry_open = float(ent["open"])
            # диапазон от свечи СКО
            hi = float(sko["high"]); lo = float(sko["low"])
            rng = hi - lo

            # направление (по цвету СКО)
            sko_is_green = bool(sko["close"] > sko["open"])
            dir_sig = "LONG" if sko_is_green else "SHORT"

            # SL/TP по правилу 1R
            if dir_sig == "LONG":
                sl = lo
                tp = entry_open + rng
            else:
                sl = hi
                tp = entry_open - rng

            rows.append({
                "sko_time": str(sko.name),
                "entry_time_t1": str(ent.name),
                "dir": dir_sig,
                "t_open": float(sko["open"]),
                "t_high": hi,
                "t_low": lo,
                "t_close": float(sko["close"]),
                "t_volume": float(sko["volume"]),
                "zscore": float(sko.get("sko_zscore", float("nan"))),
                "is_valid": bool(sko.get("sko_is_valid", False)),
                "entry_open_t1": entry_open,
                "R_range": rng,
                "SL": sl,
                "TP": tp,
                "has_long_flag": bool(pd.notna(ent.get("sko_entry_long"))),
                "has_short_flag": bool(pd.notna(ent.get("sko_entry_short"))),
            })

        self._diag_write_rows(rows)

    # ---------- freqtrade API ----------
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = self._ind.compute(dataframe)
        # каждый
