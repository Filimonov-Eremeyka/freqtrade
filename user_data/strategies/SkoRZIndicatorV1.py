# SkoRZIndicatorV1.py
# Version: 1.0.2
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

__all__ = ["SkoConfig", "SkoRZIndicatorV1"]

@dataclass
class SkoConfig:
    min_vol_ratio_prev: float = 1.2
    min_vol_ratio_next: float = 1.3
    zscore_lookback: int = 100
    zscore_threshold: float = 2.0
    use_log_volume: bool = True
    signal_lag_bars: int = 1

class SkoRZIndicatorV1:
    """
    SKO (Свеча Контрастного Объёма): ratio к соседям + z-score по объёму.
    Вход на t+1 (open текущей свечи), SL/TP = 1R от диапазона свечи SKO (t).
    """
    __version__ = "1.0.2"

    def __init__(self, cfg: Optional[SkoConfig] = None):
        self.cfg = cfg or SkoConfig()

    def _zscore(self, x: pd.Series, lookback: int) -> pd.Series:
        roll = x.rolling(lookback, min_periods=lookback)
        mean = roll.mean()
        std = roll.std(ddof=0)
        z = (x - mean) / std.replace(0.0, np.nan)
        return z

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        vol = df["volume"].astype(float)
        vol_ref = np.log1p(np.clip(vol, 0.0, None)) if self.cfg.use_log_volume else vol

        z = self._zscore(vol_ref, self.cfg.zscore_lookback)
        df["sko_zscore"] = z

        vol_prev = vol.shift(1)
        vol_next = vol.shift(-1)
        neighbors_ok = vol_prev.notna() & vol_next.notna()

        ratio_prev_ok = vol > (vol_prev * self.cfg.min_vol_ratio_prev)
        ratio_next_ok = vol > (vol_next * self.cfg.min_vol_ratio_next)
        z_ok = z >= self.cfg.zscore_threshold

        is_sko = neighbors_ok & ratio_prev_ok & ratio_next_ok & z_ok
        df["sko_is_valid"] = is_sko

        s = self.cfg.signal_lag_bars
        rng = (df["high"] - df["low"]).shift(s)

        sig_green = df["close"] > df["open"]
        sig_red = df["close"] < df["open"]
        # избегаем FutureWarning / object downcast
        valid_signal = is_sko.shift(s).fillna(False).astype(bool)

        entry_price = df["open"]

        df["sko_entry_long"]  = np.where(valid_signal & sig_green, entry_price, np.nan)
        df["sko_stop_long"]   = np.where(valid_signal & sig_green, df["low"].shift(s), np.nan)
        df["sko_take_long"]   = np.where(valid_signal & sig_green, entry_price + rng, np.nan)

        df["sko_entry_short"] = np.where(valid_signal & sig_red, entry_price, np.nan)
        df["sko_stop_short"]  = np.where(valid_signal & sig_red, df["high"].shift(s), np.nan)
        df["sko_take_short"]  = np.where(valid_signal & sig_red, entry_price - rng, np.nan)

        return df
