# indicators/polus_x.py
import numpy as np
import pandas as pd
from typing import Tuple

# ----------------- вспомогательные функции -----------------
def _find_volume_minimum(df: pd.DataFrame, center: int) -> int:
    """Возвращает индекс локального минимума объёма или -1."""
    if center <= 0 or center >= len(df) - 1:
        return -1
    vol = df['volume'].values
    if vol[center - 1] > vol[center] < vol[center + 1]:
        return center
    return -1

# ----------------- основной класс -----------------
class PolusX:
    """
    Класс-обёртка для индикатора PolusX.
    Принимает DataFrame с колонками:
    open, high, low, close, volume (и open_time как индекс или колонка)
    """
    def __init__(self,
                 use_tick_volume: bool = False,  # Изменено на False, т.к. tick_volume может отсутствовать
                 show_lines: bool = True,
                 ind1: bool = True,
                 ind2: bool = True,
                 ind3: bool = True):
        self.use_tick_volume = use_tick_volume
        self.show_lines = show_lines
        self.ind1 = ind1
        self.ind2 = ind2
        self.ind3 = ind3

    # ---------- индикатор-1 ----------
    def _calc_ind1(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        high = df['high'].values
        low = df['low'].values
        # Исправлено: используем volume, если tick_volume недоступен
        volume_col = 'tick_volume' if self.use_tick_volume and 'tick_volume' in df.columns else 'volume'
        vol = df[volume_col].values

        max_arr = np.full(len(df), np.nan)  # Изменено для корректной инициализации
        min_arr = np.full(len(df), np.nan)

        if len(df) == 0:
            return max_arr, min_arr

        max_val = high[0]  # Инициализация первым значением
        min_val = low[0]
        last_dj = 0
        
        for i in range(len(df)):
            if i == 0:
                max_val = high[i]
                min_val = low[i]
            else:
                local_dj = _find_volume_minimum(df, i)
                if local_dj != -1:
                    last_dj = local_dj

                if high[i] >= max_val and last_dj < len(df):
                    if last_dj < len(df):
                        min_val = low[last_dj]

                if i < len(df) - 1 and df['close'].iloc[i + 1] < min_val and last_dj < len(df):
                    max_val = high[i + 1] if i + 1 < len(high) else high[i]
                    if last_dj < len(df):
                        min_val = low[last_dj]

                if i < len(df) - 1 and i + 1 < len(high) and high[i + 1] >= max_val:
                    max_val = high[i + 1]

            max_arr[i] = max_val if self.show_lines else np.nan
            min_arr[i] = min_val

        return max_arr, min_arr

    # ---------- индикатор-2 ----------
    def _calc_ind2(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        high = df['high'].values
        low = df['low'].values
        # Исправлено: используем volume, если tick_volume недоступен
        volume_col = 'tick_volume' if self.use_tick_volume and 'tick_volume' in df.columns else 'volume'
        vol = df[volume_col].values

        max_arr = np.full(len(df), np.nan)
        min_arr = np.full(len(df), np.nan)

        if len(df) == 0:
            return max_arr, min_arr

        max2 = high[0]
        min2 = low[0]
        dj = 0
        
        for i in range(len(df)):
            if i == 0:
                max2 = high[i]
                min2 = low[i]
            else:
                if i > 0 and i < len(df) - 1:
                    if vol[i - 1] > vol[i] < vol[i + 1]:
                        dj = i

                if low[i] <= min2 and dj < len(df):
                    if dj < len(high):
                        max2 = high[dj]

                if i < len(df) - 1 and df['close'].iloc[i + 1] > max2 and dj < len(df):
                    if i + 1 < len(low):
                        min2 = low[i + 1]
                    if dj < len(high):
                        max2 = high[dj]

                if i < len(df) - 1 and i + 1 < len(low) and low[i + 1] <= min2:
                    min2 = low[i + 1]

            max_arr[i] = max2
            min_arr[i] = min2 if self.show_lines else np.nan

        return max_arr, min_arr

    # ---------- индикатор-3 ----------
    def _calc_ind3(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        # Исправлено: используем volume, если tick_volume недоступен
        volume_col = 'tick_volume' if self.use_tick_volume and 'tick_volume' in df.columns else 'volume'
        vol = df[volume_col].values

        max3_arr = np.full(len(df), np.nan)
        min3_arr = np.full(len(df), np.nan)
        cl_arr = np.full(len(df), np.nan)

        for i in range(1, len(df) - 1):
            if vol[i - 1] > vol[i] < vol[i + 1]:
                max3_arr[i] = high[i]
                min3_arr[i] = low[i]
                cl_arr[i] = close[i - 1]

        return max3_arr, min3_arr, cl_arr