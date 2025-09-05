# user_data/indicators/polus_levels.py
import numpy as np
import pandas as pd
from typing import Tuple

class PolusLevels:
    """
    Облегченный класс индикатора Polus.
    Рассчитывает только уровни поддержки/сопротивления и "цену закрытия"
    на основе локальных минимумов объема.
    """
    def __init__(self, use_tick_volume: bool = False):
        """
        Инициализация индикатора.
        :param use_tick_volume: Использовать 'tick_volume' вместо 'volume', если доступно.
        """
        self.use_tick_volume = use_tick_volume

    def calculate_levels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Основной метод расчета. Возвращает три массива numpy.
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Выбираем колонку с объемом
        volume_col = 'tick_volume' if self.use_tick_volume and 'tick_volume' in df.columns else 'volume'
        vol = df[volume_col].values

        # Инициализируем пустые массивы
        high_levels = np.full(len(df), np.nan)
        low_levels = np.full(len(df), np.nan)
        close_levels = np.full(len(df), np.nan)

        # Проходим по всем свечам, начиная со второй
        # (т.к. нам нужно смотреть на предыдущую свечу vol[i-1])
        for i in range(1, len(df) - 1):
            # Ваше ключевое условие: находим локальный минимум объема
            if vol[i - 1] > vol[i] < vol[i + 1]:
                high_levels[i] = high[i]
                low_levels[i] = low[i]
                close_levels[i] = close[i - 1]

        return high_levels, low_levels, close_levels