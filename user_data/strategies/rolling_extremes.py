# user_data/strategies/rolling_extremes.py
import pandas as pd
from pandas import DataFrame
from typing import Tuple

class RollingExtremes:
    """
    Класс для расчета "Скользящих Экстремумов".
    Находит самые высокие максимумы и самые низкие минимумы
    за определенный период (окно).
    """
    def __init__(self, window: int = 10):
        """
        :param window: Количество свечей для поиска экстремума.
        """
        self.window = window

    def calculate_rolling_extremes(self, df: DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Возвращает две серии: со скользящими максимумами и минимумами.
        """
        # .rolling(window=self.window) - создает "скользящее окно" из N свечей
        # .max() / .min() - находит экстремум в этом окне
        rolling_high = df['high'].rolling(window=self.window).max()
        rolling_low = df['low'].rolling(window=self.window).min()

        return rolling_high, rolling_low