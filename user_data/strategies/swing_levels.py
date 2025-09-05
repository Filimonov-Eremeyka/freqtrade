# user_data/strategies/swing_levels.py
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Tuple

class SwingLevels:
    """
    Класс для расчета Свинг-уровней (пивотов) на основе фракталов Билла Вильямса.
    Рассчитывает последние подтвержденные точки Свинг-Хай и Свинг-Лоу.
    """
    def __init__(self, k: int = 2):
        """
        Инициализация индикатора.
        :param k: Радиус пивота. k=2 означает, что свеча должна быть
                  максимумом/минимумом среди 5 свечей (2 слева, 2 справа, сама свеча).
        """
        if k < 1:
            raise ValueError("Радиус пивота 'k' должен быть 1 или больше.")
        self.k = k

    def calculate_swing_levels(self, df: DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Основной метод расчета.
        Возвращает два "ступенчатых" массива numpy:
        - tp_long: Уровень последнего подтвержденного Свинг-Хая.
        - tp_short: Уровень последнего подтвержденного Свинг-Лоу.
        """
        # 1. Находим "сырые" пивоты.
        #    Это точки, которые являются локальными экстремумами в окне из (2*k + 1) свечей.
        #    Важно: .rolling() с center=True использует будущие данные, поэтому сигнал
        #    будет подтвержден только через k свечей.
        window_size = 2 * self.k + 1
        roll_max = df["high"].rolling(window=window_size, center=True).max()
        roll_min = df["low"].rolling(window=window_size, center=True).min()

        ph_raw = (df["high"] == roll_max)
        pl_raw = (df["low"] == roll_min)

        # 2. Подтверждаем пивоты.
        #    Мы сдвигаем "сырые" сигналы на k свечей вправо. Это симулирует задержку,
        #    с которой мы бы узнали об этом пивоте в реальной торговле.
        ph_confirmed = ph_raw.shift(self.k).fillna(False)
        pl_confirmed = pl_raw.shift(self.k).fillna(False)

        # 3. Создаем "точечные" уровни только в местах подтвержденных пивотов.
        tp_long_points = np.where(ph_confirmed, df["high"], np.nan)
        tp_short_points = np.where(pl_confirmed, df["low"], np.nan)

        # 4. "Растягиваем" эти точечные уровни вправо, создавая "ступеньки".
        #    Это делается с помощью ffill (forward fill), который заполняет
        #    пустые значения последним известным не-пустым значением.
        tp_long_steps = pd.Series(tp_long_points, index=df.index).ffill()
        tp_short_steps = pd.Series(tp_short_points, index=df.index).ffill()

        # Возвращаем результат в виде массивов numpy
        return tp_long_steps.values, tp_short_steps.values

