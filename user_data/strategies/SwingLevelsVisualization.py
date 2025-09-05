# user_data/strategies/SwingLevelsVisualization.py
from pandas import DataFrame
from freqtrade.strategy import IStrategy

# Импортируем наш новый индикатор. Это должно сработать, т.к. файлы лежат рядом.
from swing_levels import SwingLevels

class SwingLevelsVisualization(IStrategy):
    """
    Эта стратегия предназначена ТОЛЬКО для проверки,
    что индикатор swing_levels.py работает правильно.
    Она должна нарисовать две ступенчатые линии на графике.
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 200

    # Отключаем торговлю
    minimal_roi = {"0": 100}
    stoploss = -1.0

    # Явно указываем, что мы хотим нарисовать две колонки,
    # которые будет создавать наш индикатор.
    plot_config = {
        'main_plot': {
            'tp_long': {'color': 'blue', 'style': 'dash'},
            'tp_short': {'color': 'red', 'style': 'dash'},
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1. Создаем экземпляр нашего индикатора
        # k=2 - это классический фрактал Вильямса (5 свечей)
        swing = SwingLevels(k=2)

        # 2. Вызываем метод расчета
        tp_long_values, tp_short_values = swing.calculate_swing_levels(dataframe)

        # 3. Добавляем полученные данные в dataframe, чтобы их можно было нарисовать
        dataframe['tp_long'] = tp_long_values
        dataframe['tp_short'] = tp_short_values

        print("INFO: swing_levels.py - индикатор успешно отработал.")

        return dataframe

    # Пустые методы, чтобы стратегия была валидной
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe