# user_data/strategies/CompareLevelsVisualization.py
from pandas import DataFrame
from freqtrade.strategy import IStrategy

# Импортируем ОБА наших индикатора
from swing_levels import SwingLevels
from rolling_extremes import RollingExtremes

class CompareLevelsVisualization(IStrategy):
    """
    Эта стратегия сравнивает ДВА подхода к поиску уровней:
    1. Старый (Фракталы) - swing_...
    2. Новый (Скользящие Экстремумы) - rolling_...
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 200

    minimal_roi = {"0": 100}
    stoploss = -1.0

    plot_config = {
        'main_plot': {
            # --- Старый индикатор (Фракталы) ---
            'swing_high': {'color': 'gray', 'style': 'dot'},
            'swing_low': {'color': 'gray', 'style': 'dot'},
            # --- НОВЫЙ ИНДИКАТОР (Скользящие Экстремумы) ---
            'rolling_high': {'color': 'blue', 'width': 2},
            'rolling_low': {'color': 'red', 'width': 2},
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- Вызываем старый индикатор ---
        swing = SwingLevels(k=2)
        swing_high_values, swing_low_values = swing.calculate_swing_levels(dataframe)
        dataframe['swing_high'] = swing_high_values
        dataframe['swing_low'] = swing_low_values

        # --- Вызываем НОВЫЙ индикатор ---
        # Попробуем с окном в 10 свечей. Этот параметр можно будет менять.
        rolling = RollingExtremes(window=10)
        rolling_high_values, rolling_low_values = rolling.calculate_rolling_extremes(dataframe)
        dataframe['rolling_high'] = rolling_high_values
        dataframe['rolling_low'] = rolling_low_values

        return dataframe

    # Пустые методы
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        return dataframe
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe