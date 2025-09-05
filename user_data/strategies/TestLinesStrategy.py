# user_data/strategies/TestLinesStrategy.py
import pandas as pd
from pandas import DataFrame
import plotly.graph_objects as go
from freqtrade.strategy import IStrategy

class TestLinesStrategy(IStrategy):
    """
    Эта стратегия предназначена только для одного: проверить, работает ли
    функция custom_plot_additions в принципе.
    Она должна нарисовать одну синюю горизонтальную линию в центре графика.
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 20 # Нам не нужны индикаторы, поэтому 20 достаточно

    # Нам даже не нужен plot_config
    plot_config = {}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Ничего не делаем, просто возвращаем исходные данные
        print("INFO: TestLinesStrategy - populate_indicators called.")
        return dataframe

    def custom_plot_additions(self, fig: go.Figure):
        # Это наш главный тест. Мы не используем данные из dataframe.
        # Мы просто рисуем одну линию с жестко заданными координатами.
        
        print("INFO: TestLinesStrategy - custom_plot_additions CALLED. Trying to draw a line...")
        
        # Берем даты из середины вашего тестового диапазона
        start_test_date = "2025-08-17 10:00:00"
        end_test_date = "2025-08-17 18:00:00"
        
        # Берем цену из середины вашего ценового диапазона на скриншоте
        price_level = 118.1 * 1000 # 118.1k
        
        fig.add_shape(
            type="line",
            x0=start_test_date, y0=price_level,
            x1=end_test_date, y1=price_level,
            line=dict(color="blue", width=3, dash="dash")
        )
        
        print(f"INFO: TestLinesStrategy - Line from {start_test_date} to {end_test_date} at {price_level} SHOULD BE DRAWN.")
        
        return fig

    # Пустые методы, чтобы стратегия была валидной
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe