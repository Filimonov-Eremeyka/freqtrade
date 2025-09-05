# user_data/strategies/PolusAdvancedVisualization.py
import numpy as np
import pandas as pd
from pandas import DataFrame
import plotly.graph_objects as go
from freqtrade.strategy import IStrategy

# Этот импорт теперь ГАРАНТИРОВАННО сработает,
# т.к. polus_levels.py лежит в той же папке.
from polus_levels import PolusLevels

class PolusAdvancedVisualization(IStrategy):
    # ... (весь остальной код стратегии остается без изменений, как в моем позапрошлом сообщении)
    """
    Продвинутая стратегия для визуализации индикатора Polus.
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 200

    minimal_roi = {"0": 100}
    stoploss = -1.0

    plot_config = {
        'main_plot': {
            'polus_high_base': {'color': 'steelblue', 'width': 2},
            'polus_low_base': {'color': 'orangered', 'width': 2},
            'polus_close_base': {'color': 'gray', 'width': 1, 'style': 'dot'},
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        polus = PolusLevels()
        high_signals, low_signals, close_signals = polus.calculate_levels(dataframe)

        dataframe['polus_high_signal'] = high_signals
        dataframe['polus_low_signal'] = low_signals

        dataframe['polus_high_base'] = pd.Series(high_signals, index=dataframe.index).ffill()
        dataframe['polus_low_base'] = pd.Series(low_signals, index=dataframe.index).ffill()
        dataframe['polus_close_base'] = pd.Series(close_signals, index=dataframe.index).ffill()

        return dataframe

    def custom_plot_additions(self, fig: go.Figure):
        df = self.dp.get_analyzed_dataframe(self.processed)

        active_high_levels = []
        active_low_levels = []

        for i in range(len(df)):
            candle = df.iloc[i]

            levels_to_remove = []
            for level in active_high_levels:
                if candle['high'] >= level['value']:
                    fig.add_shape(type="line",
                                  x0=level['start_date'], y0=level['value'],
                                  x1=candle['date'], y1=level['value'],
                                  line=dict(color="blue", width=1, dash="dash"))
                    levels_to_remove.append(level)
            
            if levels_to_remove:
                active_high_levels = [lvl for lvl in active_high_levels if lvl not in levels_to_remove]

            if pd.notna(candle['polus_high_signal']):
                active_high_levels.append({
                    'value': candle['polus_high_signal'],
                    'start_date': candle['date']
                })

            levels_to_remove = []
            for level in active_low_levels:
                if candle['low'] <= level['value']:
                    fig.add_shape(type="line",
                                  x0=level['start_date'], y0=level['value'],
                                  x1=candle['date'], y1=level['value'],
                                  line=dict(color="red", width=1, dash="dash"))
                    levels_to_remove.append(level)

            if levels_to_remove:
                active_low_levels = [lvl for lvl in active_low_levels if lvl not in levels_to_remove]

            if pd.notna(candle['polus_low_signal']):
                active_low_levels.append({
                    'value': candle['polus_low_signal'],
                    'start_date': candle['date']
                })

        last_date = df.iloc[-1]['date']
        for level in active_high_levels:
            fig.add_shape(type="line",
                          x0=level['start_date'], y0=level['value'],
                          x1=last_date, y1=level['value'],
                          line=dict(color="blue", width=1, dash="dash"))
        for level in active_low_levels:
            fig.add_shape(type="line",
                          x0=level['start_date'], y0=level['value'],
                          x1=last_date, y1=level['value'],
                          line=dict(color="red", width=1, dash="dash"))
                          
        return fig

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = False
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = False
        return dataframe