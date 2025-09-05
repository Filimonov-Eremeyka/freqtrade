# user_data/strategies/PolusVisualization.py
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
import sys
import os

# Add the user_data directory to Python path to find our indicators
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from indicators.polus_x import PolusX

class PolusVisualization(IStrategy):
    """
    Стратегия ТОЛЬКО для визуализации индикатора PolusX.
    Не генерирует сигналы для торговли - только показывает индикаторы на графике.
    """
    
    INTERFACE_VERSION = 3
    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 200
    
    # Отключаем торговлю
    minimal_roi = {"0": 100}  # Никогда не закрывать по ROI
    stoploss = -1.0  # Никогда не закрывать по стоп-лоссу
    
    # Конфигурация графика - ОСНОВНОЕ для визуализации
    plot_config = {
        'main_plot': {
            # Основные уровни PolusX
            'polus_high1': {
                'color': 'green', 
                'type': 'line',
                'width': 2
            },
            'polus_min1': {
                'color': 'red', 
                'type': 'line',
                'width': 2
            },
            'polus_max2': {
                'color': 'aqua', 
                'type': 'line',
                'width': 1
            },
            'polus_min2': {
                'color': 'magenta', 
                'type': 'line',
                'width': 1
            },
            'polus_max3': {
                'color': 'steelblue', 
                'type': 'scatter',  # Точки для третьего индикатора
                'width': 1
            },
            'polus_min3': {
                'color': 'orangered', 
                'type': 'scatter',
                'width': 1
            },
            'polus_cl': {
                'color': 'gold', 
                'type': 'scatter',
                'width': 1
            },
        },
        'subplots': {
            # Можно добавить дополнительные графики
            "Volume Analysis": {
                'volume': {'color': 'lightblue'},
            }
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Добавляет только индикаторы PolusX для визуализации
        """
        if dataframe.empty:
            return dataframe

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in dataframe.columns:
                print(f"Warning: Missing column {col}")
                return dataframe

        try:
            # Создаём индикатор
            polus = PolusX(use_tick_volume=False, show_lines=True, ind1=True, ind2=True, ind3=True)

            # Вычисляем все индикаторы
            high1, min1 = polus._calc_ind1(dataframe)
            max2, min2 = polus._calc_ind2(dataframe)  
            max3, min3, cl = polus._calc_ind3(dataframe)

            # Добавляем в dataframe
            dataframe['polus_high1'] = high1
            dataframe['polus_min1'] = min1
            dataframe['polus_max2'] = max2
            dataframe['polus_min2'] = min2
            dataframe['polus_max3'] = max3
            dataframe['polus_min3'] = min3
            dataframe['polus_cl'] = cl

            # Заполняем NaN для корректного отображения
            indicator_columns = ['polus_high1', 'polus_min1', 'polus_max2', 'polus_min2', 
                               'polus_max3', 'polus_min3', 'polus_cl']
            
            for col in indicator_columns:
                dataframe[col] = dataframe[col].fillna(method='ffill')
            
            print(f"PolusX indicators calculated for {metadata.get('pair', 'Unknown')} - {len(dataframe)} candles")
            
        except Exception as e:
            print(f"Error calculating PolusX indicators: {e}")
            import traceback
            traceback.print_exc()
            # Заполняем NaN в случае ошибки
            for col in ['polus_high1', 'polus_min1', 'polus_max2', 'polus_min2',
                       'polus_max3', 'polus_min3', 'polus_cl']:
                dataframe[col] = np.nan

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        НЕ генерируем сигналы входа - только для визуализации
        """
        dataframe.loc[:, 'enter_long'] = False  # Никогда не входим
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        НЕ генерируем сигналы выхода - только для визуализации
        """
        dataframe.loc[:, 'exit_long'] = False  # Никогда не выходим
        return dataframe