# user_data/strategies/PolusStrategy.py
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy

# Убедитесь, что путь к индикатору правильный
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from indicators.polus_x import PolusX


class PolusStrategy(IStrategy):
    """
    Стратегия на основе индикатора PolusX
    """
    
    # Базовые настройки стратегии
    INTERFACE_VERSION = 3
    
    # Настройки временного интервала
    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 200

    # Настройки ROI и стоп-лосса
    minimal_roi = {
        "60": 0.01,
        "30": 0.02, 
        "0": 0.04
    }
    stoploss = -0.10
    trailing_stop = False
    
    # Настройки сигналов выхода
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Конфигурация графика
    plot_config = {
        'main_plot': {
            'polus_high1': {'color': 'green'},
            'polus_min1':  {'color': 'red'},
            'polus_max2':  {'color': 'aqua'},
            'polus_min2':  {'color': 'magenta'},
            'polus_max3':  {'color': 'steelblue'},
            'polus_min3':  {'color': 'orangered'},
            'polus_cl':    {'color': 'gold'},
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Добавляет индикаторы PolusX к dataframe
        """
        if dataframe.empty:
            return dataframe

        # Проверяем наличие необходимых колонок
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in dataframe.columns:
                print(f"Warning: Missing column {col}")
                return dataframe

        try:
            # Создаём экземпляр индикатора
            polus = PolusX(use_tick_volume=False, show_lines=True, ind1=True, ind2=True, ind3=True)

            # Вычисляем индикаторы
            high1, min1 = polus._calc_ind1(dataframe)
            max2, min2 = polus._calc_ind2(dataframe)  
            max3, min3, cl = polus._calc_ind3(dataframe)

            # Добавляем индикаторы в dataframe
            dataframe['polus_high1'] = high1
            dataframe['polus_min1']  = min1
            dataframe['polus_max2']  = max2
            dataframe['polus_min2']  = min2
            dataframe['polus_max3']  = max3
            dataframe['polus_min3']  = min3
            dataframe['polus_cl']    = cl

            # Добавляем сдвинутые значения для сравнения
            dataframe['close_prev'] = dataframe['close'].shift(1)
            dataframe['ph1_prev']   = dataframe['polus_high1'].shift(1)
            dataframe['pm1_prev']   = dataframe['polus_min1'].shift(1)
            
            # Заполняем NaN значения для корректной работы
            for col in ['polus_high1', 'polus_min1', 'polus_max2', 'polus_min2', 
                       'polus_max3', 'polus_min3', 'polus_cl']:
                dataframe[col] = dataframe[col].fillna(method='ffill')
            
        except Exception as e:
            print(f"Error in populate_indicators: {e}")
            # В случае ошибки возвращаем dataframe с NaN значениями
            for col in ['polus_high1', 'polus_min1', 'polus_max2', 'polus_min2',
                       'polus_max3', 'polus_min3', 'polus_cl', 'close_prev', 
                       'ph1_prev', 'pm1_prev']:
                dataframe[col] = np.nan

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Определяет сигналы на вход в позицию
        """
        # Инициализируем колонку входа
        dataframe.loc[:, 'enter_long'] = False
        
        # Проверяем наличие необходимых колонок
        required_cols = ['close', 'polus_high1', 'close_prev', 'ph1_prev']
        if not all(col in dataframe.columns for col in required_cols):
            return dataframe
            
        try:
            # Условие входа: цена закрытия пересекает вверх polus_high1
            cond = (
                (dataframe['close'] > dataframe['polus_high1']) & 
                (dataframe['close_prev'] <= dataframe['ph1_prev']) &
                (dataframe['volume'] > 0) &  # Убеждаемся что есть объём
                (~dataframe['polus_high1'].isna()) &  # Индикатор не NaN
                (~dataframe['ph1_prev'].isna())  # Предыдущее значение не NaN
            )
            
            dataframe.loc[cond, 'enter_long'] = True
            dataframe.loc[cond, 'enter_tag'] = 'cross_ph1_up'
            
        except Exception as e:
            print(f"Error in populate_entry_trend: {e}")
            
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Определяет сигналы на выход из позиции
        Выход когда close пересекает ВНИЗ polus_min1 (пробой «нижнего» уровня).
        """
        # Инициализируем колонку выхода
        dataframe.loc[:, 'exit_long'] = False

        # Проверяем наличие необходимых колонок
        required_cols = ['close', 'polus_min1', 'close_prev', 'pm1_prev']
        if not all(col in dataframe.columns for col in required_cols):
            return dataframe

        try:
            # Условие выхода: цена закрытия пересекает вниз polus_min1
            cond_cross_down = (
                (dataframe['close'] < dataframe['polus_min1']) &
                (dataframe['close_prev'] >= dataframe['pm1_prev']) &
                (dataframe['volume'] > 0) &  # Убеждаемся что есть объём
                (~dataframe['polus_min1'].isna()) &  # Индикатор не NaN
                (~dataframe['pm1_prev'].isna())  # Предыдущее значение не NaN
            )

            dataframe.loc[cond_cross_down, 'exit_long'] = True
            dataframe.loc[cond_cross_down, 'exit_tag'] = 'cross_pm1_down'
            
        except Exception as e:
            print(f"Error in populate_exit_trend: {e}")

        return dataframe