# user_data/strategies/PolusDebugSimple.py
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
import sys
import os

# Add the user_data directory to Python path to find our indicators
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from indicators.polus_x import PolusX

class PolusDebugSimple(IStrategy):
    """
    Отладочная версия стратегии для проверки базовой работы PolusX
    """
    
    INTERFACE_VERSION = 3
    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 200
    
    # Отключаем торговлю
    minimal_roi = {"0": 100}
    stoploss = -1.0
    
    # Только базовые линии без пунктиров для начала
    plot_config = {
        'main_plot': {
            'polus_high': {'color': 'steelblue'},
            'polus_low': {'color': 'orangered'}, 
            'polus_anvar': {'color': 'gray'},
            'test_dashed_high': {'color': 'blue'},
            'test_dashed_low': {'color': 'red'},
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in dataframe.columns:
                print(f"Warning: Missing column {col}")
                return dataframe

        try:
            print(f"Debug: Starting PolusX calculation for {len(dataframe)} candles")
            
            # Создаём индикатор
            polus = PolusX(use_tick_volume=False, show_lines=True, ind1=True, ind2=True, ind3=True)

            # Вычисляем только 3-й уровень индикаторов
            max3, min3, cl = polus._calc_ind3(dataframe)
            
            print(f"Debug: PolusX calculation completed")
            print(f"Debug: max3 type: {type(max3)}, shape: {max3.shape if hasattr(max3, 'shape') else 'no shape'}")
            print(f"Debug: min3 type: {type(min3)}, shape: {min3.shape if hasattr(min3, 'shape') else 'no shape'}")
            print(f"Debug: cl type: {type(cl)}, shape: {cl.shape if hasattr(cl, 'shape') else 'no shape'}")

            # Правильно преобразуем numpy arrays в pandas Series
            max3_series = pd.Series(max3, index=dataframe.index, name='max3')
            min3_series = pd.Series(min3, index=dataframe.index, name='min3') 
            cl_series = pd.Series(cl, index=dataframe.index, name='cl')
            
            print(f"Debug: max3_series - non-null count: {max3_series.notna().sum()}")
            print(f"Debug: min3_series - non-null count: {min3_series.notna().sum()}")
            print(f"Debug: cl_series - non-null count: {cl_series.notna().sum()}")
            
            # Создаём основные линии
            dataframe['polus_high'] = max3_series.ffill()
            dataframe['polus_low'] = min3_series.ffill()
            dataframe['polus_anvar'] = cl_series.ffill()
            
            # Простые тестовые пунктирные линии (без сложной логики)
            dataframe['test_dashed_high'] = np.nan
            dataframe['test_dashed_low'] = np.nan
            
            # Заполним несколько тестовых значений
            non_null_high = max3_series.dropna()
            non_null_low = min3_series.dropna()
            
            if len(non_null_high) > 0:
                # Просто копируем первые несколько сигналов для теста
                for idx in non_null_high.index[:min(5, len(non_null_high))]:
                    end_idx = min(idx + 10, len(dataframe) - 1)  # 10 свечей вперед
                    dataframe.loc[idx:end_idx, 'test_dashed_high'] = non_null_high.loc[idx]
                    
            if len(non_null_low) > 0:
                for idx in non_null_low.index[:min(5, len(non_null_low))]:
                    end_idx = min(idx + 10, len(dataframe) - 1)
                    dataframe.loc[idx:end_idx, 'test_dashed_low'] = non_null_low.loc[idx]
            
            print(f"Debug: Final dataframe columns: {list(dataframe.columns)}")
            print(f"Debug: polus_high non-null: {dataframe['polus_high'].notna().sum()}")
            print(f"Debug: polus_low non-null: {dataframe['polus_low'].notna().sum()}")
            print(f"Debug: test_dashed_high non-null: {dataframe['test_dashed_high'].notna().sum()}")
            print(f"Debug: test_dashed_low non-null: {dataframe['test_dashed_low'].notna().sum()}")
            
        except Exception as e:
            print(f"Error calculating PolusX indicators: {e}")
            import traceback
            traceback.print_exc()
            # Заполняем тестовыми данными в случае ошибки
            dataframe['polus_high'] = dataframe['high'] * 1.001  # Чуть выше цены
            dataframe['polus_low'] = dataframe['low'] * 0.999    # Чуть ниже цены
            dataframe['polus_anvar'] = dataframe['close']
            dataframe['test_dashed_high'] = np.nan
            dataframe['test_dashed_low'] = np.nan

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = False
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = False
        return dataframe