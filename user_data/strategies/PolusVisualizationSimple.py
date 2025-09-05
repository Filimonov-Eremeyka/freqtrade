# user_data/strategies/PolusVisualizationSimple.py
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy

# Правильный импорт, который будет работать с настройкой docker-compose.yml
from polus_x import PolusX

class PolusVisualizationSimple(IStrategy):
    """
    Стратегия для визуализации индикатора PolusX с динамическими пунктирными уровнями.
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 200
    
    # Отключаем торговлю
    minimal_roi = {"0": 100}
    stoploss = -1.0
    
    # --- ОСНОВНОЕ ИЗМЕНЕНИЕ: plot_config для МНОЖЕСТВЕННЫХ уровней ---
    plot_config = {
        'main_plot': {
            'polus_high': {'color': 'steelblue', 'width': 2},
            'polus_low': {'color': 'orangered', 'width': 2},
            'polus_anvar': {'color': 'gray', 'width': 1},
        }
    }
    
    # Динамически добавляем пунктирные линии в plot_config
    # Это позволяет легко менять количество поддерживаемых уровней
    MAX_LEVELS = 5
    for i in range(MAX_LEVELS):
        plot_config['main_plot'][f'dashed_high_{i}'] = {'color': 'blue', 'type': 'line', 'style': 'dash'}
        plot_config['main_plot'][f'dashed_low_{i}'] = {'color': 'red', 'type': 'line', 'style': 'dash'}


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Добавляет индикаторы и динамические уровни
        """
        try:
            # Создаём индикатор
            polus = PolusX(use_tick_volume=False, show_lines=True, ind1=True, ind2=True, ind3=True)

            # Вычисляем только 3-й уровень
            max3, min3, cl = polus._calc_ind3(dataframe)
            
            # Создаём сплошные линии с помощью ffill
            dataframe['polus_high'] = pd.Series(max3, index=dataframe.index).fillna(method='ffill')
            dataframe['polus_low'] = pd.Series(min3, index=dataframe.index).fillna(method='ffill')
            dataframe['polus_anvar'] = pd.Series(cl, index=dataframe.index).fillna(method='ffill')
            
            # Создаём пунктирные динамические уровни
            dataframe = self._create_dynamic_levels(dataframe, self.MAX_LEVELS)
            
        except Exception as e:
            print(f"Error in populate_indicators: {e}")
            import traceback
            traceback.print_exc()

        return dataframe

    def _create_dynamic_levels(self, dataframe: DataFrame, max_levels: int) -> DataFrame:
        """
        Создает динамические горизонтальные уровни на основе ИЗМЕНЕНИЙ 
        в сплошных линиях polus_high и polus_low.
        """
        # Инициализируем колонки для N одновременных уровней
        for i in range(max_levels):
            dataframe[f'dashed_high_{i}'] = np.nan
            dataframe[f'dashed_low_{i}'] = np.nan

        active_high_levels = []
        active_low_levels = []
        
        # Используем .shift() для векторного определения моментов появления новых сигналов
        new_high_signal = (dataframe['polus_high'] != dataframe['polus_high'].shift(1)) & dataframe['polus_high'].notna()
        new_low_signal = (dataframe['polus_low'] != dataframe['polus_low'].shift(1)) & dataframe['polus_low'].notna()

        # Итерируемся по dataframe для stateful-логики (отслеживания активных уровней)
        for i in range(1, len(dataframe)):
            # --- Обработка ВЕРХНИХ уровней ---
            
            # 1. Проверяем, не коснулась ли цена существующих уровней
            high_levels_to_remove = []
            for level in active_high_levels:
                if dataframe.at[i, 'high'] >= level:
                    high_levels_to_remove.append(level)
            
            # Удаляем "пробитые" уровни
            if high_levels_to_remove:
                active_high_levels = [lvl for lvl in active_high_levels if lvl not in high_levels_to_remove]

            # 2. Проверяем, не появился ли НОВЫЙ уровень
            if new_high_signal.iloc[i] and len(active_high_levels) < max_levels:
                new_level = dataframe.at[i, 'polus_high']
                if new_level not in active_high_levels:
                     active_high_levels.append(new_level)

            # 3. "Рисуем" все активные на данный момент уровни в их колонки
            for idx, level in enumerate(active_high_levels):
                dataframe.loc[i, f'dashed_high_{idx}'] = level

            # --- Обработка НИЖНИХ уровней (аналогичная логика) ---
            
            # 1. Проверяем касание
            low_levels_to_remove = []
            for level in active_low_levels:
                if dataframe.at[i, 'low'] <= level:
                    low_levels_to_remove.append(level)

            if low_levels_to_remove:
                active_low_levels = [lvl for lvl in active_low_levels if lvl not in low_levels_to_remove]

            # 2. Проверяем новый сигнал
            if new_low_signal.iloc[i] and len(active_low_levels) < max_levels:
                new_level = dataframe.at[i, 'polus_low']
                if new_level not in active_low_levels:
                    active_low_levels.append(new_level)

            # 3. Рисуем активные уровни
            for idx, level in enumerate(active_low_levels):
                dataframe.loc[i, f'dashed_low_{idx}'] = level
                
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = False
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = False
        return dataframe