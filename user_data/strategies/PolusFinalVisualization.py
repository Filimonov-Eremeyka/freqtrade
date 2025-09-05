# user_data/strategies/PolusFinalVisualization.py
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy

# Импорт работает, т.к. polus_levels.py лежит в той же папке
from polus_levels import PolusLevels

class PolusFinalVisualization(IStrategy):
    """
    ФИНАЛЬНАЯ СТРАТЕГИЯ: отказ от custom_plot_additions.
    Используем единственный 100% рабочий метод: N колонок в DataFrame + plot_config.
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 200

    minimal_roi = {"0": 100}
    stoploss = -1.0

    # --- КЛЮЧЕВОЙ МОМЕНТ: ЯВНО ПРОПИСЫВАЕМ ВСЕ КОЛОНКИ УРОВНЕЙ ---
    plot_config = {
        'main_plot': {
            # --- Уровни сопротивления (High) ---
            'level_high_0': {'color': 'blue', 'style': 'dash'},
            'level_high_1': {'color': 'blue', 'style': 'dash'},
            'level_high_2': {'color': 'blue', 'style': 'dash'},
            'level_high_3': {'color': 'blue', 'style': 'dash'},
            'level_high_4': {'color': 'blue', 'style': 'dash'},
            # --- Уровни поддержки (Low) ---
            'level_low_0': {'color': 'red', 'style': 'dash'},
            'level_low_1': {'color': 'red', 'style': 'dash'},
            'level_low_2': {'color': 'red', 'style': 'dash'},
            'level_low_3': {'color': 'red', 'style': 'dash'},
            'level_low_4': {'color': 'red', 'style': 'dash'},
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1. Получаем точечные сигналы от вашего индикатора
        polus = PolusLevels()
        high_signals, low_signals, _ = polus.calculate_levels(dataframe)

        dataframe['polus_high_signal'] = high_signals
        dataframe['polus_low_signal'] = low_signals

        # 2. Управляем жизненным циклом уровней и заполняем N колонок
        dataframe = self._manage_levels(dataframe, max_levels=5)
        
        return dataframe

        # ЗАМЕНИТЕ СТАРУЮ ФУНКЦИЮ _manage_levels НА ЭТУ:
    def _manage_levels(self, dataframe: DataFrame, max_levels: int) -> DataFrame:
        # Инициализируем все N колонок
        for i in range(max_levels):
            dataframe[f'level_high_{i}'] = np.nan
            dataframe[f'level_low_{i}'] = np.nan

        active_high_levels = {}  # {price: slot_index}
        active_low_levels = {}   # {price: slot_index}

        for i in range(len(dataframe)):
            # --- ВЕРХНИЕ УРОВНИ ---
            
            # 1. Проверяем пробой существующих уровней
            levels_to_remove = []
            for price, slot in active_high_levels.items():
                if dataframe.at[i, 'high'] >= price:
                    levels_to_remove.append((price, slot))
                    # ИСПРАВЛЕНИЕ: В момент пробоя, ставим NaN, чтобы "разорвать" линию
                    dataframe.loc[i, f'level_high_{slot}'] = np.nan

            # Удаляем пробитые уровни из активного списка
            for price, slot in levels_to_remove:
                del active_high_levels[price]

            # 2. Проверяем появление нового сигнала
            new_high_signal = dataframe.at[i, 'polus_high_signal']
            if pd.notna(new_high_signal) and new_high_signal not in active_high_levels:
                used_slots = active_high_levels.values()
                for slot in range(max_levels):
                    if slot not in used_slots:
                        active_high_levels[new_high_signal] = slot
                        break
            
            # 3. "Протягиваем" все еще активные уровни в их колонках
            for price, slot in active_high_levels.items():
                dataframe.loc[i, f'level_high_{slot}'] = price

            # --- НИЖНИЕ УРОВНИ (аналогичное исправление) ---
            levels_to_remove = []
            for price, slot in active_low_levels.items():
                if dataframe.at[i, 'low'] <= price:
                    levels_to_remove.append((price, slot))
                    # ИСПРАВЛЕНИЕ: В момент пробоя, ставим NaN, чтобы "разорвать" линию
                    dataframe.loc[i, f'level_low_{slot}'] = np.nan

            for price, slot in levels_to_remove:
                del active_low_levels[price]

            new_low_signal = dataframe.at[i, 'polus_low_signal']
            if pd.notna(new_low_signal) and new_low_signal not in active_low_levels:
                used_slots = active_low_levels.values()
                for slot in range(max_levels):
                    if slot not in used_slots:
                        active_low_levels[new_low_signal] = slot
                        break
            
            for price, slot in active_low_levels.items():
                dataframe.loc[i, f'level_low_{slot}'] = price

        return dataframe

    # Пустые методы
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe