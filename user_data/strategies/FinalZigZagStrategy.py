# user_data/strategies/FinalZigZagStrategy.py
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import logging

# Импортируем нашу реализацию ZigZag
from zigzag_indicator import zigzag

class FinalZigZagStrategy(IStrategy):
    """
    ФИНАЛЬНАЯ СТРАТЕГИЯ ВЕРСИЯ 4.0 (с исправлениями от эксперта):
    - Понижен порог ZigZag для 5m таймфрейма.
    - Исправлена ошибка TypeError при логировании.
    - Улучшена работа с DataFrame для предотвращения проблем с типами данных.
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 200 # <-- Возвращаем стандартное значение, т.к. таймрейндж достаточный

    logger = logging.getLogger(__name__)

    minimal_roi = {"0": 100}
    stoploss = -1.0

    # --- ИЗМЕНЕННЫЕ ПАРАМЕТРЫ для теста ---
    zigzag_percent: float = 0.001    # 0.5% - более адекватный порог для 5m
    level_len: int = 60              # Время жизни уровня
    max_slots: int = 4               # Максимум параллельных уровней
    min_dist_pct: float = 0.0001      # 0.1% - более мягкий фильтр для близких уровней

    # --- plot_config с plotly-синтаксисом ---
    plot_config = {
        'main_plot': {
            'zigzag_peak': {
                'type': 'scatter',
                'plotly': {'mode': 'markers', 'marker': {'symbol': 'diamond', 'size': 9}}
            },
            'zigzag_trough': {
                'type': 'scatter',
                'plotly': {'mode': 'markers', 'marker': {'symbol': 'diamond', 'size': 9}}
            },
            'zigzag_high_0': {'type': 'scatter', 'plotly': {'mode': 'lines'}},
            'zigzag_high_1': {'type': 'scatter', 'plotly': {'mode': 'lines'}},
            'zigzag_high_2': {'type': 'scatter', 'plotly': {'mode': 'lines'}},
            'zigzag_high_3': {'type': 'scatter', 'plotly': {'mode': 'lines'}},
            'zigzag_low_0':  {'type': 'scatter', 'plotly': {'mode': 'lines'}},
            'zigzag_low_1':  {'type': 'scatter', 'plotly': {'mode': 'lines'}},
            'zigzag_low_2':  {'type': 'scatter', 'plotly': {'mode': 'lines'}},
            'zigzag_low_3':  {'type': 'scatter', 'plotly': {'mode': 'lines'}},
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1) ZigZag
        peaks, troughs = zigzag(dataframe, peak_pct=self.zigzag_percent, trough_pct=self.zigzag_percent)
        dataframe['zigzag_peak'] = peaks
        dataframe['zigzag_trough'] = troughs

        # --- Логи по ZigZag ---
        peak_cnt = int(pd.notna(dataframe['zigzag_peak']).sum())
        trough_cnt = int(pd.notna(dataframe['zigzag_trough']).sum())
        self.logger.info(f"[ZZ] peaks={peak_cnt}, troughs={trough_cnt}, timeframe={self.timeframe}")

        if peak_cnt == 0 and trough_cnt == 0:
            self.logger.warning(f"[ZZ] No peaks/troughs. Try lower zigzag_percent (now {self.zigzag_percent}).")

        if peak_cnt > 0:
            peak_idx = list(np.where(pd.notna(dataframe['zigzag_peak']))[0])
            self.logger.info(f"[ZZ] first/last peak idx: {peak_idx[0]} / {peak_idx[-1]}")
        if trough_cnt > 0:
            trough_idx = list(np.where(pd.notna(dataframe['zigzag_trough']))[0])
            self.logger.info(f"[ZZ] first/last trough idx: {trough_idx[0]} / {trough_idx[-1]}")

        # 2) Уровни
        dataframe = self._build_timeboxed_levels(
            df=dataframe,
            point_high_col='zigzag_peak',
            point_low_col='zigzag_trough'
        )

        # --- ИСПРАВЛЕННЫЕ ЛОГИ по уровням ---
        level_cols = [c for c in dataframe.columns if c.startswith('zigzag_high_') or c.startswith('zigzag_low_')]
        nonempty_levels = {c: int(pd.notna(dataframe[c]).sum()) for c in level_cols}
        total_points = sum(nonempty_levels.values())
        self.logger.info(f"[LVL] total level-points={total_points} per column={nonempty_levels}")

        if total_points == 0 and (peak_cnt > 0 or trough_cnt > 0):
            self.logger.warning("[LVL] All level columns are empty despite ZZ points. Check logic.")

        return dataframe

    def _build_timeboxed_levels(self, df: DataFrame, point_high_col: str, point_low_col: str) -> DataFrame:
        # ИСПРАВЛЕНИЕ: работаем с копией и сбрасываем индекс, чтобы не менять оригинал
        df_processed = df.copy()
        df_processed.reset_index(drop=True, inplace=True)
        
        for s in range(self.max_slots):
            df_processed[f'zigzag_high_{s}'] = np.nan
            df_processed[f'zigzag_low_{s}']  = np.nan

        active_high = {}
        active_low = {}
        created_high = 0
        created_low = 0

        def _far_enough(active_dict, price) -> bool:
            for v in active_dict.values():
                if v['price'] == 0: continue
                if abs(price - v['price']) / v['price'] < self.min_dist_pct:
                    return False
            return True

        def _place_new_level(active_dict, price, kind: str):
            nonlocal created_high, created_low
            for s in range(self.max_slots):
                if s not in active_dict:
                    active_dict[s] = {'price': float(price), 'ttl': self.level_len}
                    if kind == 'H': created_high += 1
                    else: created_low += 1
                    return
            oldest_slot = min(active_dict, key=lambda k: active_dict[k]['ttl'])
            active_dict[oldest_slot] = {'price': float(price), 'ttl': self.level_len}
            if kind == 'H': created_high += 1
            else: created_low += 1

        high_col_idx = df_processed.columns.get_loc(point_high_col)
        low_col_idx = df_processed.columns.get_loc(point_low_col)

        for i in range(len(df_processed)):
            for s, v in list(active_high.items()):
                df_processed.iat[i, df_processed.columns.get_loc(f'zigzag_high_{s}')] = v['price']
                v['ttl'] -= 1
                if v['ttl'] <= 0:
                    del active_high[s]

            for s, v in list(active_low.items()):
                df_processed.iat[i, df_processed.columns.get_loc(f'zigzag_low_{s}')] = v['price']
                v['ttl'] -= 1
                if v['ttl'] <= 0:
                    del active_low[s]

            peak_price = df_processed.iat[i, high_col_idx]
            if not np.isnan(peak_price) and _far_enough(active_high, peak_price):
                _place_new_level(active_high, peak_price, 'H')

            trough_price = df_processed.iat[i, low_col_idx]
            if not np.isnan(trough_price) and _far_enough(active_low, trough_price):
                _place_new_level(active_low, trough_price, 'L')

        self.logger.info(f"[LVL] created_high={created_high}, created_low={created_low}")

        # ИСПРАВЛЕНИЕ: Копируем только нужные колонки обратно в исходный df
        level_cols = [c for c in df_processed.columns if c.startswith('zigzag_high_') or c.startswith('zigzag_low_')]
        df[level_cols] = df_processed[level_cols]
        
        return df

    # Пустые методы
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe