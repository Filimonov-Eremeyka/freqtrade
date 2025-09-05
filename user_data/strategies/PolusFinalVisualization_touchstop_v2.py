
# user_data/strategies/PolusFinalVisualization_touchstop_v2.py
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas as pd
import numpy as np

# Try to import user's PolusLevels, but don't rely on its internals.
try:
    from polus_levels import PolusLevels as _ExternalPolusLevels
except Exception:
    _ExternalPolusLevels = None

class PolusFinalVisualization_touchstop_v2(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    minimal_roi = {'0': 10}
    stoploss = -0.99
    use_custom_stoploss = False
    process_only_new_candles = False
    startup_candle_count = 200

    # Neutral gray viz
    plot_config = {
        'main_plot': {
            'polus_high_signal': {'type': 'scatter', 'plotly': {'mode': 'markers', 'marker': {'symbol': 'x', 'size': 8, 'color': '#888888'}}},
            'polus_low_signal':  {'type': 'scatter', 'plotly': {'mode': 'markers', 'marker': {'symbol': 'x', 'size': 8, 'color': '#888888'}}},
            # Dynamically add 6 level lines for high/low
            **{f'level_high_{i}': {'type': 'scatter', 'plotly': {'mode': 'lines', 'line': {'color': '#888888', 'width': 1}}} for i in range(6)},
            **{f'level_low_{i}':  {'type': 'scatter', 'plotly': {'mode': 'lines', 'line': {'color': '#888888', 'width': 1}}} for i in range(6)},
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If external class exists, instantiate it. We'll still compute signals internally.
        self._pl = _ExternalPolusLevels(use_tick_volume=False) if _ExternalPolusLevels else None

    @staticmethod
    def _compute_polus_signals(df: DataFrame, use_tick_volume: bool = False):
        """Self-contained signal detection compatible with the user's logic.
        Returns 3 numpy arrays: high_levels, low_levels, close_ref (all length len(df))."""
        high = df['high'].to_numpy(copy=False) if 'high' in df.columns else df['close'].to_numpy(copy=False)
        low  = df['low'].to_numpy(copy=False)  if 'low' in df.columns  else df['close'].to_numpy(copy=False)
        close = df['close'].to_numpy(copy=False)
        volcol = 'tick_volume' if use_tick_volume and 'tick_volume' in df.columns else ('volume' if 'volume' in df.columns else None)
        vol = df[volcol].to_numpy(copy=False) if volcol else np.ones_like(close)

        hi_out = np.full(len(df), np.nan, dtype='float64')
        lo_out = np.full(len(df), np.nan, dtype='float64')
        cl_out = np.full(len(df), np.nan, dtype='float64')

        for i in range(1, len(df) - 1):
            if vol[i - 1] > vol[i] < vol[i + 1]:
                hi_out[i] = high[i]
                lo_out[i] = low[i]
                cl_out[i] = close[i - 1]
        return hi_out, lo_out, cl_out

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        # Use internal computation (robust). If external is present and has compute(), we could cross-check.
        hi, lo, cl = self._compute_polus_signals(df, use_tick_volume=False)
        df['polus_high_signal'] = pd.Series(hi, index=df.index)
        df['polus_low_signal']  = pd.Series(lo, index=df.index)
        df['polus_close_ref']   = pd.Series(cl, index=df.index)

        df = self._manage_levels(df, max_levels=6)
        return df

    def _manage_levels(self, dataframe: DataFrame, max_levels: int) -> DataFrame:
        # Pre-create columns
        for slot in range(max_levels):
            dataframe[f'level_high_{slot}'] = np.nan
            dataframe[f'level_low_{slot}'] = np.nan

        active_high = {}  # price -> slot
        active_low = {}

        eps = 1e-6
        for i in range(len(dataframe)):
            hi = float(dataframe.at[i, 'high']) if 'high' in dataframe.columns else float(dataframe.at[i, 'close'])
            lo = float(dataframe.at[i, 'low'])  if 'low'  in dataframe.columns else float(dataframe.at[i, 'close'])

            # draw-until-touch (exclusive): don't write value on the touching bar
            to_remove = []
            for price, slot in active_high.items():
                if hi >= price * (1 - eps):
                    to_remove.append(price)
                else:
                    dataframe.loc[i, f'level_high_{slot}'] = price
            for price in to_remove:
                active_high.pop(price, None)

            to_remove = []
            for price, slot in active_low.items():
                if lo <= price * (1 + eps):
                    to_remove.append(price)
                else:
                    dataframe.loc[i, f'level_low_{slot}'] = price
            for price in to_remove:
                active_low.pop(price, None)

            # add new levels from signals
            new_high = dataframe.at[i, 'polus_high_signal']
            if pd.notna(new_high) and float(new_high) not in active_high:
                used = set(active_high.values())
                for slot in range(max_levels):
                    if slot not in used:
                        active_high[float(new_high)] = slot
                        break

            new_low = dataframe.at[i, 'polus_low_signal']
            if pd.notna(new_low) and float(new_low) not in active_low:
                used = set(active_low.values())
                for slot in range(max_levels):
                    if slot not in used:
                        active_low[float(new_low)] = slot
                        break

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe
