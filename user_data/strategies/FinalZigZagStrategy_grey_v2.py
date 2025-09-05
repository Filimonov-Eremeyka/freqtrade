
# user_data/strategies/FinalZigZagStrategy_grey_v2.py
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas as pd
import numpy as np

# Prefer the user's zigzag implementation
try:
    from zigzag_indicator import zigzag
except Exception:
    zigzag = None

class FinalZigZagStrategy_grey_v2(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    minimal_roi = {'0': 10}
    stoploss = -0.99  # viz only
    use_custom_stoploss = False
    process_only_new_candles = False
    startup_candle_count = 200

    # All-gray visualization
    plot_config = {
        'main_plot': {
            'zigzag_peak':   {'type': 'scatter', 'plotly': {'mode': 'markers', 'marker': {'symbol': 'diamond', 'size': 9, 'color': '#888888'}}},
            'zigzag_trough': {'type': 'scatter', 'plotly': {'mode': 'markers', 'marker': {'symbol': 'diamond', 'size': 9, 'color': '#888888'}}},
            'zigzag_high_0': {'type': 'scatter', 'plotly': {'mode': 'lines', 'line': {'color': '#888888', 'width': 1}}},
            'zigzag_high_1': {'type': 'scatter', 'plotly': {'mode': 'lines', 'line': {'color': '#888888', 'width': 1}}},
            'zigzag_high_2': {'type': 'scatter', 'plotly': {'mode': 'lines', 'line': {'color': '#888888', 'width': 1}}},
            'zigzag_high_3': {'type': 'scatter', 'plotly': {'mode': 'lines', 'line': {'color': '#888888', 'width': 1}}},
            'zigzag_low_0':  {'type': 'scatter', 'plotly': {'mode': 'lines', 'line': {'color': '#888888', 'width': 1}}},
            'zigzag_low_1':  {'type': 'scatter', 'plotly': {'mode': 'lines', 'line': {'color': '#888888', 'width': 1}}},
            'zigzag_low_2':  {'type': 'scatter', 'plotly': {'mode': 'lines', 'line': {'color': '#888888', 'width': 1}}},
            'zigzag_low_3':  {'type': 'scatter', 'plotly': {'mode': 'lines', 'line': {'color': '#888888', 'width': 1}}},
        }
    }

    def _safe_zigzag(self, df: DataFrame, peak_pct: float, trough_pct: float):
        """Return (peaks, troughs) series regardless of the underlying zigzag() shape."""
        if callable(zigzag):
            zz = zigzag(df, peak_pct=peak_pct, trough_pct=trough_pct)
            # The user's zigzag returns a tuple (peaks, troughs).
            if isinstance(zz, tuple) and len(zz) >= 2:
                peaks, troughs = zz[0], zz[1]
                # ensure Series with index
                if not isinstance(peaks, pd.Series):
                    peaks = pd.Series(peaks, index=df.index, dtype='float64')
                if not isinstance(troughs, pd.Series):
                    troughs = pd.Series(troughs, index=df.index, dtype='float64')
                return peaks, troughs
            # Some implementations may return a DataFrame
            if isinstance(zz, pd.DataFrame):
                peaks = zz.get('zigzag_peak', pd.Series(index=df.index, dtype='float64'))
                troughs = zz.get('zigzag_trough', pd.Series(index=df.index, dtype='float64'))
                return peaks, troughs

        # Fallback: naive detector based on percentage moves
        peaks = pd.Series(index=df.index, dtype='float64')
        troughs = pd.Series(index=df.index, dtype='float64')
        last = df['close'].iloc[0]
        state = 0  # 1 looking for peak, -1 trough
        pivot_idx = 0
        pivot_price = df['close'].iloc[0]
        for i in range(1, len(df)):
            c = df['close'].iloc[i]
            change = (c - pivot_price) / pivot_price if pivot_price != 0 else 0.0
            if state >= 0 and change >= peak_pct:
                peaks.iloc[i] = c
                pivot_price = c
                state = 1
            elif state <= 0 and change <= -trough_pct:
                troughs.iloc[i] = c
                pivot_price = c
                state = -1
            last = c
        return peaks, troughs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        # 1) Compute zigzag pivots (robust to different zigzag() APIs)
        peaks, troughs = self._safe_zigzag(df, peak_pct=0.01, trough_pct=0.01)
        df['zigzag_peak'] = peaks
        df['zigzag_trough'] = troughs

        # 2) Build up to 4 concurrent horizontal levels that stop on touch
        max_levels = 4
        for i in range(max_levels):
            df[f'zigzag_high_{i}'] = np.nan
            df[f'zigzag_low_{i}'] = np.nan

        active_high = {}
        active_low = {}
        eps = 1e-6

        for i in range(len(df)):
            hi = float(df['high'].iloc[i] if 'high' in df.columns else df['close'].iloc[i])
            lo = float(df['low'].iloc[i] if 'low' in df.columns else df['close'].iloc[i])

            # End lines on wick touch
            to_del = []
            for lvl, slot in active_high.items():
                if hi >= lvl * (1 - eps):
                    to_del.append(lvl)
                else:
                    df.iloc[i, df.columns.get_loc(f'zigzag_high_{slot}')] = lvl
            for lvl in to_del:
                active_high.pop(lvl, None)

            to_del = []
            for lvl, slot in active_low.items():
                if lo <= lvl * (1 + eps):
                    to_del.append(lvl)
                else:
                    df.iloc[i, df.columns.get_loc(f'zigzag_low_{slot}')] = lvl
            for lvl in to_del:
                active_low.pop(lvl, None)

            # Add new pivot levels
            if not pd.isna(df['zigzag_peak'].iloc[i]):
                used = set(active_high.values())
                for slot in range(max_levels):
                    if slot not in used:
                        active_high[float(df['zigzag_peak'].iloc[i])] = slot
                        break
            if not pd.isna(df['zigzag_trough'].iloc[i]):
                used = set(active_low.values())
                for slot in range(max_levels):
                    if slot not in used:
                        active_low[float(df['zigzag_trough'].iloc[i])] = slot
                        break

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe
