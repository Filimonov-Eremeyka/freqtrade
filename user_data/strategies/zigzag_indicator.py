# user_data/strategies/zigzag_indicator.py
import numpy as np
import pandas as pd
from pandas import DataFrame

def zigzag(df: DataFrame, peak_pct=0.02, trough_pct=0.02):
    """
    Надежная реализация ZigZag, адаптированная из источников сообщества.
    Возвращает DataFrame с точечными сигналами вершин и впадин.
    """
    peaks = pd.Series(index=df.index, dtype='float64')
    troughs = pd.Series(index=df.index, dtype='float64')
    
    last_pivot_price = 0
    last_pivot_idx = 0
    direction = 0  # 1 for peak, -1 for trough

    for i in df.index:
        price = df.at[i, 'high'] if direction >= 0 else df.at[i, 'low']

        if direction == 0:  # Start of search
            if last_pivot_price == 0:
                last_pivot_price = df.at[i, 'close']
                last_pivot_idx = i
                continue
            
            if price > last_pivot_price * (1 + peak_pct):
                direction = 1
            elif price < last_pivot_price * (1 - trough_pct):
                direction = -1
        
        elif direction == 1:  # Searching for a peak
            if price > last_pivot_price:
                last_pivot_price = price
                last_pivot_idx = i
            elif price < last_pivot_price * (1 - trough_pct):
                peaks[last_pivot_idx] = last_pivot_price
                direction = -1
                last_pivot_price = price
                last_pivot_idx = i

        elif direction == -1:  # Searching for a trough
            if price < last_pivot_price:
                last_pivot_price = price
                last_pivot_idx = i
            elif price > last_pivot_price * (1 + peak_pct):
                troughs[last_pivot_idx] = last_pivot_price
                direction = 1
                last_pivot_price = price
                last_pivot_idx = i

    return peaks, troughs