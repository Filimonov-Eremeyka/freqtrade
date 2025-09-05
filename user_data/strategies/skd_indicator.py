import numpy as np
import pandas as pd

class SkdIndicator:
    """
    СКД (Свеча с Контрастным Объемом):
      volume[t] > volume[t-1] и volume[t] > volume[t+1]
    На сигнальной свече (t+2) строим уровни по твоим правилам:
      BUY  после красной СКД:  сигнальная медвежья -> Close; бычья -> Open
      SELL после зелёной СКД:  сигнальная бычья  -> Close; медвежья -> Open
    """
    def compute(self, df: pd.DataFrame, vol_min: float | None = None) -> pd.DataFrame:
        assert {'open','high','low','close','volume'}.issubset(df.columns)

        vol_prev = df['volume'].shift(1)
        vol_next = df['volume'].shift(-1)

        is_skd = (df['volume'] > vol_prev) & (df['volume'] > vol_next)
        if vol_min is not None:
            is_skd &= df['volume'] >= vol_min

        is_green = df['close'] > df['open']
        is_red   = df['close'] < df['open']

        # Направление СКД на t
        skd_dir = np.where(is_skd & is_green,  1.0, np.nan)
        skd_dir = np.where(is_skd & is_red,   -1.0, skd_dir)
        df['skd_dir'] = pd.Series(skd_dir, index=df.index, dtype='float64')

        # На t+2 «видим» СКД
        sig_dir = df['skd_dir'].shift(2)
        df['skd_at_signal'] = sig_dir

        is_bull = df['close'] > df['open']
        is_bear = df['close'] < df['open']

        # BUY после красной СКД
        buy_price = np.where(sig_dir == -1.0,
                             np.where(is_bear, df['close'],
                                      np.where(is_bull, df['open'], np.nan)),
                             np.nan)
        # SELL после зелёной СКД
        sell_price = np.where(sig_dir == 1.0,
                              np.where(is_bull, df['close'],
                                       np.where(is_bear, df['open'], np.nan)),
                              np.nan)

        df['skd_buy_price']  = pd.Series(buy_price,  index=df.index, dtype='float64')
        df['skd_sell_price'] = pd.Series(sell_price, index=df.index, dtype='float64')
        return df
