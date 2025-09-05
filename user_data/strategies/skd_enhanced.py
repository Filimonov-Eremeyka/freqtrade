import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


__all__ = ["EnhancedSkdIndicator", "SkdLevelVisualizer"]


# --------- внутренние структуры ----------

@dataclass
class _ActiveLevel:
    price: float
    ttl: int
    start_idx: int
    side: str  # "buy" | "sell"


# --------- СКО индикатор ----------

class EnhancedSkdIndicator:
    """
    Индикатор СКО (свеча контрастного объёма).
    Базовое определение: volume[t] > volume[t-1] и volume[t] > volume[t+1].
    Сигнал = t + signal_lag_bars (по ТЗ — 1).

    Правило уровня на сигнальной свече s (t+1):
      - СКО БЫЧЬЯ (+1):
          • сигнальная МЕДВЕЖЬЯ → SELL @ Close[s]  (разворот)
          • сигнальная БЫЧЬЯ   → BUY  @ Open[s]   (продолжение)
      - СКО МЕДВЕЖЬЯ (−1):
          • сигнальная БЫЧЬЯ   → BUY  @ Close[s]
          • сигнальная МЕДВЕЖЬЯ → SELL @ Open[s]
      - Дожи → цена = Close[s].
    """

    def __init__(
        self,
        *,
        # метод контрастности (по умолчанию базовый, без «умных» фильтров)
        method: str = "basic",            # "basic" | "percentile" | "rvol" | "zscore" | "stoch"
        vol_min: Optional[float] = None,  # None = не фильтровать по минимальному объёму

        # параметры альтернатив (на будущее)
        percentile_lookback: int = 20,
        percentile: float = 70.0,
        rvol_lookback: int = 20,
        rvol_threshold: float = 1.5,
        zscore_lookback: int = 20,
        zscore_threshold: float = 2.0,
        stoch_lookback: int = 14,
        stoch_threshold: float = 80.0,

        # лаг сигнала
        signal_lag_bars: int = 1,

        # контроль минимальной дистанции (ПО УМОЛЧАНИЮ ВЫКЛ.)
        min_distance_mode: str = "pct",   # "pct" | "atr"
        min_distance_pct: float = 0.0,    # 0.0% → выкл.
        atr_period: int = 14,
        min_distance_atr_mult: float = 0.0,  # 0*ATR → выкл.
        filter_ttl_bars: int = 1,         # окно контроля
    ):
        self.method = method
        self.vol_min = vol_min

        self.percentile_lookback = percentile_lookback
        self.percentile = percentile
        self.rvol_lookback = rvol_lookback
        self.rvol_threshold = rvol_threshold
        self.zscore_lookback = zscore_lookback
        self.zscore_threshold = zscore_threshold
        self.stoch_lookback = stoch_lookback
        self.stoch_threshold = stoch_threshold

        self.signal_lag_bars = int(signal_lag_bars)

        self.min_distance_mode = min_distance_mode
        self.min_distance_pct = float(min_distance_pct)
        self.atr_period = int(atr_period)
        self.min_distance_atr_mult = float(min_distance_atr_mult)
        self.filter_ttl_bars = int(filter_ttl_bars)

        self.logger = logging.getLogger(__name__)
        self.stats: Dict[str, int] = {
            "total_skd_found": 0,
            "green_skd": 0,
            "red_skd": 0,
            "levels_created": 0,
            "levels_filtered_distance": 0,
            "edge_cases_no_neighbors": 0,
        }

    # ---------- публичный API ----------

    def compute(self, df_in: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df_in.columns):
            missing = required - set(df_in.columns)
            raise ValueError(f"Missing required columns: {missing}")

        df = df_in.copy().reset_index(drop=True)

        # ATR на будущее (для режима 'atr')
        df["atr"] = self._atr(df, self.atr_period)

        # 1) поиск СКО по выбранному методу
        is_skd = self._detect_skd(df)

        # 2) направление СКО
        skd_dir = np.full(len(df), np.nan, dtype="float64")
        is_green = df["close"] > df["open"]
        is_red = df["close"] < df["open"]
        skd_dir[np.where(is_skd & is_green)] = 1.0
        skd_dir[np.where(is_skd & is_red)] = -1.0
        df["skd_dir"] = skd_dir

        # 3) сигнальная свеча = t + lag
        df["skd_at_signal"] = df["skd_dir"].shift(self.signal_lag_bars)

        # 4) цены и стороны уровней на сигнальной свече
        df["skd_buy_price"] = np.nan
        df["skd_sell_price"] = np.nan

        active_levels: List[_ActiveLevel] = []  # для дистанции (сейчас выкл.)

        for i in range(len(df)):
            # чистим устаревшие для окна контроля
            active_levels = [L for L in active_levels if (i - L.start_idx) <= self.filter_ttl_bars]

            dir_at_sig = df.at[i, "skd_at_signal"]
            if pd.isna(dir_at_sig):
                continue

            side, price = self._side_and_price_on_signal(df, i, float(dir_at_sig))
            if price is None or pd.isna(price):
                continue

            # минимальная дистанция — по умолчанию выкл.
            if self._distance_enabled():
                same_side = [L for L in active_levels if L.side == side]
                ref_price = df.at[i, "close"]
                atr_val = df.at[i, "atr"]
                if not self._far_enough(price, same_side, ref_price, atr_val):
                    self.stats["levels_filtered_distance"] += 1
                    continue

            if side == "buy":
                df.at[i, "skd_buy_price"] = float(price)
                df.at[i, "skd_sell_price"] = np.nan   # гарантируем одну сторону
            else:
                df.at[i, "skd_sell_price"] = float(price)
                df.at[i, "skd_buy_price"] = np.nan    # гарантируем одну сторону

            active_levels.append(_ActiveLevel(price=float(price), ttl=self.filter_ttl_bars, start_idx=i, side=side))
            self.stats["levels_created"] += 1

        # диагностика
        self.stats["total_skd_found"] = int(pd.notna(df["skd_dir"]).sum())
        self.stats["green_skd"] = int((df["skd_dir"] == 1.0).sum())
        self.stats["red_skd"] = int((df["skd_dir"] == -1.0).sum())
        self.logger.info(f"[SKD] stats: {self.stats}")

        return df

    # ---------- вспомогательные ----------

    def _detect_skd(self, df: pd.DataFrame) -> pd.Series:
        vol = df["volume"]
        prev = vol.shift(1)
        nxt = vol.shift(-1)

        edge = prev.isna() | nxt.isna()
        if edge.any():
            self.stats["edge_cases_no_neighbors"] = int(edge.sum())

        method = (self.method or "basic").lower()

        if method == "basic":
            mask = (vol > prev) & (vol > nxt)

        elif method == "percentile":
            thr = vol.rolling(self.percentile_lookback).quantile(self.percentile / 100.0)
            mask = vol > thr

        elif method == "rvol":
            ma = vol.rolling(self.rvol_lookback).mean()
            rvol = vol / ma
            df["skd_rvol"] = rvol
            mask = rvol > self.rvol_threshold

        elif method == "zscore":
            mean = vol.rolling(self.zscore_lookback).mean()
            std = vol.rolling(self.zscore_lookback).std()
            z = (vol - mean) / std
            df["skd_vol_zscore"] = z
            mask = z > self.zscore_threshold

        elif method == "stoch":
            vmin = vol.rolling(self.stoch_lookback).min()
            vmax = vol.rolling(self.stoch_lookback).max()
            stoch = (vol - vmin) / (vmax - vmin)
            df["skd_vol_stoch"] = stoch * 100.0
            mask = df["skd_vol_stoch"] > self.stoch_threshold

        else:
            raise ValueError(f"Unknown method: {self.method}")

        if self.vol_min is not None:
            mask = mask & (vol >= self.vol_min)

        return mask.fillna(False)

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift(1)).abs()
        lc = (df["low"] - df["close"].shift(1)).abs()
        tr = np.maximum(hl, np.maximum(hc, lc))
        return tr.rolling(int(period)).mean()

    @staticmethod
    def _side_and_price_on_signal(df: pd.DataFrame, i: int, dir_at_sig: float) -> Tuple[str, Optional[float]]:
        """
        Возвращает (side, price) на сигнальной свече i.
        dir_at_sig: +1 (СКО была бычья), -1 (СКО была медвежья)
        """
        row = df.iloc[i]
        c, o = float(row["close"]), float(row["open"])
        is_bull = c > o
        is_bear = c < o

        # дожи: цена = Close; сторона — по СКО
        if not is_bull and not is_bear:
            return ("sell" if dir_at_sig > 0.0 else "buy", c)

        if dir_at_sig > 0.0:        # СКО бычья
            if is_bear:             # разворот
                return "sell", c
            else:                   # продолжение вверх
                return "buy", o
        else:                        # СКО медвежья
            if is_bull:             # разворот вверх
                return "buy", c
            else:                   # продолжение вниз
                return "sell", o

    def _distance_enabled(self) -> bool:
        if self.min_distance_mode == "atr":
            return self.min_distance_atr_mult > 0.0
        return self.min_distance_pct > 0.0

    def _far_enough(
        self,
        new_price: float,
        same_side_levels: List[_ActiveLevel],
        ref_price: float,
        atr_val: float,
    ) -> bool:
        if not self._distance_enabled():
            return True

        if self.min_distance_mode == "atr" and pd.notna(atr_val):
            min_dist = float(atr_val) * self.min_distance_atr_mult
        else:
            min_dist = float(ref_price) * (self.min_distance_pct / 100.0)

        for L in same_side_levels:
            if abs(new_price - L.price) < min_dist:
                return False
        return True


# --------- визуализатор линий ----------

class SkdLevelVisualizer:
    """
    Превращает точки 'skd_buy_price' / 'skd_sell_price' в горизонтальные отрезки (слоты против «склейки»).
    Дополнительно:
      - slot_cooldown_bars: слот нельзя переиспользовать на соседней свече (гарантированный разрыв);
      - expire_opposite_on_new: при появлении новой стороны гасим противоположные активные уровни;
      - same_side_latest_only: держим только последний уровень одной стороны (старые сразу снимаем).
    """

    def __init__(
        self,
        *,
        max_slots: int = 4,
        level_ttl: int = 5,
        slot_cooldown_bars: int = 1,
        expire_opposite_on_new: bool = True,
        same_side_latest_only: bool = True,
    ):
        self.max_slots = int(max_slots)
        self.level_ttl = int(level_ttl)
        self.slot_cooldown_bars = int(slot_cooldown_bars)
        self.expire_opposite_on_new = bool(expire_opposite_on_new)
        self.same_side_latest_only = bool(same_side_latest_only)
        self.logger = logging.getLogger(__name__)

    def render_levels(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy().reset_index(drop=True)

        # маркеры старта
        df["skd_buy_marker"] = df["skd_buy_price"]
        df["skd_sell_marker"] = df["skd_sell_price"]

        # подготовим слоты
        for s in range(self.max_slots):
            df[f"skd_buy_signal_line_{s}"] = np.nan
            df[f"skd_sell_signal_line_{s}"] = np.nan
            df[f"skd_buy_signal_line_{s}"] = df[f"skd_buy_signal_line_{s}"].astype("float64")
            df[f"skd_sell_signal_line_{s}"] = df[f"skd_sell_signal_line_{s}"].astype("float64")

        active_buy: Dict[int, _ActiveLevel] = {}
        active_sell: Dict[int, _ActiveLevel] = {}
        last_used_idx_buy: Dict[int, int] = {}
        last_used_idx_sell: Dict[int, int] = {}

        buy_idx = {s: df.columns.get_loc(f"skd_buy_signal_line_{s}") for s in range(self.max_slots)}
        sell_idx = {s: df.columns.get_loc(f"skd_sell_signal_line_{s}") for s in range(self.max_slots)}

        def pick_slot(active: Dict[int, _ActiveLevel], last_used_map: Dict[int, int], i_curr: int) -> int:
            # 1) свободный и не под cooldown
            for s in range(self.max_slots):
                if s in active:
                    continue
                last_i = last_used_map.get(s, -10**9)
                if i_curr - last_i > self.slot_cooldown_bars:
                    return s
            # 2) любой свободный
            for s in range(self.max_slots):
                if s not in active:
                    return s
            # 3) вытеснить самый старый
            return min(active.keys(), key=lambda k: active[k].ttl)

        for i in range(len(df)):
            # продлеваем BUY
            for s, L in list(active_buy.items()):
                df.iat[i, buy_idx[s]] = L.price
                L.ttl -= 1
                if L.ttl <= 0:
                    del active_buy[s]
                    last_used_idx_buy[s] = i  # слот освободился здесь

            # продлеваем SELL
            for s, L in list(active_sell.items()):
                df.iat[i, sell_idx[s]] = L.price
                L.ttl -= 1
                if L.ttl <= 0:
                    del active_sell[s]
                    last_used_idx_sell[s] = i  # слот освободился здесь

            # старт BUY
            bp = df.at[i, "skd_buy_price"]
            if pd.notna(bp):
                # 1) при старте BUY — по желанию гасим все SELL
                if self.expire_opposite_on_new and active_sell:
                    for s in list(active_sell.keys()):
                        del active_sell[s]
                        last_used_idx_sell[s] = i  # остановились на текущем баре

                # 2) по желанию держим только последний BUY — чистим старые BUY
                if self.same_side_latest_only and active_buy:
                    for s in list(active_buy.keys()):
                        del active_buy[s]
                        last_used_idx_buy[s] = i

                s = pick_slot(active_buy, last_used_idx_buy, i)
                active_buy[s] = _ActiveLevel(price=float(bp), ttl=self.level_ttl, start_idx=i, side="buy")

            # старт SELL
            sp = df.at[i, "skd_sell_price"]
            if pd.notna(sp):
                # 1) при старте SELL — по желанию гасим все BUY
                if self.expire_opposite_on_new and active_buy:
                    for s in list(active_buy.keys()):
                        del active_buy[s]
                        last_used_idx_buy[s] = i

                # 2) по желанию держим только последний SELL — чистим старые SELL
                if self.same_side_latest_only and active_sell:
                    for s in list(active_sell.keys()):
                        del active_sell[s]
                        last_used_idx_sell[s] = i

                s = pick_slot(active_sell, last_used_idx_sell, i)
                active_sell[s] = _ActiveLevel(price=float(sp), ttl=self.level_ttl, start_idx=i, side="sell")

        # сводка
        cols = [c for c in df.columns if c.startswith("skd_buy_signal_line_") or c.startswith("skd_sell_signal_line_")]
        summary = {c: int(pd.notna(df[c]).sum()) for c in cols}
        self.logger.info(f"[SKD-plot] non-empty points per line: {summary}")

        return df
