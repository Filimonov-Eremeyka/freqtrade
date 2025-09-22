import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


__all__ = ["ClaudeSkdIndicator", "ClaudeSkdVisualizer", "SkdTradeManager"]


# ========== СТРУКТУРЫ ДАННЫХ ==========

@dataclass
class _ActiveLevel:
    """Активный уровень СКО"""
    price: float
    ttl: int
    start_idx: int
    side: str  # "buy" | "sell"
    skd_idx: int  # индекс оригинальной СКО свечи
    skd_high: float  # high СКО свечи (для расчета стопа/тейка)
    skd_low: float   # low СКО свечи


@dataclass
class SkdTrade:
    """Информация о сделке"""
    entry_idx: int
    entry_price: float
    side: str  # "buy" | "sell"
    stop_price: float
    take_price: float
    skd_size: float  # размер СКО свечи (high - low)
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "stop" | "take" | "expired"
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


@dataclass
class SkdConfig:
    """Единая конфигурация параметров СКО"""
    # Метод определения СКО
    method: str = "basic"
    vol_min: Optional[float] = None
    
    # Новые фильтры соотношения объемов
    min_vol_ratio_prev: float = 1.2  # volume[t] должен быть больше volume[t-1] в N раз
    min_vol_ratio_next: float = 1.3  # volume[t] должен быть больше volume[t+1] в N раз
    
    # Альтернативные методы
    percentile_lookback: int = 20
    percentile: float = 70.0
    rvol_lookback: int = 20
    rvol_threshold: float = 1.5
    zscore_lookback: int = 20
    zscore_threshold: float = 2.0
    stoch_lookback: int = 14
    stoch_threshold: float = 80.0
    
    # Сигналы и фильтры
    signal_lag_bars: int = 1
    min_distance_mode: str = "pct"
    min_distance_pct: float = 0.0
    atr_period: int = 14
    min_distance_atr_mult: float = 0.0
    filter_ttl_bars: int = 1
    
    # Визуализация
    level_ttl: int = 5
    max_slots: int = 4
    slot_cooldown_bars: int = 1
    expire_opposite_on_new: bool = True
    same_side_latest_only: bool = True
    
    # Торговля
    trade_enabled: bool = True
    max_open_trades: int = 2  # макс. кол-во открытых сделок
    trade_expire_bars: int = 20  # через сколько свечей закрыть сделку принудительно


# ========== ИНДИКАТОР СКО ==========

class ClaudeSkdIndicator:
    """
    Улучшенный индикатор СКО с фильтрами соотношения объемов.
    Интегрированная архитектура без дублирования параметров.
    """
    
    def __init__(self, config: Optional[SkdConfig] = None):
        self.config = config or SkdConfig()
        self.logger = logging.getLogger(__name__)
        self.stats: Dict[str, int] = {
            "total_skd_found": 0,
            "green_skd": 0,
            "red_skd": 0,
            "levels_created": 0,
            "levels_filtered_distance": 0,
            "levels_filtered_ratio": 0,
            "edge_cases_no_neighbors": 0,
        }
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Вычисление СКО и уровней"""
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        # ATR для режима минимальной дистанции
        df["atr"] = self._atr(df, self.config.atr_period)
        
        # 1) Поиск СКО по выбранному методу с улучшенными фильтрами
        is_skd = self._detect_skd_enhanced(df)
        
        # 2) Направление СКО
        skd_dir = np.full(len(df), np.nan, dtype="float64")
        is_green = df["close"] > df["open"]
        is_red = df["close"] < df["open"]
        skd_dir[np.where(is_skd & is_green)] = 1.0
        skd_dir[np.where(is_skd & is_red)] = -1.0
        df["skd_dir"] = skd_dir
        
        # Сохраняем индексы СКО свечей
        df["skd_idx"] = np.where(is_skd, np.arange(len(df)), np.nan)
        
        # 3) Сигнальная свеча = t + lag
        df["skd_at_signal"] = df["skd_dir"].shift(self.config.signal_lag_bars)
        df["skd_idx_at_signal"] = df["skd_idx"].shift(self.config.signal_lag_bars)
        
        # 4) Цены и стороны уровней на сигнальной свече
        df["skd_buy_price"] = np.nan
        df["skd_sell_price"] = np.nan
        df["skd_stop_price"] = np.nan
        df["skd_take_price"] = np.nan
        
        active_levels: List[_ActiveLevel] = []
        
        for i in range(len(df)):
            # Чистим устаревшие уровни
            active_levels = [
                L for L in active_levels 
                if (i - L.start_idx) <= self.config.filter_ttl_bars
            ]
            
            dir_at_sig = df.at[i, "skd_at_signal"]
            if pd.isna(dir_at_sig):
                continue
            
            # Получаем индекс оригинальной СКО свечи
            skd_idx = df.at[i, "skd_idx_at_signal"]
            if pd.isna(skd_idx):
                continue
            skd_idx = int(skd_idx)
            
            side, price = self._side_and_price_on_signal(df, i, float(dir_at_sig))
            if price is None or pd.isna(price):
                continue
            
            # Проверка минимальной дистанции
            if self._distance_enabled():
                same_side = [L for L in active_levels if L.side == side]
                ref_price = df.at[i, "close"]
                atr_val = df.at[i, "atr"]
                if not self._far_enough(price, same_side, ref_price, atr_val):
                    self.stats["levels_filtered_distance"] += 1
                    continue
            
            # Расчет стопа и тейка на основе СКО свечи
            skd_high = df.at[skd_idx, "high"]
            skd_low = df.at[skd_idx, "low"]
            skd_size = skd_high - skd_low
            
            if side == "buy":
                df.at[i, "skd_buy_price"] = float(price)
                df.at[i, "skd_sell_price"] = np.nan
                df.at[i, "skd_stop_price"] = float(skd_low)  # стоп за low СКО
                df.at[i, "skd_take_price"] = float(price + skd_size)  # тейк = вход + размер СКО
            else:
                df.at[i, "skd_sell_price"] = float(price)
                df.at[i, "skd_buy_price"] = np.nan
                df.at[i, "skd_stop_price"] = float(skd_high)  # стоп за high СКО
                df.at[i, "skd_take_price"] = float(price - skd_size)  # тейк = вход - размер СКО
            
            active_levels.append(_ActiveLevel(
                price=float(price),
                ttl=self.config.filter_ttl_bars,
                start_idx=i,
                side=side,
                skd_idx=skd_idx,
                skd_high=float(skd_high),
                skd_low=float(skd_low)
            ))
            self.stats["levels_created"] += 1
        
        # Статистика
        self.stats["total_skd_found"] = int(pd.notna(df["skd_dir"]).sum())
        self.stats["green_skd"] = int((df["skd_dir"] == 1.0).sum())
        self.stats["red_skd"] = int((df["skd_dir"] == -1.0).sum())
        self.logger.info(f"[ClaudeSkd] stats: {self.stats}")
        
        return df
    
    def _detect_skd_enhanced(self, df: pd.DataFrame) -> pd.Series:
        """Улучшенное определение СКО с фильтрами соотношения объемов"""
        vol = df["volume"]
        prev = vol.shift(1)
        nxt = vol.shift(-1)
        
        edge = prev.isna() | nxt.isna()
        if edge.any():
            self.stats["edge_cases_no_neighbors"] = int(edge.sum())
        
        method = (self.config.method or "basic").lower()
        
        # Базовое определение для всех методов
        if method == "basic":
            # Применяем улучшенные фильтры соотношения
            mask = (
                (vol > prev * self.config.min_vol_ratio_prev) & 
                (vol > nxt * self.config.min_vol_ratio_next)
            )
            
            # Подсчет отфильтрованных по соотношению
            basic_mask = (vol > prev) & (vol > nxt)
            self.stats["levels_filtered_ratio"] = int((basic_mask & ~mask).sum())
            
        elif method == "percentile":
            thr = vol.rolling(self.config.percentile_lookback).quantile(self.config.percentile / 100.0)
            mask = vol > thr
            
        elif method == "rvol":
            ma = vol.rolling(self.config.rvol_lookback).mean()
            rvol = vol / ma
            df["skd_rvol"] = rvol
            mask = rvol > self.config.rvol_threshold
            
        elif method == "zscore":
            mean = vol.rolling(self.config.zscore_lookback).mean()
            std = vol.rolling(self.config.zscore_lookback).std()
            z = (vol - mean) / std
            df["skd_vol_zscore"] = z
            mask = z > self.config.zscore_threshold
            
        elif method == "stoch":
            vmin = vol.rolling(self.config.stoch_lookback).min()
            vmax = vol.rolling(self.config.stoch_lookback).max()
            stoch = (vol - vmin) / (vmax - vmin)
            df["skd_vol_stoch"] = stoch * 100.0
            mask = df["skd_vol_stoch"] > self.config.stoch_threshold
            
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
        
        # Дополнительно применяем фильтры соотношения для альтернативных методов
        if method != "basic":
            ratio_mask = (
                (vol > prev * self.config.min_vol_ratio_prev) & 
                (vol > nxt * self.config.min_vol_ratio_next)
            )
            mask = mask & ratio_mask
        
        # Фильтр минимального объема
        if self.config.vol_min is not None:
            mask = mask & (vol >= self.config.vol_min)
        
        return mask.fillna(False)
    
    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        """Расчет ATR"""
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift(1)).abs()
        lc = (df["low"] - df["close"].shift(1)).abs()
        tr = np.maximum(hl, np.maximum(hc, lc))
        return tr.rolling(int(period)).mean()
    
    @staticmethod
    def _side_and_price_on_signal(df: pd.DataFrame, i: int, dir_at_sig: float) -> Tuple[str, Optional[float]]:
        """Определение стороны и цены входа на сигнальной свече"""
        row = df.iloc[i]
        c, o = float(row["close"]), float(row["open"])
        is_bull = c > o
        is_bear = c < o
        
        # Дожи: цена = Close, сторона по СКО
        if not is_bull and not is_bear:
            return ("sell" if dir_at_sig > 0.0 else "buy", c)
        
        if dir_at_sig > 0.0:  # СКО бычья
            if is_bear:  # разворот
                return "sell", c
            else:  # продолжение вверх
                return "buy", o
        else:  # СКО медвежья
            if is_bull:  # разворот вверх
                return "buy", c
            else:  # продолжение вниз
                return "sell", o
    
    def _distance_enabled(self) -> bool:
        """Проверка включен ли фильтр дистанции"""
        if self.config.min_distance_mode == "atr":
            return self.config.min_distance_atr_mult > 0.0
        return self.config.min_distance_pct > 0.0
    
    def _far_enough(
        self,
        new_price: float,
        same_side_levels: List[_ActiveLevel],
        ref_price: float,
        atr_val: float,
    ) -> bool:
        """Проверка минимальной дистанции до существующих уровней"""
        if not self._distance_enabled():
            return True
        
        if self.config.min_distance_mode == "atr" and pd.notna(atr_val):
            min_dist = float(atr_val) * self.config.min_distance_atr_mult
        else:
            min_dist = float(ref_price) * (self.config.min_distance_pct / 100.0)
        
        for L in same_side_levels:
            if abs(new_price - L.price) < min_dist:
                return False
        return True


# ========== МЕНЕДЖЕР СДЕЛОК ==========

class SkdTradeManager:
    """
    Управление сделками на основе СКО сигналов.
    Отслеживание входов, стопов, тейков и результатов.
    """
    
    def __init__(self, config: Optional[SkdConfig] = None):
        self.config = config or SkdConfig()
        self.logger = logging.getLogger(__name__)
        self.trades: List[SkdTrade] = []
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "expired_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
        }
    
    def process_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка сделок и добавление торговых сигналов"""
        if not self.config.trade_enabled:
            return df
        
        # Инициализация колонок для торговых сигналов
        df["trade_enter_long"] = 0
        df["trade_enter_short"] = 0
        df["trade_exit_long"] = 0
        df["trade_exit_short"] = 0
        df["trade_stop_long"] = np.nan
        df["trade_stop_short"] = np.nan
        df["trade_take_long"] = np.nan
        df["trade_take_short"] = np.nan
        
        open_trades: List[SkdTrade] = []
        
        for i in range(len(df)):
            # Проверка выхода из открытых сделок
            for trade in open_trades[:]:
                exit_reason, exit_price = self._check_exit(df, i, trade)
                
                if exit_reason:
                    trade.exit_idx = i
                    trade.exit_price = exit_price
                    trade.exit_reason = exit_reason
                    
                    # Расчет P&L
                    if trade.side == "buy":
                        trade.pnl = exit_price - trade.entry_price
                        df.at[i, "trade_exit_long"] = 1
                    else:
                        trade.pnl = trade.entry_price - exit_price
                        df.at[i, "trade_exit_short"] = 1
                    
                    trade.pnl_pct = (trade.pnl / trade.entry_price) * 100
                    
                    # Обновление статистики
                    self.stats["total_pnl"] += trade.pnl
                    if trade.pnl > 0:
                        self.stats["winning_trades"] += 1
                    elif trade.pnl < 0:
                        self.stats["losing_trades"] += 1
                    if exit_reason == "expired":
                        self.stats["expired_trades"] += 1
                    
                    self.trades.append(trade)
                    open_trades.remove(trade)
            
            # Проверка новых входов (если не превышен лимит открытых сделок)
            if len(open_trades) < self.config.max_open_trades:
                # Вход в long
                buy_price = df.at[i, "skd_buy_price"]
                if pd.notna(buy_price):
                    stop_price = df.at[i, "skd_stop_price"]
                    take_price = df.at[i, "skd_take_price"]
                    
                    if pd.notna(stop_price) and pd.notna(take_price):
                        trade = SkdTrade(
                            entry_idx=i,
                            entry_price=float(buy_price),
                            side="buy",
                            stop_price=float(stop_price),
                            take_price=float(take_price),
                            skd_size=abs(take_price - buy_price)
                        )
                        open_trades.append(trade)
                        df.at[i, "trade_enter_long"] = 1
                        df.at[i, "trade_stop_long"] = float(stop_price)
                        df.at[i, "trade_take_long"] = float(take_price)
                        self.stats["total_trades"] += 1
                
                # Вход в short
                sell_price = df.at[i, "skd_sell_price"]
                if pd.notna(sell_price) and len(open_trades) < self.config.max_open_trades:
                    stop_price = df.at[i, "skd_stop_price"]
                    take_price = df.at[i, "skd_take_price"]
                    
                    if pd.notna(stop_price) and pd.notna(take_price):
                        trade = SkdTrade(
                            entry_idx=i,
                            entry_price=float(sell_price),
                            side="sell",
                            stop_price=float(stop_price),
                            take_price=float(take_price),
                            skd_size=abs(sell_price - take_price)
                        )
                        open_trades.append(trade)
                        df.at[i, "trade_enter_short"] = 1
                        df.at[i, "trade_stop_short"] = float(stop_price)
                        df.at[i, "trade_take_short"] = float(take_price)
                        self.stats["total_trades"] += 1
        
        # Расчет итоговой статистики
        if self.stats["total_trades"] > 0:
            self.stats["win_rate"] = (
                self.stats["winning_trades"] / self.stats["total_trades"] * 100
            )
        
        self.logger.info(f"[TradeManager] {self.stats}")
        return df
    
    def _check_exit(self, df: pd.DataFrame, i: int, trade: SkdTrade) -> Tuple[Optional[str], Optional[float]]:
        """Проверка условий выхода из сделки"""
        row = df.iloc[i]
        
        # Проверка истечения времени
        if i - trade.entry_idx >= self.config.trade_expire_bars:
            return "expired", float(row["close"])
        
        # Для long позиций
        if trade.side == "buy":
            # Проверка стопа
            if row["low"] <= trade.stop_price:
                return "stop", trade.stop_price
            # Проверка тейка
            if row["high"] >= trade.take_price:
                return "take", trade.take_price
        
        # Для short позиций
        else:
            # Проверка стопа
            if row["high"] >= trade.stop_price:
                return "stop", trade.stop_price
            # Проверка тейка
            if row["low"] <= trade.take_price:
                return "take", trade.take_price
        
        return None, None


# ========== ВИЗУАЛИЗАТОР ==========

class ClaudeSkdVisualizer:
    """
    Визуализация уровней СКО и сделок.
    Интегрированная работа с единой конфигурацией.
    """
    
    def __init__(self, config: Optional[SkdConfig] = None):
        self.config = config or SkdConfig()
        self.logger = logging.getLogger(__name__)
    
    def render_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Отрисовка горизонтальных линий уровней"""
        # Маркеры старта
        df["skd_buy_marker"] = df["skd_buy_price"]
        df["skd_sell_marker"] = df["skd_sell_price"]
        
        # Подготовка слотов для линий
        for s in range(self.config.max_slots):
            df[f"skd_buy_signal_line_{s}"] = np.nan
            df[f"skd_sell_signal_line_{s}"] = np.nan
        
        active_buy: Dict[int, _ActiveLevel] = {}
        active_sell: Dict[int, _ActiveLevel] = {}
        last_used_idx_buy: Dict[int, int] = {}
        last_used_idx_sell: Dict[int, int] = {}
        
        buy_idx = {
            s: df.columns.get_loc(f"skd_buy_signal_line_{s}") 
            for s in range(self.config.max_slots)
        }
        sell_idx = {
            s: df.columns.get_loc(f"skd_sell_signal_line_{s}") 
            for s in range(self.config.max_slots)
        }
        
        def pick_slot(
            active: Dict[int, _ActiveLevel], 
            last_used_map: Dict[int, int], 
            i_curr: int
        ) -> int:
            # Поиск свободного слота с учетом cooldown
            for s in range(self.config.max_slots):
                if s in active:
                    continue
                last_i = last_used_map.get(s, -10**9)
                if i_curr - last_i > self.config.slot_cooldown_bars:
                    return s
            
            # Любой свободный слот
            for s in range(self.config.max_slots):
                if s not in active:
                    return s
            
            # Вытеснение самого старого
            return min(active.keys(), key=lambda k: active[k].ttl)
        
        for i in range(len(df)):
            # Продление BUY линий
            for s, L in list(active_buy.items()):
                df.iat[i, buy_idx[s]] = L.price
                L.ttl -= 1
                if L.ttl <= 0:
                    del active_buy[s]
                    last_used_idx_buy[s] = i
            
            # Продление SELL линий
            for s, L in list(active_sell.items()):
                df.iat[i, sell_idx[s]] = L.price
                L.ttl -= 1
                if L.ttl <= 0:
                    del active_sell[s]
                    last_used_idx_sell[s] = i
            
            # Старт новой BUY линии
            bp = df.at[i, "skd_buy_price"]
            if pd.notna(bp):
                # Гасим противоположные
                if self.config.expire_opposite_on_new and active_sell:
                    for s in list(active_sell.keys()):
                        del active_sell[s]
                        last_used_idx_sell[s] = i
                
                # Оставляем только последний той же стороны
                if self.config.same_side_latest_only and active_buy:
                    for s in list(active_buy.keys()):
                        del active_buy[s]
                        last_used_idx_buy[s] = i
                
                s = pick_slot(active_buy, last_used_idx_buy, i)
                
                # Получаем данные СКО свечи
                skd_idx = df.at[i, "skd_idx_at_signal"]
                if pd.notna(skd_idx):
                    skd_idx = int(skd_idx)
                    active_buy[s] = _ActiveLevel(
                        price=float(bp),
                        ttl=self.config.level_ttl,
                        start_idx=i,
                        side="buy",
                        skd_idx=skd_idx,
                        skd_high=df.at[skd_idx, "high"],
                        skd_low=df.at[skd_idx, "low"]
                    )
            
            # Старт новой SELL линии
            sp = df.at[i, "skd_sell_price"]
            if pd.notna(sp):
                # Гасим противоположные
                if self.config.expire_opposite_on_new and active_buy:
                    for s in list(active_buy.keys()):
                        del active_buy[s]
                        last_used_idx_buy[s] = i
                
                # Оставляем только последний той же стороны
                if self.config.same_side_latest_only and active_sell:
                    for s in list(active_sell.keys()):
                        del active_sell[s]
                        last_used_idx_sell[s] = i
                
                s = pick_slot(active_sell, last_used_idx_sell, i)
                
                # Получаем данные СКО свечи
                skd_idx = df.at[i, "skd_idx_at_signal"]
                if pd.notna(skd_idx):
                    skd_idx = int(skd_idx)
                    active_sell[s] = _ActiveLevel(
                        price=float(sp),
                        ttl=self.config.level_ttl,
                        start_idx=i,
                        side="sell",
                        skd_idx=skd_idx,
                        skd_high=df.at[skd_idx, "high"],
                        skd_low=df.at[skd_idx, "low"]
                    )
        
        # Статистика линий
        cols = [
            c for c in df.columns 
            if c.startswith("skd_buy_signal_line_") or c.startswith("skd_sell_signal_line_")
        ]
        summary = {c: int(pd.notna(df[c]).sum()) for c in cols}
        self.logger.info(f"[Visualizer] lines summary: {summary}")
        
        return df