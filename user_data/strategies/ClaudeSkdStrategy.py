import logging
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np

from claude_skd_enhanced import ClaudeSkdIndicator, ClaudeSkdVisualizer, SkdTradeManager, SkdConfig


class ClaudeSkdStrategy(IStrategy):
    """
    Торговая стратегия на основе улучшенного индикатора СКО.
    
    Особенности:
    - Улучшенные фильтры соотношения объемов (1.2x и 1.3x)
    - Торговая логика с соотношением риск/прибыль 1:1
    - Визуализация входов, выходов, стопов и тейков
    - Поддержка разнонаправленных сделок (long и short одновременно)
    - Единая конфигурация без дублирования параметров
    """
    
    INTERFACE_VERSION = 3
    logger = logging.getLogger(__name__)
    
    # ========== ОСНОВНЫЕ ПАРАМЕТРЫ СТРАТЕГИИ ==========
    
    timeframe = "5m"
    startup_candle_count = 100
    
    # Freqtrade требования
    minimal_roi = {"0": 100}  # Отключаем ROI, используем свои тейки
    stoploss = -1.0  # Отключаем глобальный стоплосс, используем свои стопы
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    
    # ========== ПАРАМЕТРЫ СКО (через единую конфигурацию) ==========
    
    # Метод определения СКО
    skd_method: str = "basic"  # "basic" | "zscore" | "percentile" | "rvol" | "stoch"
    skd_vol_min: float | None = None
    
    # НОВЫЕ улучшенные фильтры соотношения объемов
    skd_min_vol_ratio_prev: float = 1.2  # volume[t] > volume[t-1] * 1.2
    skd_min_vol_ratio_next: float = 1.3  # volume[t] > volume[t+1] * 1.3
    
    # Альтернативные методы
    skd_percentile_lookback: int = 20
    skd_percentile: float = 70.0
    skd_rvol_lookback: int = 20
    skd_rvol_threshold: float = 1.5
    skd_zscore_lookback: int = 20
    skd_zscore_threshold: float = 2.0
    skd_stoch_lookback: int = 14
    skd_stoch_threshold: float = 80.0
    
    # Сигналы и фильтры
    skd_signal_lag_bars: int = 1
    skd_min_distance_mode: str = "pct"
    skd_min_distance_pct: float = 0.0  # 0 = выключено
    skd_atr_period: int = 14
    skd_min_distance_atr_mult: float = 0.0  # 0 = выключено
    skd_filter_ttl_bars: int = 1
    
    # Визуализация линий
    skd_level_ttl: int = 5
    skd_max_slots: int = 4
    skd_slot_cooldown_bars: int = 1
    skd_expire_opposite_on_new: bool = True
    skd_same_side_latest_only: bool = True
    
    # Торговые параметры
    skd_trade_enabled: bool = True
    skd_max_open_trades: int = 2  # Макс. кол-во открытых позиций
    skd_trade_expire_bars: int = 20  # Принудительное закрытие через N свечей
    
    # ========== ВИЗУАЛИЗАЦИЯ ==========
    
    # Цвета для графика
    _BUY_COLOR = "#2ecc71"   # зелёный
    _SELL_COLOR = "#e74c3c"  # красный
    _STOP_COLOR = "#f39c12"  # оранжевый
    _TAKE_COLOR = "#3498db"  # синий
    
    _BUY_LINE = {"mode": "lines", "line": {"color": _BUY_COLOR, "width": 2}}
    _SELL_LINE = {"mode": "lines", "line": {"color": _SELL_COLOR, "width": 2}}
    _BUY_MARKER = {"mode": "markers", "marker": {"symbol": "circle", "size": 10, "color": _BUY_COLOR}}
    _SELL_MARKER = {"mode": "markers", "marker": {"symbol": "circle", "size": 10, "color": _SELL_COLOR}}
    
    # Маркеры входов и выходов
    _ENTRY_LONG = {"mode": "markers", "marker": {"symbol": "triangle-up", "size": 12, "color": _BUY_COLOR}}
    _ENTRY_SHORT = {"mode": "markers", "marker": {"symbol": "triangle-down", "size": 12, "color": _SELL_COLOR}}
    _EXIT_MARKER = {"mode": "markers", "marker": {"symbol": "x", "size": 10, "color": "#95a5a6"}}
    
    # Линии стопов и тейков
    _STOP_LINE = {"mode": "lines", "line": {"color": _STOP_COLOR, "dash": "dash", "width": 1}}
    _TAKE_LINE = {"mode": "lines", "line": {"color": _TAKE_COLOR, "dash": "dash", "width": 1}}
    
    plot_config = {
        "main_plot": {
            # Маркеры СКО уровней
            "skd_buy_marker": {"type": "scatter", "plotly": _BUY_MARKER},
            "skd_sell_marker": {"type": "scatter", "plotly": _SELL_MARKER},
            
            # Горизонтальные линии уровней (4 слота для каждой стороны)
            "skd_buy_signal_line_0": {"type": "scatter", "plotly": _BUY_LINE},
            "skd_buy_signal_line_1": {"type": "scatter", "plotly": _BUY_LINE},
            "skd_buy_signal_line_2": {"type": "scatter", "plotly": _BUY_LINE},
            "skd_buy_signal_line_3": {"type": "scatter", "plotly": _BUY_LINE},
            
            "skd_sell_signal_line_0": {"type": "scatter", "plotly": _SELL_LINE},
            "skd_sell_signal_line_1": {"type": "scatter", "plotly": _SELL_LINE},
            "skd_sell_signal_line_2": {"type": "scatter", "plotly": _SELL_LINE},
            "skd_sell_signal_line_3": {"type": "scatter", "plotly": _SELL_LINE},
            
            # Торговые сигналы - входы
            "entry_long_signal": {"type": "scatter", "plotly": _ENTRY_LONG},
            "entry_short_signal": {"type": "scatter", "plotly": _ENTRY_SHORT},
            
            # Торговые сигналы - выходы
            "exit_long_signal": {"type": "scatter", "plotly": _EXIT_MARKER},
            "exit_short_signal": {"type": "scatter", "plotly": _EXIT_MARKER},
            
            # Линии стопов и тейков
            "stop_loss_long": {"type": "scatter", "plotly": _STOP_LINE},
            "stop_loss_short": {"type": "scatter", "plotly": _STOP_LINE},
            "take_profit_long": {"type": "scatter", "plotly": _TAKE_LINE},
            "take_profit_short": {"type": "scatter", "plotly": _TAKE_LINE},
        },
        "subplots": {
            # Объем для визуального контроля СКО
            "Volume": {"type": "bar", "color": "#7f8c8d"}
        }
    }
    
    def __init__(self, config: dict) -> None:
        """Инициализация стратегии с единой конфигурацией"""
        super().__init__(config)
        
        # Создаем единую конфигурацию для всех компонентов
        self.skd_config = SkdConfig(
            # Метод и фильтры
            method=self.skd_method,
            vol_min=self.skd_vol_min,
            min_vol_ratio_prev=self.skd_min_vol_ratio_prev,
            min_vol_ratio_next=self.skd_min_vol_ratio_next,
            
            # Альтернативные методы
            percentile_lookback=self.skd_percentile_lookback,
            percentile=self.skd_percentile,
            rvol_lookback=self.skd_rvol_lookback,
            rvol_threshold=self.skd_rvol_threshold,
            zscore_lookback=self.skd_zscore_lookback,
            zscore_threshold=self.skd_zscore_threshold,
            stoch_lookback=self.skd_stoch_lookback,
            stoch_threshold=self.skd_stoch_threshold,
            
            # Сигналы
            signal_lag_bars=self.skd_signal_lag_bars,
            min_distance_mode=self.skd_min_distance_mode,
            min_distance_pct=self.skd_min_distance_pct,
            atr_period=self.skd_atr_period,
            min_distance_atr_mult=self.skd_min_distance_atr_mult,
            filter_ttl_bars=self.skd_filter_ttl_bars,
            
            # Визуализация
            level_ttl=self.skd_level_ttl,
            max_slots=self.skd_max_slots,
            slot_cooldown_bars=self.skd_slot_cooldown_bars,
            expire_opposite_on_new=self.skd_expire_opposite_on_new,
            same_side_latest_only=self.skd_same_side_latest_only,
            
            # Торговля
            trade_enabled=self.skd_trade_enabled,
            max_open_trades=self.skd_max_open_trades,
            trade_expire_bars=self.skd_trade_expire_bars,
        )
        
        # Инициализация компонентов с единой конфигурацией
        self.indicator = ClaudeSkdIndicator(self.skd_config)
        self.visualizer = ClaudeSkdVisualizer(self.skd_config)
        self.trade_manager = SkdTradeManager(self.skd_config)
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Вычисление индикаторов и подготовка визуализации.
        Единая точка обработки без дублирования.
        """
        # 1. Вычисляем СКО индикатор
        df = self.indicator.compute(dataframe)
        
        # 2. Обрабатываем торговые сигналы
        df = self.trade_manager.process_trades(df)
        
        # 3. Готовим визуализацию линий
        df = self.visualizer.render_levels(df)
        
        # 4. Добавляем визуальные маркеры для торговых сигналов
        df = self._add_trade_visualization(df)
        
        # Логирование статистики
        self.logger.info(
            f"[Claude-SKD] Найдено СКО: {self.indicator.stats.get('total_skd_found')} | "
            f"Создано уровней: {self.indicator.stats.get('levels_created')} | "
            f"Отфильтровано по соотношению: {self.indicator.stats.get('levels_filtered_ratio')} | "
            f"Отфильтровано по дистанции: {self.indicator.stats.get('levels_filtered_distance')} | "
            f"Сделок: {self.trade_manager.stats.get('total_trades')} | "
            f"Win Rate: {self.trade_manager.stats.get('win_rate'):.1f}%"
        )
        
        return df
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Определение точек входа на основе торговых сигналов.
        Поддержка разнонаправленных сделок.
        """
        # Long сигналы
        dataframe.loc[
            (dataframe["trade_enter_long"] == 1),
            "enter_long"
        ] = 1
        
        # Short сигналы
        dataframe.loc[
            (dataframe["trade_enter_short"] == 1),
            "enter_short"
        ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Определение точек выхода на основе стопов и тейков.
        Управление выходами через trade_manager.
        """
        # Long выходы
        dataframe.loc[
            (dataframe["trade_exit_long"] == 1),
            "exit_long"
        ] = 1
        
        # Short выходы
        dataframe.loc[
            (dataframe["trade_exit_short"] == 1),
            "exit_short"
        ] = 1
        
        return dataframe
    
    def _add_trade_visualization(self, df: DataFrame) -> DataFrame:
        """
        Добавление визуальных маркеров для торговых сигналов.
        Линии стопов и тейков для активных сделок.
        """
        # Маркеры входов
        df["entry_long_signal"] = np.where(
            df["trade_enter_long"] == 1,
            df["skd_buy_price"],
            np.nan
        )
        df["entry_short_signal"] = np.where(
            df["trade_enter_short"] == 1,
            df["skd_sell_price"],
            np.nan
        )
        
        # Маркеры выходов
        df["exit_long_signal"] = np.where(
            df["trade_exit_long"] == 1,
            df["close"],
            np.nan
        )
        df["exit_short_signal"] = np.where(
            df["trade_exit_short"] == 1,
            df["close"],
            np.nan
        )
        
        # Линии стопов и тейков (продлеваем их визуально)
        df["stop_loss_long"] = self._extend_line(df["trade_stop_long"], periods=5)
        df["stop_loss_short"] = self._extend_line(df["trade_stop_short"], periods=5)
        df["take_profit_long"] = self._extend_line(df["trade_take_long"], periods=5)
        df["take_profit_short"] = self._extend_line(df["trade_take_short"], periods=5)
        
        return df
    
    @staticmethod
    def _extend_line(series: DataFrame, periods: int = 5) -> DataFrame:
        """
        Продление линии на N периодов вперед для визуализации.
        Помогает видеть уровни стопов и тейков на графике.
        """
        result = series.copy()
        for i in range(len(series)):
            if not np.isnan(series.iloc[i]):
                # Продлеваем значение на N свечей вперед
                end_idx = min(i + periods, len(series))
                for j in range(i, end_idx):
                    if np.isnan(result.iloc[j]):
                        result.iloc[j] = series.iloc[i]
        return result
    
    def custom_exit(self, pair: str, trade, current_time, current_rate,
                    current_profit, **kwargs) -> str | None:
        """
        Кастомная логика выхода для точного контроля стопов и тейков.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if dataframe.empty:
            return None
        
        last_candle = dataframe.iloc[-1]
        
        # Для long позиций
        if trade.is_open and trade.is_long:
            stop_price = last_candle.get("trade_stop_long")
            take_price = last_candle.get("trade_take_long")
            
            if not np.isnan(stop_price) and current_rate <= stop_price:
                return "stop_loss_hit"
            if not np.isnan(take_price) and current_rate >= take_price:
                return "take_profit_hit"
        
        # Для short позиций
        elif trade.is_open and trade.is_short:
            stop_price = last_candle.get("trade_stop_short")
            take_price = last_candle.get("trade_take_short")
            
            if not np.isnan(stop_price) and current_rate >= stop_price:
                return "stop_loss_hit"
            if not np.isnan(take_price) and current_rate <= take_price:
                return "take_profit_hit"
        
        return None